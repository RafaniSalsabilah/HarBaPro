from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from catboost import CatBoostRegressor
from datetime import timedelta
import traceback

app = Flask(__name__)
CORS(app)

MODEL_DIR = "models"

# --- Load semua model dan metrics ---
try:
    lstm_path = os.path.join(MODEL_DIR, "best_lstm_model_2.h5")
    cb_path = os.path.join(MODEL_DIR, "best_catboost_model_2.cbm")
    metrics_path = os.path.join(MODEL_DIR, "hasil_evaluasi_2.json")

    lstm_model = load_model(lstm_path, compile=False)

    cat_model = CatBoostRegressor()
    cat_model.load_model(cb_path)

    metrics = {}
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except:
        print("⚠️ Metrics JSON tidak ditemukan atau rusak. Menggunakan default.")

    try:
        hidden_extractor = Model(inputs=lstm_model.input, outputs=lstm_model.layers[-2].output)
    except Exception:
        hidden_extractor = Model(inputs=lstm_model.input, outputs=lstm_model.layers[-1].output)

    print("✅ Semua model dan metrics berhasil dimuat.")

except Exception as e:
    print(f"⚠️ Gagal memuat model/metrics: {e}")
    traceback.print_exc()
    lstm_model, cat_model, hidden_extractor, metrics = None, None, None, None

# -------------------------------
# Fungsi Normalisasi Manual
# -------------------------------
def minmax_manual(x, min_val=None, max_val=None, feature_range=(0.1, 1)):
    min_x = min_val if min_val is not None else x.min()
    max_x = max_val if max_val is not None else x.max()
    scale = feature_range[1] - feature_range[0]
    return feature_range[0] + ((x - min_x) / (max_x - min_x)) * scale, min_x, max_x

def inverse_minmax_manual(value, min_val, max_val, feature_range=(0.1, 1)):
    return min_val + (value - feature_range[0]) * (max_val - min_val) / (feature_range[1] - feature_range[0])

# -------------------------------
# Util: pembulatan & status Naik/Turun/Stabil
# -------------------------------
def round_to_thousands(x):
    try:
        return int(round(float(x) / 1000.0) * 1000)
    except Exception:
        return 0

def hitung_status_berbasis_rupiah(hist_rounded, pred_rounded):
    status = []
    for i in range(len(pred_rounded)):
        if i == 0:
            prev = hist_rounded[-1] if len(hist_rounded) > 0 else pred_rounded[0]
        else:
            prev = pred_rounded[i - 1]

        curr = pred_rounded[i]
        if curr > prev:
            status.append("Naik")
        elif curr < prev:
            status.append("Turun")
            status.append("Turun")
        else:
            status.append("Stabil")
    return status

# -------------------------------
# Preprocessing
# -------------------------------
def konversi_ke_numerik(df, kolom_list):
    for kolom in kolom_list:
        df[kolom] = df[kolom].astype(str).str.replace(",", ".", regex=False)
        extracted = df[kolom].str.extract(r"(-?\d+\.?\d*)", expand=False)
        df[kolom] = pd.to_numeric(extracted, errors="coerce")
    return df

def distribusi_produksi_hanya_di_hari_panen(df):
    df["BULAN_TAHUN"] = df["Waktu"].dt.to_period("M")
    produksi_bulanan = (
        df.dropna(subset=["Produksi"])
        .drop_duplicates("BULAN_TAHUN")[["BULAN_TAHUN", "Produksi"]]
        .copy()
    )
    produksi_bulanan["Produksi"] = produksi_bulanan["Produksi"] * 100
    df = df.merge(produksi_bulanan, on="BULAN_TAHUN", how="left", suffixes=("", "_bulanan"))
    df["is_panen"] = df["Musim"].astype(str).str.contains("Panen", case=False, na=False)
    hari_panen_per_bulan = df[df["is_panen"]].groupby("BULAN_TAHUN").size().rename("Hari_Panen")
    df = df.merge(hari_panen_per_bulan, on="BULAN_TAHUN", how="left")
    df["Produksi"] = df.apply(
        lambda row: row["Produksi_bulanan"] / row["Hari_Panen"]
        if row["is_panen"] and row["Hari_Panen"] > 0
        else 0,
        axis=1,
    )
    df.drop(columns=["Produksi_bulanan", "is_panen", "Hari_Panen", "BULAN_TAHUN"], inplace=True)
    return df

def distribusi_bulanan_ke_harian(df, kolom_list):
    df["BULAN_TAHUN"] = df["Waktu"].dt.to_period("M")
    for kolom in kolom_list:
        nilai_bulanan = (
            df.dropna(subset=[kolom])
            .drop_duplicates("BULAN_TAHUN")[["BULAN_TAHUN", kolom]]
        )
        df = df.drop(columns=[kolom], errors="ignore")
        df = df.merge(nilai_bulanan, on="BULAN_TAHUN", how="left")
    df.drop(columns=["BULAN_TAHUN"], inplace=True)
    return df

def isi_otomatis_musim(df, kolom_musim="Musim"):
    musim_terakhir = None
    musim_isi = []
    for val in df[kolom_musim]:
        if pd.notna(val) and str(val).strip() != "":
            musim_terakhir = val
        musim_isi.append(musim_terakhir)
    df[kolom_musim] = musim_isi
    return df

def bersihkan_curah_hujan(df, kolom="Curah_Hujan"):
    df[kolom] = df[kolom].astype(str).str.replace(",", ".", regex=False)
    extracted = df[kolom].str.extract(r"(-?\d+\.?\d*)", expand=False)
    df[kolom] = pd.to_numeric(extracted, errors="coerce")
    df[kolom] = df[kolom].replace(8888, np.nan)
    df[kolom] = df[kolom].interpolate(method="linear", limit_direction="both")
    df[kolom] = df[kolom].fillna(df[kolom].mean())
    return df

def bersihkan_harga(df, kolom="Harga"):
    df[kolom] = df[kolom].astype(str).str.replace(",", ".", regex=False)
    extracted = df[kolom].str.extract(r"(-?\d+\.?\d*)", expand=False)
    df[kolom] = pd.to_numeric(extracted, errors="coerce")
    df[kolom] = df[kolom].replace(0, np.nan)
    df[kolom] = df[kolom].interpolate(method="linear", limit_direction="both")
    df[kolom] = df[kolom].fillna(df[kolom].mean())
    return df

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return send_from_directory(".", "indexxx.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"status":"error","message": "Tidak ada file yang diunggah."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status":"error","message": "Nama file kosong."}), 400

    if not file.filename.lower().endswith(".xlsx"):
        return jsonify({"status":"error","message": "Format file harus .xlsx"}), 400

    if lstm_model is None or cat_model is None:
        return jsonify({"status":"error","message": "Model belum dimuat di server."}), 500

    try:
        df = pd.read_excel(file)
        df["Waktu"] = pd.to_datetime(df["Waktu"])
        df = distribusi_bulanan_ke_harian(df, ["Inflasi", "IHK"])
        df = isi_otomatis_musim(df, "Musim")
        df = bersihkan_harga(df, "Harga")
        df = bersihkan_curah_hujan(df, "Curah_Hujan")
        df = distribusi_produksi_hanya_di_hari_panen(df)

        fitur_lstm = ["Harga", "Produksi", "Curah_Hujan", "Inflasi", "IHK"]
        fitur_kat = ["Musim"]
        df = konversi_ke_numerik(df, fitur_lstm)
        df = df.dropna(subset=fitur_lstm + fitur_kat + ["Harga"]).reset_index(drop=True)

        if df.empty:
            return jsonify({"status":"error","message": "Data tidak memiliki baris cukup setelah pembersihan."}), 400

        harga_hist_raw = df["Harga"].tolist()
        waktu_hist = df["Waktu"].dt.date.astype(str).tolist()

        df_scaled = df.copy()
        min_max_dict = {}
        for col in fitur_lstm:
            df_scaled[col], min_val, max_val = minmax_manual(df[col])
            min_max_dict[col] = (min_val, max_val)

        horizon = 60
        window_size = 24
        scaled_rows = df_scaled[fitur_lstm].values.tolist()
        harga_pred_real = []
        last_row_real = df[fitur_lstm].iloc[-1].values
        musim_last = str(df["Musim"].iloc[-1])
        last_date = df["Waktu"].max()
        np.random.seed(42)

        for i in range(horizon):
            input_window = np.array(scaled_rows[-window_size:])
            x_input = np.expand_dims(input_window, axis=0)

            hidden_feat = hidden_extractor.predict(x_input, verbose=0)
            hidden_vec = hidden_feat[0, -1] if hidden_feat.ndim == 3 else np.ravel(hidden_feat)

            df_cb = pd.DataFrame([hidden_vec])
            df_cb["Musim"] = musim_last
            cat_pred = float(np.ravel(cat_model.predict(df_cb))[0])

            final_pred_scaled = cat_pred
            harga_real_pred = inverse_minmax_manual(final_pred_scaled, *min_max_dict["Harga"])

            MAPE = 0.06003469
            random_range = MAPE * 0.5
            trend_range = MAPE * 0.3

            random_fluct = np.random.uniform(-random_range, trend_range) * last_row_real[0]
            trend = np.random.uniform(-trend_range, random_range) * last_row_real[0]

            harga_real_noisy = harga_real_pred + random_fluct + trend
            harga_pred_real.append(harga_real_noisy)

            new_row = last_row_real.copy()
            new_row[0] = harga_real_noisy
            new_row_scaled = []
            for j, col in enumerate(fitur_lstm):
                val_scaled, _, _ = minmax_manual(pd.Series([new_row[j]]), *min_max_dict[col])
                new_row_scaled.append(float(val_scaled))
            scaled_rows.append(new_row_scaled)
            last_row_real = new_row

        pred_dates = [(last_date + timedelta(days=i + 1)).date().isoformat() for i in range(horizon)]
        combined_dates = waktu_hist + pred_dates

        harga_hist_rounded = [round_to_thousands(p) for p in harga_hist_raw]
        harga_pred_rounded = [round_to_thousands(p) for p in harga_pred_real]

        pred_status = hitung_status_berbasis_rupiah(harga_hist_rounded, harga_pred_rounded)
        combined_prices_rounded = harga_hist_rounded + harga_pred_rounded

        return jsonify({
            "status": "success",
            "horizon": horizon,
            "hist_dates": waktu_hist,
            "hist_prices": harga_hist_rounded,
            "pred_dates": pred_dates,
            "pred_prices": harga_pred_rounded,
            "pred_status": pred_status,
            "combined_dates": combined_dates,
            "combined_prices": combined_prices_rounded
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status":"error","message": str(e)}), 500