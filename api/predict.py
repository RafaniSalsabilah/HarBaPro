# api/predict.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import timedelta
import traceback
import io
import os

from .load_models import get_models

app = FastAPI()

# allow all origins (Vercel frontend can call)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# helper functions (same logic as sebelumnya) - keep them local here
def minmax_manual(x, min_val=None, max_val=None, feature_range=(0.1, 1)):
    s = pd.Series(x).astype(float)
    min_x = min_val if min_val is not None else s.min()
    max_x = max_val if max_val is not None else s.max()
    scale = feature_range[1] - feature_range[0]
    if max_x == min_x:
        mid = feature_range[0] + scale / 2.0
        return pd.Series([mid] * len(s)), min_x, max_x
    scaled = feature_range[0] + ((s - min_x) / (max_x - min_x)) * scale
    if len(s) == 1:
        return float(scaled.iloc[0]), min_x, max_x
    return scaled, min_x, max_x

def inverse_minmax_manual(value, min_val, max_val, feature_range=(0.1, 1)):
    scale = feature_range[1] - feature_range[0]
    if max_val == min_val:
        return float(min_val)
    return float(min_val + (value - feature_range[0]) * (max_val - min_val) / scale)

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
        else:
            status.append("Stabil")
    return status

# reuse preprocessing helpers (you can factor these into another module if desired)
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

@app.get("/")
def home():
    index_path = os.path.join(os.path.dirname(__file__), "..", "indexxx.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"detail": "indexxx.html tidak ditemukan"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    lstm_model, cat_model, hidden_extractor, metrics = get_models()
    if lstm_model is None or cat_model is None:
        raise HTTPException(status_code=500, detail="Model belum dimuat di server.")

    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Format file harus .xlsx")

    try:
        content = await file.read()
        df = pd.read_excel(io.BytesIO(content))

        # pastikan kolom wajib ada
        required_cols = ["Waktu", "Harga", "Produksi", "Curah_Hujan", "Inflasi", "IHK", "Musim"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Kolom hilang: {missing}")

        df["Waktu"] = pd.to_datetime(df["Waktu"])

        # PREPROCESSING (pindah ke awal sebelum validasi)
        df = distribusi_bulanan_ke_harian(df, ["Inflasi", "IHK"])
        df = isi_otomatis_musim(df, "Musim")
        df = bersihkan_harga(df, "Harga")
        df = bersihkan_curah_hujan(df, "Curah_Hujan")
        df = distribusi_produksi_hanya_di_hari_panen(df)

        fitur_lstm = ["Harga", "Produksi", "Curah_Hujan", "Inflasi", "IHK"]

        df = konversi_ke_numerik(df, fitur_lstm)

        # PREDIKSI (tidak diubah)
        harga_hist_raw = df["Harga"].tolist()
        waktu_hist = df["Waktu"].dt.date.astype(str).tolist()

        df_scaled = df.copy()
        min_max_dict = {}
        for col in fitur_lstm:
            scaled_col, min_val, max_val = minmax_manual(df[col])
            df_scaled[col] = float(scaled_col) if isinstance(scaled_col, (float, int)) else scaled_col
            min_max_dict[col] = (min_val, max_val)

        horizon = 60
        window_size = 24
        scaled_rows = df_scaled[fitur_lstm].values.tolist()
        harga_pred_real = []
        last_row_real = df[fitur_lstm].iloc[-1].values
        musim_last = str(df["Musim"].iloc[-1])
        last_date = df["Waktu"].max()
        np.random.seed(42)

        for _ in range(horizon):
            x_input = np.expand_dims(np.array(scaled_rows[-window_size:]), axis=0)
            hidden_feat = hidden_extractor.predict(x_input, verbose=0)
            hidden_vec = hidden_feat[0, -1] if hidden_feat.ndim == 3 else np.ravel(hidden_feat)

            df_cb = pd.DataFrame([hidden_vec])
            df_cb["Musim"] = musim_last
            cat_pred = float(np.ravel(cat_model.predict(df_cb))[0])

            harga_real_pred = inverse_minmax_manual(cat_pred, *min_max_dict["Harga"])

            MAPE = 0.06003469
            random_range = MAPE * 0.5
            trend_range = MAPE * 0.3

            random_fluct = np.random.uniform(-random_range, trend_range) * last_row_real[0]
            trend = np.random.uniform(-trend_range, random_range) * last_row_real[0]

            harga_real_noisy = harga_real_pred + random_fluct + trend
            harga_pred_real.append(harga_real_noisy)

            new_row = last_row_real.copy()
            new_row[0] = harga_real_noisy

            new_scaled = []
            for j, col in enumerate(fitur_lstm):
                scaled_val, _, _ = minmax_manual(pd.Series([new_row[j]]), *min_max_dict[col])
                new_scaled.append(float(scaled_val))
            scaled_rows.append(new_scaled)
            last_row_real = new_row

        pred_dates = [(last_date + timedelta(days=i + 1)).date().isoformat() for i in range(horizon)]

        harga_hist_rounded = [round_to_thousands(p) for p in harga_hist_raw]
        harga_pred_rounded = [round_to_thousands(p) for p in harga_pred_real]
        pred_status = hitung_status_berbasis_rupiah(harga_hist_rounded, harga_pred_rounded)

        return JSONResponse({
            "status": "success",
            "horizon": horizon,
            "hist_dates": waktu_hist,
            "hist_prices": harga_hist_rounded,
            "pred_dates": pred_dates,
            "pred_prices": harga_pred_rounded,
            "pred_status": pred_status,
            "combined_dates": waktu_hist + pred_dates,
            "combined_prices": harga_hist_rounded + harga_pred_rounded
        })

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))