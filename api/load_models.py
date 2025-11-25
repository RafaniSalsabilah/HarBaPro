# api/load_models.py
import os
import json
import traceback

# global placeholders
lstm_model = None
cat_model = None
hidden_extractor = None
metrics = {}

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def try_load_models():
    global lstm_model, cat_model, hidden_extractor, metrics
    if lstm_model is not None and cat_model is not None:
        return

    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras import Model
        from catboost import CatBoostRegressor
    except Exception as e:
        # dependencies not installed or incompatible
        print("⚠️ Tidak bisa import TensorFlow/CatBoost:", e)
        return

    try:
        lstm_path = os.path.join(MODEL_DIR, "best_lstm_model_2.h5")
        cb_path = os.path.join(MODEL_DIR, "best_catboost_model_2.cbm")
        metrics_path = os.path.join(MODEL_DIR, "hasil_evaluasi_2.json")

        lstm_model = load_model(lstm_path, compile=False)
        cat_model = CatBoostRegressor()
        cat_model.load_model(cb_path)

        try:
            hidden_extractor = Model(inputs=lstm_model.input, outputs=lstm_model.layers[-2].output)
        except Exception:
            hidden_extractor = Model(inputs=lstm_model.input, outputs=lstm_model.layers[-1].output)

        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {}
            print("⚠️ metrics JSON tidak ditemukan, menggunakan default.")

        print("✅ Model berhasil dimuat.")

    except Exception as e:
        print("⚠️ Gagal memuat model/metrics:", e)
        traceback.print_exc()
        # set to None to indicate failure
        lstm_model, cat_model, hidden_extractor, metrics = None, None, None, None

def get_models():
    # lazily load
    if lstm_model is None or cat_model is None:
        try_load_models()
    return lstm_model, cat_model, hidden_extractor, metrics
