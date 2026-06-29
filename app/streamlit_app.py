import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import fetch_prices, build_dataset
from src.model import make_model
from src.engine import train_model, metrics, predict_price, directional_accuracy, set_seed
from src.forecast import recursive_forecast

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "outputs", "models")

st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("Hisse Fiyat Tahmini: RNN / LSTM / GRU")

device = "cuda" if torch.cuda.is_available() else "cpu"


def try_load_pretrained(kind, n_feat, ticker, start, end, lookback, hidden, layers, device):
    config_path = os.path.join(MODELS_DIR, "config.json")
    weights_path = os.path.join(MODELS_DIR, f"{kind}.pt")
    if not os.path.exists(config_path) or not os.path.exists(weights_path):
        return None
    with open(config_path) as f:
        cfg = json.load(f)
    matches = (
        cfg.get("ticker", "").upper() == ticker.upper()
        and cfg.get("start") == start
        and cfg.get("end") == end
        and cfg.get("lookback") == lookback
        and cfg.get("hidden") == hidden
        and cfg.get("layers") == layers
    )
    if not matches:
        return None
    model = make_model(kind, input_dim=n_feat, hidden=hidden, layers=layers)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

with st.sidebar:
    st.header("Ayarlar")
    ticker = st.text_input("Ticker", "AMZN").upper().strip()
    start = st.text_input("Başlangıç", "2010-01-01")
    end = st.text_input("Bitiş", "2023-01-01")
    kind = st.selectbox("Model", ["gru", "lstm", "rnn"])
    lookback = st.slider("Lookback", 10, 60, 20, step=5)
    epochs = st.slider("Epochs", 50, 400, 200, step=50)
    horizon = st.slider("Forecast (gün)", 5, 60, 30, step=5)
    st.subheader("Model mimarisi")
    hidden = st.slider("Hidden size", 8, 128, 32, step=8)
    layers = st.slider("Layer sayısı", 1, 3, 2)
    lr = st.select_slider("Learning rate", [0.1, 0.05, 0.01, 0.005, 0.001], value=0.01)
    run = st.button("Eğit ve Tahmin Et", type="primary")


@st.cache_data(show_spinner=False)
def load_data(ticker, start, end, lookback):
    df = fetch_prices(ticker, start, end)
    return build_dataset(df, lookback=lookback)


if not run:
    st.info("Soldan parametreleri seç ve Eğit ve Tahmin Et'e bas.")
    st.stop()

try:
    with st.spinner("Veri indiriliyor..."):
        data = load_data(ticker, start, end, lookback)

    min_rows = lookback + 60  # SMA50 penceresi + en az bir lookback penceresi
    if len(data["df"]) < min_rows:
        st.error(
            f"'{ticker}' için yeterli veri yok (yalnızca {len(data['df'])} satır, en az {min_rows} gerekiyor). "
            "Ticker'ı, tarih aralığını veya lookback değerini kontrol et."
        )
        st.stop()
    if len(data["X_tr"]) == 0 or len(data["X_val"]) == 0 or len(data["X_te"]) == 0:
        st.error("Veri seti train/val/test olarak bölünemiyor. Tarih aralığını genişlet.")
        st.stop()

    n_feat = len(data["features"])
    st.success(f"{ticker}: {len(data['df'])} gün, {n_feat} özellik")

    pretrained = try_load_pretrained(kind, n_feat, ticker, start, end, lookback, hidden, layers, device)
    if pretrained is not None:
        st.info(f"Önceden eğitilmiş {kind.upper()} modeli kullanılıyor (parametreler eşleşti).")
        model = pretrained
    else:
        with st.spinner(f"{kind.upper()} eğitiliyor..."):
            set_seed(42)
            model = make_model(kind, input_dim=n_feat, hidden=hidden, layers=layers)
            model, hist = train_model(model, data, epochs=epochs, lr=lr, device=device)

    y_true = data["yprice_te"]
    y_pred = predict_price(model, data["X_te"], data["anchor_te"], data["rscaler"], device)
    m = metrics(y_true, y_pred)
    da = directional_accuracy(y_true, y_pred, data["anchor_te"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{m['RMSE']:.2f}")
    c2.metric("MAE", f"{m['MAE']:.2f}")
    c3.metric("MAPE", f"{m['MAPE_%']:.2f}%")
    c4.metric("Yön doğruluğu", f"{da:.1f}%")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y_true.ravel(), label="Actual", lw=2)
    ax.plot(y_pred.ravel(), label=f"{kind.upper()} pred", alpha=0.8)
    ax.set_title(f"{ticker} test seti")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    with st.spinner("Forecast hesaplanıyor..."):
        future = recursive_forecast(model, data, lookback, steps=horizon, device=device)

    resid_std = float(np.std(y_true.ravel() - y_pred.ravel()))
    band = 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))

    hist_close = data["df"]["Close"].values[-120:]
    future_x = range(len(hist_close), len(hist_close) + horizon)
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(range(len(hist_close)), hist_close, label="History")
    ax2.plot(future_x, future, "--", label=f"{horizon} gün forecast")
    ax2.fill_between(future_x, future - band, future + band, alpha=0.2, label="~%95 belirsizlik bandı")
    ax2.set_title(f"{ticker} {kind.upper()} forecast")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

    st.caption("Eğitim amaçlıdır, yatırım tavsiyesi değildir.")

except ValueError as e:
    st.error(f"Geçersiz parametre veya ticker: {e}")
except Exception as e:
    st.error(
        f"Beklenmeyen bir hata oluştu ({type(e).__name__}): {e}\n\n"
        "Ticker'ın doğru olduğundan, tarih aralığının geçerli olduğundan ve "
        "internet bağlantısının çalıştığından emin ol."
    )
