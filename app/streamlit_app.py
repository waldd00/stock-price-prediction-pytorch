import json
import os
import sys
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import fetch_prices, build_dataset
from src.model import make_model
from src.engine import train_model, metrics, predict_price, directional_accuracy, set_seed
from src.forecast import recursive_forecast

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(ROOT, "outputs", "models")

st.set_page_config(page_title="Stock Price Prediction", page_icon="📈", layout="wide")
st.title("📈 Stock Price Prediction: RNN / LSTM / GRU")
st.caption("Predicts the next-day **log-return** and reconstructs price as `price[t] * exp(return)` — not a raw-price copy of yesterday.")

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
    st.header("⚙️ Settings")
    ticker = st.text_input("Ticker", "AMZN", help="Any symbol supported by Yahoo Finance, e.g. AMZN, AAPL, MSFT.").upper().strip()
    start = st.text_input("Start date", "2010-01-01")
    end = st.text_input("End date", "2023-01-01")
    kind = st.selectbox("Model", ["gru", "lstm", "rnn"])
    lookback = st.slider("Lookback", 10, 60, 20, step=5, help="How many past days the model sees to predict the next one.")
    epochs = st.slider("Epochs", 50, 400, 200, step=50, help="Training stops early if validation loss stops improving.")
    horizon = st.slider("Forecast (days)", 5, 60, 30, step=5, help="How many days to forecast recursively into the future.")
    st.divider()
    st.subheader("🧠 Model architecture")
    hidden = st.slider("Hidden size", 8, 128, 32, step=8, help="Number of units in each recurrent layer.")
    layers = st.slider("Layers", 1, 3, 2, help="Number of stacked recurrent layers.")
    lr = st.select_slider("Learning rate", [0.1, 0.05, 0.01, 0.005, 0.001], value=0.01)
    st.divider()
    run = st.button("🚀 Train & Predict", type="primary", use_container_width=True)


@st.cache_data(show_spinner=False)
def load_data(ticker, start, end, lookback):
    df = fetch_prices(ticker, start, end)
    return build_dataset(df, lookback=lookback)


if not run:
    st.info("Set the parameters on the left and click **Train & Predict**.")
    with st.expander("ℹ️ About this project", expanded=True):
        st.markdown(
            """
Most beginner tutorials feed raw closing prices into an RNN and predict the next
raw price — on a trending series that mostly teaches the model to copy yesterday's
value forward. This project instead predicts the **next-day log-return**, which is
approximately stationary, and reconstructs price as `price[t+1] = price[t] * exp(return)`.

- Chronological train/val/test split — scalers fit on train only, no leakage.
- Early stopping on validation loss, seeded for reproducible runs.
- Hitting the defaults (AMZN, 2010–2023, lookback 20) loads **pretrained weights**
  instantly instead of retraining.
- Multi-step recursive forecast with a residual-based ~95% uncertainty band.

[Source on GitHub](https://github.com/waldd00/stock-price-prediction-pytorch) — not investment advice.
"""
        )
    st.stop()

try:
    with st.spinner("Downloading data..."):
        data = load_data(ticker, start, end, lookback)

    min_rows = lookback + 60  # SMA50 window + at least one lookback window
    if len(data["df"]) < min_rows:
        st.error(
            f"Not enough data for '{ticker}' (only {len(data['df'])} rows, need at least {min_rows}). "
            "Check the ticker, the date range, or lower the lookback value."
        )
        st.stop()
    if len(data["X_tr"]) == 0 or len(data["X_val"]) == 0 or len(data["X_te"]) == 0:
        st.error("The dataset can't be split into train/val/test. Widen the date range.")
        st.stop()

    n_feat = len(data["features"])
    st.success(f"{ticker}: {len(data['df'])} days, {n_feat} features")

    pretrained = try_load_pretrained(kind, n_feat, ticker, start, end, lookback, hidden, layers, device)
    if pretrained is not None:
        st.info(f"Using the pretrained {kind.upper()} model (parameters match).")
        model = pretrained
    else:
        with st.spinner(f"Training {kind.upper()}..."):
            set_seed(42)
            model = make_model(kind, input_dim=n_feat, hidden=hidden, layers=layers)
            model, hist = train_model(model, data, epochs=epochs, lr=lr, device=device)

    y_true = data["yprice_te"]
    y_pred = predict_price(model, data["X_te"], data["anchor_te"], data["rscaler"], device)
    m = metrics(y_true, y_pred)
    da = directional_accuracy(y_true, y_pred, data["anchor_te"])

    tab_results, tab_forecast, tab_about = st.tabs(["📊 Test Results", "🔮 Forecast", "ℹ️ About"])

    with tab_results:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📐 RMSE", f"{m['RMSE']:.2f}")
        c2.metric("📏 MAE", f"{m['MAE']:.2f}")
        c3.metric("📊 MAPE", f"{m['MAPE_%']:.2f}%")
        c4.metric("🎯 Directional accuracy", f"{da:.1f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_true.ravel(), name="Actual", line=dict(width=2)))
        fig.add_trace(go.Scatter(y=y_pred.ravel(), name=f"{kind.upper()} pred", line=dict(width=1.5)))
        fig.update_layout(
            title=f"{ticker} test set: actual vs predicted",
            xaxis_title="Test set day index", yaxis_title="Price",
            hovermode="x unified", template="plotly_white", height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_forecast:
        with st.spinner("Computing forecast..."):
            future = recursive_forecast(model, data, lookback, steps=horizon, device=device)

        resid_std = float(np.std(y_true.ravel() - y_pred.ravel()))
        band = 1.96 * resid_std * np.sqrt(np.arange(1, horizon + 1))

        hist_close = data["df"]["Close"].values[-120:]
        hist_x = list(range(len(hist_close)))
        future_x = list(range(len(hist_close), len(hist_close) + horizon))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=hist_x, y=hist_close, name="History", line=dict(width=2)))
        fig2.add_trace(go.Scatter(x=future_x, y=future, name=f"{horizon}-day forecast", line=dict(width=2, dash="dash")))
        fig2.add_trace(go.Scatter(
            x=future_x + future_x[::-1], y=list(future + band) + list((future - band)[::-1]),
            fill="toself", fillcolor="rgba(37, 99, 235, 0.15)", line=dict(width=0),
            name="~95% uncertainty band", hoverinfo="skip",
        ))
        fig2.update_layout(
            title=f"{ticker} {kind.upper()} forecast",
            xaxis_title="Day index", yaxis_title="Price",
            hovermode="x unified", template="plotly_white", height=450,
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("The band widens with √(days ahead) to reflect compounding forecast error in the recursive multi-step prediction.")

    with tab_about:
        st.markdown(
            """
### Why log-returns, not raw price?
Raw closing prices drift with the trend, so a model fed raw prices mostly learns to
copy yesterday's value forward — accurate-looking on a chart, but with no real signal,
and it extrapolates badly once the trend breaks. Log-returns are approximately
stationary, which is what these models actually need to learn something beyond
"yesterday's price."

### Pretrained model reuse
Hitting the default parameters (AMZN, 2010-01-01–2023-01-01, lookback 20, hidden 32,
2 layers) loads the pretrained weights from `outputs/models/` instead of retraining
from scratch. Change any of those and the app trains a fresh model.

### Honest results
The naive baseline (predict tomorrow = today) performs about as well as these models
on RMSE/MAE — expected for daily price data. Directional accuracy near 50% means no
edge beyond chance.

**For educational purposes only. Not investment advice.**

[Source code on GitHub](https://github.com/waldd00/stock-price-prediction-pytorch)
"""
        )

except ValueError as e:
    st.error(f"Invalid parameter or ticker: {e}")
except Exception as e:
    st.error(
        f"An unexpected error occurred ({type(e).__name__}): {e}\n\n"
        "Make sure the ticker is valid, the date range is correct, and "
        "your internet connection is working."
    )
