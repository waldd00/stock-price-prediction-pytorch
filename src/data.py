import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

FEATURES = ["ret", "close_sma10", "close_sma50", "rsi", "logvol"]
TARGET = "Close"


def fetch_prices(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def engineer(df):
    out = df.copy()
    out["ret"] = np.log(out["Close"]).diff()
    sma10 = out["Close"].rolling(10).mean()
    sma50 = out["Close"].rolling(50).mean()
    out["close_sma10"] = out["Close"] / sma10 - 1.0
    out["close_sma50"] = out["Close"] / sma50 - 1.0
    delta = out["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    out["rsi"] = (100 - 100 / (1 + rs)) / 100.0
    out["logvol"] = np.log(out["Volume"] + 1.0)
    return out.dropna()


def build_dataset(df, lookback=20, train_pct=0.70, val_pct=0.15, features=FEATURES, target=TARGET):
    d = engineer(df)
    if len(d) < lookback + 10:
        raise ValueError(
            f"Not enough data ({len(d)} rows). The ticker or date range may be invalid, "
            f"or there isn't enough history for the lookback ({lookback})."
        )
    feat = d[features].values.astype("float32")
    ret = d["ret"].values.astype("float32")
    close = d[target].values.astype("float32")
    n = len(d)
    L = lookback
    i_tr = int(n * train_pct)
    i_val = int(n * (train_pct + val_pct))

    fscaler = StandardScaler()
    feat_s = np.empty_like(feat)
    feat_s[:i_tr] = fscaler.fit_transform(feat[:i_tr])
    feat_s[i_tr:] = fscaler.transform(feat[i_tr:])

    X, y_ret, anchor, y_price = [], [], [], []
    for i in range(n - L):
        X.append(feat_s[i:i + L])
        y_ret.append(ret[i + L])
        anchor.append(close[i + L - 1])
        y_price.append(close[i + L])
    X = np.asarray(X, dtype="float32")
    y_ret = np.asarray(y_ret, dtype="float32").reshape(-1, 1)
    anchor = np.asarray(anchor, dtype="float32").reshape(-1, 1)
    y_price = np.asarray(y_price, dtype="float32").reshape(-1, 1)

    tr_end = i_tr - L
    val_end = i_val - L

    rscaler = StandardScaler()
    y_ret_s = np.empty_like(y_ret)
    y_ret_s[:tr_end] = rscaler.fit_transform(y_ret[:tr_end])
    y_ret_s[tr_end:] = rscaler.transform(y_ret[tr_end:])

    return {
        "X_tr": X[:tr_end], "y_tr": y_ret_s[:tr_end],
        "X_val": X[tr_end:val_end], "y_val": y_ret_s[tr_end:val_end],
        "X_te": X[val_end:], "y_te": y_ret_s[val_end:],
        "anchor_tr": anchor[:tr_end], "anchor_val": anchor[tr_end:val_end], "anchor_te": anchor[val_end:],
        "yprice_tr": y_price[:tr_end], "yprice_val": y_price[tr_end:val_end], "yprice_te": y_price[val_end:],
        "fscaler": fscaler, "rscaler": rscaler,
        "df": d, "features": features, "target": target,
    }
