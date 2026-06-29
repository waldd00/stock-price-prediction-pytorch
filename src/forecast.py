import numpy as np
import pandas as pd
import torch

from .data import engineer


def recursive_forecast(model, data, lookback, steps=30, device="cpu"):
    raw = data["df"]
    features = data["features"]
    fscaler, rscaler = data["fscaler"], data["rscaler"]
    buf = lookback + 70
    closes = list(raw["Close"].values[-buf:].astype("float64"))
    vols = list(raw["Volume"].values[-buf:].astype("float64"))
    model.eval()
    preds = []
    for _ in range(steps):
        tmp = pd.DataFrame({"Close": closes, "Volume": vols})
        feat_df = engineer(tmp)
        window = feat_df[features].values[-lookback:].astype("float32")
        ws = fscaler.transform(window)
        x = torch.from_numpy(ws).float().unsqueeze(0).to(device)
        with torch.no_grad():
            ret_s = model(x).cpu().numpy()
        ret = float(rscaler.inverse_transform(ret_s)[0, 0])
        nxt = closes[-1] * np.exp(ret)
        preds.append(nxt)
        closes.append(nxt)
        vols.append(vols[-1])
    return np.array(preds)
