import time
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _t(arr, device):
    return torch.from_numpy(np.asarray(arr, dtype="float32")).to(device)


def train_model(model, data, epochs=300, lr=0.01, patience=25, device="cpu"):
    model = model.to(device)
    X_tr, y_tr = _t(data["X_tr"], device), _t(data["y_tr"], device)
    X_val, y_val = _t(data["X_val"], device), _t(data["y_val"], device)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr_hist, val_hist = [], []
    best, best_state, wait = float("inf"), None, 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(X_tr), y_tr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = crit(model(X_val), y_val).item()
        tr_hist.append(loss.item())
        val_hist.append(v)
        if v < best - 1e-7:
            best = v
            best_state = {k: val.clone() for k, val in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"train": tr_hist, "val": val_hist, "time": time.time() - t0, "epochs": len(tr_hist)}


def metrics(true, pred):
    true, pred = np.asarray(true).ravel(), np.asarray(pred).ravel()
    nonzero = true != 0
    mape = float(np.mean(np.abs((true[nonzero] - pred[nonzero]) / true[nonzero])) * 100) if nonzero.any() else float("nan")
    return {
        "RMSE": float(np.sqrt(np.mean((true - pred) ** 2))),
        "MAE": float(np.mean(np.abs(true - pred))),
        "MAPE_%": mape,
    }


def directional_accuracy(y_true_price, y_pred_price, anchors):
    yt = np.asarray(y_true_price).ravel()
    yp = np.asarray(y_pred_price).ravel()
    a = np.asarray(anchors).ravel()
    true_dir = np.sign(yt - a)
    pred_dir = np.sign(yp - a)
    mask = true_dir != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(true_dir[mask] == pred_dir[mask]) * 100)


def predict_price(model, X, anchors, rscaler, device="cpu"):
    model.eval()
    with torch.no_grad():
        ret_s = model(_t(X, device)).cpu().numpy()
    ret = rscaler.inverse_transform(ret_s)
    return np.asarray(anchors).reshape(-1, 1) * np.exp(ret)
