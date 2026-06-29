import os
import sys
import json
import pickle
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data import fetch_prices, build_dataset
from src.model import make_model
from src.engine import train_model, metrics, predict_price, directional_accuracy, set_seed

CFG = dict(ticker="AMZN", start="2010-01-01", end="2023-01-01",
           lookback=20, hidden=32, layers=2, epochs=300, lr=0.01, patience=25)


def main():
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = fetch_prices(CFG["ticker"], CFG["start"], CFG["end"])
    data = build_dataset(df, lookback=CFG["lookback"])
    print(f"{CFG['ticker']}: {len(data['df'])} rows, features {data['features']}")
    y_true = data["yprice_te"]
    results = {}
    set_seed(42)
    for kind in ["rnn", "lstm", "gru"]:
        model = make_model(kind, input_dim=len(data["features"]), hidden=CFG["hidden"], layers=CFG["layers"])
        model, hist = train_model(model, data, epochs=CFG["epochs"], lr=CFG["lr"], patience=CFG["patience"], device=device)
        y_pred = predict_price(model, data["X_te"], data["anchor_te"], data["rscaler"], device)
        r = metrics(y_true, y_pred)
        r["DirAcc_%"] = round(directional_accuracy(y_true, y_pred, data["anchor_te"]), 2)
        r["Train_s"] = round(hist["time"], 2)
        results[kind.upper()] = r
        torch.save(model.state_dict(), f"outputs/models/{kind}.pt")
        print(kind.upper(), r)
    naive = data["anchor_te"]
    results["NAIVE"] = {**metrics(y_true, naive), "DirAcc_%": None, "Train_s": 0.0}
    print("NAIVE", results["NAIVE"])
    with open("outputs/models/scaler.pkl", "wb") as f:
        pickle.dump({"fscaler": data["fscaler"], "rscaler": data["rscaler"]}, f)
    with open("outputs/metrics/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("outputs/models/config.json", "w") as f:
        json.dump(CFG, f, indent=2)
    print("saved to outputs/")


if __name__ == "__main__":
    main()
