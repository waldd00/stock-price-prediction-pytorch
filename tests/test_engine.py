import numpy as np
import torch

from src.engine import directional_accuracy, metrics, set_seed, train_model
from src.model import make_model


def test_metrics_basic():
    true = np.array([100.0, 200.0])
    pred = np.array([110.0, 190.0])
    m = metrics(true, pred)
    assert abs(m["MAE"] - 10.0) < 1e-6
    assert m["RMSE"] > 0
    expected_mape = ((10 / 100 + 10 / 200) / 2) * 100
    assert abs(m["MAPE_%"] - expected_mape) < 1e-4


def test_metrics_zero_true_does_not_crash():
    true = np.array([0.0, 1.0])
    pred = np.array([0.1, 1.1])
    m = metrics(true, pred)
    assert not np.isnan(m["RMSE"])
    assert not np.isnan(m["MAPE_%"])


def test_metrics_all_zero_true_mape_is_nan():
    true = np.array([0.0, 0.0])
    pred = np.array([1.0, 2.0])
    m = metrics(true, pred)
    assert np.isnan(m["MAPE_%"])


def test_directional_accuracy_masks_zero_diff():
    true_price = np.array([101.0, 100.0, 102.0])
    pred_price = np.array([102.0, 100.0, 101.0])
    anchors = np.array([100.0, 100.0, 100.0])
    da = directional_accuracy(true_price, pred_price, anchors)
    assert da == 100.0


def test_set_seed_makes_init_reproducible():
    set_seed(42)
    m1 = make_model("lstm", input_dim=3, hidden=8, layers=1)
    set_seed(42)
    m2 = make_model("lstm", input_dim=3, hidden=8, layers=1)
    p1 = next(m1.parameters())
    p2 = next(m2.parameters())
    assert torch.allclose(p1, p2)


def test_train_model_reduces_training_loss():
    set_seed(0)
    X = np.random.randn(50, 10, 3).astype("float32")
    y = np.random.randn(50, 1).astype("float32")
    data = {"X_tr": X, "y_tr": y, "X_val": X[:10], "y_val": y[:10]}
    model = make_model("gru", input_dim=3, hidden=8, layers=1)
    model, hist = train_model(model, data, epochs=30, lr=0.01, patience=30, device="cpu")
    assert hist["train"][-1] < hist["train"][0]
