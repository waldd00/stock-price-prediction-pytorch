import numpy as np
import pandas as pd
import pytest

from src.data import FEATURES, build_dataset, engineer


def make_synthetic_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    close = 100 * np.exp(np.cumsum(returns))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=idx)


def test_engineer_no_nans_and_has_features():
    df = make_synthetic_df()
    out = engineer(df)
    assert not out[FEATURES].isna().any().any()
    for col in FEATURES:
        assert col in out.columns


def test_build_dataset_split_sizes_match():
    df = make_synthetic_df()
    lookback = 20
    data = build_dataset(df, lookback=lookback)
    n_windows = len(data["X_tr"]) + len(data["X_val"]) + len(data["X_te"])
    expected = len(data["df"]) - lookback
    assert n_windows == expected


def test_scaler_fit_only_on_train_slice():
    df = make_synthetic_df()
    lookback = 20
    train_pct = 0.70
    data = build_dataset(df, lookback=lookback, train_pct=train_pct)
    d = data["df"]
    i_tr = int(len(d) * train_pct)
    feat = d[FEATURES].values.astype("float32")
    expected_mean = feat[:i_tr].mean(axis=0)
    expected_std = feat[:i_tr].std(axis=0)
    assert np.allclose(data["fscaler"].mean_, expected_mean, atol=1e-4)
    assert np.allclose(data["fscaler"].scale_, expected_std, atol=1e-4)


def test_build_dataset_raises_on_too_short_data():
    df = make_synthetic_df(n=15)
    with pytest.raises(ValueError):
        build_dataset(df, lookback=20)
