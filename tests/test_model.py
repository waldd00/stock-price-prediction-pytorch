import torch

from src.model import make_model


def test_output_shape_for_each_kind():
    for kind in ["rnn", "lstm", "gru"]:
        model = make_model(kind, input_dim=5, hidden=16, layers=1)
        x = torch.randn(4, 20, 5)
        out = model(x)
        assert out.shape == (4, 1)
