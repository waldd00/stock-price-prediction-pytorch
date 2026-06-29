import torch.nn as nn

_RNN = {"lstm": nn.LSTM, "gru": nn.GRU, "rnn": nn.RNN}


class RNNRegressor(nn.Module):
    def __init__(self, kind="lstm", input_dim=1, hidden=32, layers=2, output_dim=1):
        super().__init__()
        if kind not in _RNN:
            raise ValueError("kind must be rnn, lstm or gru")
        self.kind = kind
        self.rnn = _RNN[kind](input_dim, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def make_model(kind, input_dim, hidden=32, layers=2):
    return RNNRegressor(kind, input_dim=input_dim, hidden=hidden, layers=layers)
