import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context


class MultitaskLSTMCNNAttention(nn.Module):

    def __init__(
        self,
        input_dim: int = 7,
        cnn_channels: int = 32,
        lstm_hidden: int = 64,
        num_lstm_layers: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout_rate if num_lstm_layers > 1 and dropout_rate > 0 else 0.0,
            batch_first=True,
        )

        self.attention = Attention(lstm_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    # x: (batch, seq_len, features)
    def forward(self, x):

        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        shared = self.attention(x)

        class_logits = self.classifier(shared)
        intensity = self.regressor(shared).squeeze(-1)

        return class_logits, intensity