"""Base-paper braking model reproduction (Yang et al., IJCAS 2024).

Faithful to the paper's *parallel* design: an LSTM branch with self-attention
runs in parallel with a CNN branch (temporal conv + max-pool); the two feature
vectors are concatenated and passed through two dense layers to the output.

Used as the "[Yang 2024] base architecture" comparator in the sweep. Adapted
only where our task requires it (multi-channel IMU input, binary head); those
are the sole deviations from the paper.
"""
import torch
import torch.nn as nn


class _SelfAttention(nn.Module):
    """Scaled dot-product self-attention over LSTM hidden states -> context vec."""

    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, h):                      # h: (B, T, dim)
        attn = torch.softmax((self.q(h) @ self.k(h).transpose(1, 2)) / self.scale, dim=-1)
        return (attn @ self.v(h)).mean(dim=1)  # (B, dim)


class YangLSTMCNNAttention(nn.Module):
    def __init__(self, input_dim=7, lstm_hidden=64, lstm_layers=2, cnn_filters=64,
                 kernel_size=3, dense_units=64, num_classes=2, dropout=0.2):
        super().__init__()
        # LSTM branch + self-attention
        self.lstm = nn.LSTM(input_dim, lstm_hidden, lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.attn = _SelfAttention(lstm_hidden)
        # CNN branch: temporal conv -> ReLU -> global max-pool
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_filters, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )
        # concat-fuse -> two dense layers -> output (paper's dense1/dense2/softmax)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + cnn_filters, dense_units), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dense_units, dense_units), nn.ReLU(),
            nn.Linear(dense_units, num_classes),
        )

    def forward(self, x):                      # x: (B, T, C)
        lstm_feat = self.attn(self.lstm(x)[0])                 # (B, lstm_hidden)
        cnn_feat = self.cnn(x.transpose(1, 2)).squeeze(-1)     # (B, cnn_filters)
        logits = self.head(torch.cat([lstm_feat, cnn_feat], dim=1))
        # dummy intensity keeps the (logits, intensity) interface; Yang is single-task
        return logits, logits.new_zeros(logits.size(0))
