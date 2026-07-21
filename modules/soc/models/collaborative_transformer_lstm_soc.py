
# L-B3: Collaborative Transformer+LSTM SoC estimator - an engineering interpretation, NOT
# a verified reproduction of "Collaborative framework of Transformer and LSTM for enhanced
# state-of-charge estimation in lithium-ion batteries" (Energy, 2025). That paper is
# paywalled; this file is built only from what could be independently confirmed across
# multiple searches:
#   1. A Transformer captures long-term/global dependencies.
#   2. An LSTM refines the estimate using short-term/local dynamics.
#   3. The two are used SEQUENTIALLY - the Transformer's output feeds into the LSTM's
#      process - not as two independent parallel branches merged afterward (which is
#      already what adaptive_ensemble.py's TransformerSoCModel is, and explicitly not
#      what this paper describes).
# One detail from a single, less-corroborated search summary ("reconstruct shorter local
# sequences") was deliberately NOT used here, since it could not be independently
# confirmed by other sources - inventing a specific mechanism attributed to an
# unverifiable claim would misrepresent this as a faithful reproduction. Instead, the
# simplest standard mechanism satisfying the three CONFIRMED properties above is used:
# the Transformer's pooled context initializes the LSTM's initial hidden state, so the
# LSTM's entire pass over the window is conditioned on that global context from the
# start. This is a well-known encoder-conditions-decoder pattern, not attributed to the
# source paper's unknown exact mechanism.
#
# Self-contained rather than reusing adaptive_ensemble.py's TransformerSoCModel: repo-wide
# grep confirms it is the only other Transformer anywhere in this codebase, but it exposes
# no intermediate representation (only a final scalar via its own regressor), and reaching
# into its internal submodules from here would create an implicit, undocumented
# dependency on its internals rather than a designed reuse interface - a fragile form of
# "reuse" that a future unrelated refactor of that class could silently break. The
# self-contained encoder below is a small fraction of that class's size (no regression
# head of its own), since its only job is producing a seed context vector.

import os
import sys

import numpy as np
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class _TransformerContextEncoder(nn.Module):
    """Processes the full window, returns a pooled global-context vector - no regression
    head of its own, unlike adaptive_ensemble.py's TransformerSoCModel; this exists only
    to seed CollaborativeTransformerLSTMSoC's LSTM stage below."""

    def __init__(self, input_dim: int = 3, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        h = self.input_projection(x) * (self.d_model ** 0.5)
        h = self.transformer(h)
        return torch.mean(h, dim=1)  # (batch, d_model) - pooled global context


class CollaborativeTransformerLSTMSoC(nn.Module):
    """L-B3: sequential Transformer-to-LSTM SoC estimator. See module docstring for
    exactly which properties are confirmed-from-literature vs. this implementation's own
    choice of mechanism where the source paper's exact method could not be verified."""

    def __init__(self, input_dim: int = 3, d_model: int = 64, nhead: int = 4,
                 transformer_layers: int = 2, lstm_hidden: int = 64,
                 lstm_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

        self.context_encoder = _TransformerContextEncoder(
            input_dim=input_dim, d_model=d_model, nhead=nhead,
            num_layers=transformer_layers, dropout=dropout,
        )
        # The cascade point: projects the Transformer's pooled context into the LSTM's
        # initial (h_0, c_0), so the LSTM's entire pass is conditioned on it from the
        # start - this is what makes the design genuinely sequential rather than a
        # late-merge ensemble of two independent predictions.
        self.context_to_h0 = nn.Linear(d_model, lstm_hidden * lstm_layers)
        self.context_to_c0 = nn.Linear(d_model, lstm_hidden * lstm_layers)

        self.lstm = nn.LSTM(input_dim, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(lstm_hidden, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        context = self.context_encoder(x)  # (batch, d_model)

        # (batch, layers*hidden) -> (batch, layers, hidden) -> (layers, batch, hidden):
        # the middle reshape is a plain view (batch is already the leading dim, so memory
        # layout matches), the permute is a real axis reorder - collapsing these into one
        # .view() would silently scramble batch/layer data instead of erroring.
        h0 = self.context_to_h0(context).view(
            batch_size, self.lstm_layers, self.lstm_hidden).permute(1, 0, 2).contiguous()
        c0 = self.context_to_c0(context).view(
            batch_size, self.lstm_layers, self.lstm_hidden).permute(1, 0, 2).contiguous()

        _, (h_n, _) = self.lstm(x, (h0, c0))
        return self.head(h_n[-1]).squeeze(-1)


def train_collaborative_model(save_path: str, seed: int = 42, sample_size: int = 20000,
                               epochs: int = 5):
    """Verification-scale entry point, matching the same capped-subset/few-epoch default
    convention already established by physics_informed_soc.py's create_physics_informed_model()
    and adaptive_ensemble.py's train_adaptive_ensemble() for this class of experimental
    (non-canonical) SoC model files - NOT the full-scale training path.

    `save_path` is REQUIRED, not optional: train_soc_model() (lstm_cnn_attention_soc.py)
    unconditionally does `os.makedirs(os.path.dirname(save_path), ...)` and
    `torch.save(...)` whenever validation RMSE improves, then unconditionally reloads
    from that same path at the end - there is no built-in way to skip saving. An earlier
    draft of this function defaulted save_path to None assuming that would skip saving;
    checking train_soc_model()'s actual source (not assumed) showed that would crash
    immediately. Callers must explicitly choose a path - scratch for verification runs,
    a real modules/soc/models/ path only for an intentional, approved save."""
    from shared.dataset_loader import get_dataset_loader
    from shared.train_utils import set_seed
    from modules.soc.models.lstm_cnn_attention_soc import train_soc_model

    set_seed(seed)
    dataset_loader = get_dataset_loader()
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_loader.load_soc_dataset()

    idx = np.random.choice(len(X_train), min(sample_size, len(X_train)), replace=False)
    X_train, y_train = X_train[idx], y_train[idx]

    model = CollaborativeTransformerLSTMSoC()
    model, history = train_soc_model(
        model, X_train, y_train, X_val, y_val,
        lr=1e-3, batch_size=64, epochs=epochs, patience=2,
        device=torch.device("cpu"), save_path=save_path,
    )
    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Verification-scale run of the L-B3 collaborative model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, required=True,
                         help="Checkpoint output path (required - see train_collaborative_model docstring).")
    args = parser.parse_args()
    train_collaborative_model(save_path=args.save_path, seed=args.seed)
    print("L-B3 verification-scale run complete.")
