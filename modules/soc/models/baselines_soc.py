
# B3: same-dataset SoC baseline ladder (Linear Regression -> MLP -> Chemali-style single
# LSTM), sitting between the non-learned Coulomb Counting floor (coulomb_counting.py) and
# the deployed LSTM-CNN-Attention model.
#
# Every learned baseline here implements the same SoCBaselineModel interface
# (fit/predict/save/load) so evaluate_soc.py can iterate BASELINE_REGISTRY generically -
# no per-model branching in the evaluation code. Coulomb Counting is intentionally NOT
# part of this interface (see coulomb_counting.py's module docstring for why).
#
# LinearRegression and the MLP share the same flattened-window input (flatten_windows),
# so the only difference between those two rungs is model capacity, not feature
# engineering. The Chemali-style LSTM operates on the real [50, 3] sequence, per
# Chemali, Kollmeyer, Preindl, Emadi (IEEE TIE, 2018) - a plain single-layer LSTM, no
# CNN front-end, no attention (roadmap literature ref. L-B1) - deliberately simpler than
# the deployed model's 2-layer LSTM.

import os
import pickle
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from shared.train_utils import set_seed


def flatten_windows(X: np.ndarray) -> np.ndarray:
    """[N, seq_len, n_features] -> [N, seq_len * n_features]."""
    return X.reshape(X.shape[0], -1)


class SoCBaselineModel(ABC):
    """Common interface for every learned SoC baseline: fit/predict/save/load, so
    evaluate_soc.py never needs model-specific logic. predict() always returns unit
    [0,1]-scale SoC, matching the deployed model's Sigmoid-head convention."""

    name: str = "SoCBaselineModel"
    checkpoint_filename: str = "baseline.pkl"

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "SoCBaselineModel":
        ...


class LinearRegressionSoC(SoCBaselineModel):
    """Ordinary least squares on flattened windows - the simplest learned rung."""

    name = "Linear Regression"
    checkpoint_filename = "linreg_soc_baseline.pkl"

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
        self.model.fit(flatten_windows(X_train), y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(flatten_windows(X))
        return np.clip(preds, 0.0, 1.0)  # OLS has no output bound, unlike the Sigmoid models

    def save(self, path: Path) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: Path) -> "LinearRegressionSoC":
        instance = cls()
        with open(path, "rb") as f:
            instance.model = pickle.load(f)
        return instance


class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPBaselineSoC(SoCBaselineModel):
    """2-layer MLP on the same flattened windows as LinearRegressionSoC - isolates the
    effect of model capacity/nonlinearity from feature engineering."""

    name = "MLP"
    checkpoint_filename = "mlp_soc_baseline.pth"

    def __init__(self, input_dim: int = 150, hidden_dim: int = 64, device: Optional[torch.device] = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cpu")
        self.model = _MLPNet(input_dim, hidden_dim).to(self.device)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            lr=1e-3, batch_size=256, epochs=50, patience=7) -> None:
        Xf_train = flatten_windows(X_train)
        train_loader = DataLoader(
            TensorDataset(torch.tensor(Xf_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True,
        )
        has_val = X_val is not None and y_val is not None
        if has_val:
            Xf_val = flatten_windows(X_val)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        best_rmse, wait = float("inf"), 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

            if not has_val:
                continue

            self.model.eval()
            with torch.no_grad():
                preds = self.model(torch.tensor(Xf_val, dtype=torch.float32).to(self.device)).cpu().numpy()
            vrmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))

            if vrmse < best_rmse:
                best_rmse, wait = vrmse, 0
            else:
                wait += 1
                if wait >= patience:
                    break

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(
                torch.tensor(flatten_windows(X), dtype=torch.float32).to(self.device)
            ).cpu().numpy()
        return preds

    def save(self, path: Path) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path: Path, input_dim: int = 150, hidden_dim: int = 64) -> "MLPBaselineSoC":
        instance = cls(input_dim=input_dim, hidden_dim=hidden_dim)
        instance.model.load_state_dict(torch.load(path, map_location=instance.device))
        instance.model.eval()
        return instance


class _ChemaliLSTMNet(nn.Module):
    """Plain single-layer LSTM regressor, per Chemali et al. (2018) - no CNN, no
    attention, unlike the deployed 2-layer LSTM-CNN-Attention model."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


class ChemaliLSTMBaselineSoC(SoCBaselineModel):
    name = "Chemali-LSTM"
    checkpoint_filename = "chemali_lstm_soc_baseline.pth"

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, device: Optional[torch.device] = None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device or torch.device("cpu")
        self.model = _ChemaliLSTMNet(input_dim, hidden_dim).to(self.device)

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            lr=1e-3, batch_size=256, epochs=50, patience=7) -> None:
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32)),
            batch_size=batch_size, shuffle=True,
        )
        has_val = X_val is not None and y_val is not None

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        best_rmse, wait = float("inf"), 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            if not has_val:
                continue

            self.model.eval()
            with torch.no_grad():
                preds = self.model(torch.tensor(X_val, dtype=torch.float32).to(self.device)).cpu().numpy()
            vrmse = float(np.sqrt(np.mean((preds - y_val) ** 2)))

            if vrmse < best_rmse:
                best_rmse, wait = vrmse, 0
            else:
                wait += 1
                if wait >= patience:
                    break

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        return preds

    def save(self, path: Path) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    @classmethod
    def load(cls, path: Path, input_dim: int = 3, hidden_dim: int = 64) -> "ChemaliLSTMBaselineSoC":
        instance = cls(input_dim=input_dim, hidden_dim=hidden_dim)
        instance.model.load_state_dict(torch.load(path, map_location=instance.device))
        instance.model.eval()
        return instance


# The registry evaluate_soc.py iterates generically - add a new learned baseline here
# (one class + one entry) without touching evaluate_soc.py.
BASELINE_REGISTRY = [
    LinearRegressionSoC,
    MLPBaselineSoC,
    ChemaliLSTMBaselineSoC,
]


def train_and_save_all_baselines(model_dir: Path = None, seed: int = 42) -> None:
    """The real, full-training entry point - NOT run as part of lightweight
    verification. Fits every registered baseline on the full training set and saves its
    checkpoint into modules/soc/models/. `seed` is a reproducibility override for
    multi-seed experiments (SV-1); it does not change any model/training logic."""
    from shared.dataset_loader import get_dataset_loader

    set_seed(seed)

    model_dir = model_dir or (Path(__file__).parent)
    dataset_loader = get_dataset_loader()
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_loader.load_soc_dataset()

    for cls in BASELINE_REGISTRY:
        print(f"Training {cls.name}...")
        instance = cls()
        instance.fit(X_train, y_train, X_val, y_val)
        save_path = model_dir / cls.checkpoint_filename
        instance.save(save_path)
        print(f"  saved: {save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and save all SoC baseline models")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed override, for multi-seed experiments (SV-1).")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save checkpoints into (default: this file's own "
                             "directory, i.e. modules/soc/models/). Use a seed-specific "
                             "directory to run multiple seeds without overwriting each other.")
    args = parser.parse_args()
    train_and_save_all_baselines(
        model_dir=Path(args.output_dir) if args.output_dir else None, seed=args.seed,
    )
