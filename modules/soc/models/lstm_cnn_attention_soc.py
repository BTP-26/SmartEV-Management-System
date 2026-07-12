import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from modules.soc.soc_scale import load_soc_scale, inverse_scale_soc

class LSTMCNNAttentionSoC(nn.Module):
    def __init__(self, input_dim=3, cnn_channels=64,
                 lstm_hidden=128, num_lstm_layers=2, dropout=0.2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        self.attn_fc = nn.Linear(lstm_hidden, 1)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        c = self.cnn(x.permute(0, 2, 1))
        c = c.permute(0, 2, 1)

        h, _ = self.lstm(c)

        scores  = self.attn_fc(h)
        weights = torch.softmax(scores, dim=1)
        context = (weights * h).sum(dim=1)

        return self.head(context).squeeze(-1)


def train_soc_model(
    model, X_train, y_train, X_val, y_val,
    lr=1e-3, batch_size=256, epochs=50, patience=7,
    device=None, save_path="modules/soc/models/lstm_cnn_attention_soc.pth"
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = model.to(device)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
                              batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)),
                              batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_rmse, wait = float("inf"), 0
    history = {"train_loss": [], "val_rmse": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        model.train()
        tloss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tloss += loss.item() * len(xb)
        tloss /= len(train_loader.dataset)

        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb.to(device)).cpu().numpy())
                tgts.append(yb.numpy())
        preds, tgts = np.concatenate(preds), np.concatenate(tgts)
        vrmse = np.sqrt(np.mean((preds - tgts) ** 2))
        vmae  = np.mean(np.abs(preds - tgts))

        history["train_loss"].append(tloss)
        history["val_rmse"].append(vrmse)
        history["val_mae"].append(vmae)

        if vrmse < best_rmse:
            best_rmse, wait = vrmse, 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {tloss:.6f} | Val RMSE: {vrmse:.4f} | Val MAE: {vmae:.4f}")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nBest Val RMSE: {best_rmse:.4f} | Saved: {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history


def evaluate_soc_model(model, X_test, y_test, label="LSTM+CNN+Attention SOC", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test).to(device)).cpu().numpy()

    # Report in % SoC via the one inverse transform (roadmap B1 step 2), not the raw
    # [0,1] training scale, so this matches evaluate_soc.py's Table 6 numbers.
    scale = load_soc_scale()
    preds_pct = inverse_scale_soc(preds, scale)
    y_test_pct = inverse_scale_soc(y_test, scale)

    rmse = np.sqrt(np.mean((preds_pct - y_test_pct) ** 2))
    mae  = np.mean(np.abs(preds_pct - y_test_pct))
    mape = np.mean(np.abs((preds_pct - y_test_pct) / (y_test_pct + 1e-8))) * 100
    print(f"\n{label}")
    print(f"Test RMSE: {rmse:.4f} % SoC")
    print(f"Test MAE: {mae:.4f} % SoC")
    print(f"Test MAPE: {mape:.2f}%")
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape), "preds": preds_pct}


if __name__ == "__main__":
    # Load the real, [0,1]-scaled dataset through the shared DatasetLoader (same source as
    # the deployed pipeline and evaluate_soc.py). This previously loaded an orphaned
    # *_soc.npy convention that no preprocessing script in the repo produces, so running
    # this module directly always crashed. modules/train/train_soc.py is the canonical
    # training entry point; this block is a module self-demo.
    from shared.dataset_loader import get_dataset_loader
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset_loader().load_soc_dataset()

    model = LSTMCNNAttentionSoC()
    model, history = train_soc_model(model, X_train, y_train, X_val, y_val)
    evaluate_soc_model(model, X_test, y_test)