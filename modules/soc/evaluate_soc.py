
# Single source of truth for SoC test-set metrics (Table 6).
#
# B1 fix: previously, at least three code paths computed SoC RMSE/MAE/MAPE independently
# (run_complete_pipeline.py, lstm_cnn_attention_soc.evaluate_soc_model, and
# adaptive_ensemble.evaluate_ensemble), against two different, inconsistent target scales
# (raw 0-100 % vs an orphaned [0,1] dataset). That produced numbers that could not be
# reconciled with each other or with the paper (RMSE~73 vs RMSE~0.088). This script is the
# only place Table 6 should be generated from: it loads the real dataset (now consistently
# scaled to [0,1] by modules/soc/data/preprocess_real_data.py), inverse-transforms both
# predictions and targets back to % SoC through the single soc_scale.json definition, and
# reports metrics through the one shared calculate_regression_metrics function.

import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from shared.dataset_loader import get_dataset_loader
from shared.train_utils import calculate_regression_metrics
from modules.soc.soc_scale import load_soc_scale, inverse_scale_soc, assert_unit_scale
from modules.soc.models.lstm_cnn_attention_soc import LSTMCNNAttentionSoC

SOC_DATA_DIR = project_root / "modules" / "soc" / "data"
SOC_MODEL_DIR = project_root / "modules" / "soc" / "models"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_lstm_cnn_attention_soc(device: torch.device) -> Optional[LSTMCNNAttentionSoC]:
    model_path = SOC_MODEL_DIR / "lstm_cnn_attention_soc.pth"
    if not model_path.exists():
        print(f"warning: {model_path} not found, skipping this model")
        return None

    model = LSTMCNNAttentionSoC(
        input_dim=3, cnn_channels=64, lstm_hidden=128, num_lstm_layers=2, dropout=0.2
    )
    state_dict = torch.load(model_path, map_location=device)
    # save_model_checkpoint() wraps the state dict in a checkpoint dict; train_soc_model()
    # saves the raw state dict directly. Support both.
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_model(model, X_test: np.ndarray, y_test_unit: np.ndarray,
                    scale: Dict[str, float], device: torch.device,
                    batch_size: int = 256) -> Dict[str, float]:
    assert_unit_scale(y_test_unit, "y_test")

    preds_unit = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_x = torch.tensor(X_test[i:i + batch_size], dtype=torch.float32).to(device)
            preds_unit.append(model(batch_x).cpu().numpy())
    preds_unit = np.concatenate(preds_unit)

    assert_unit_scale(preds_unit, "predictions")

    # Report in % SoC via the one inverse transform (roadmap B1 step 2), not the raw
    # [0,1] training scale.
    y_test_pct = inverse_scale_soc(y_test_unit, scale)
    preds_pct = inverse_scale_soc(preds_unit, scale)

    return calculate_regression_metrics(y_test_pct, preds_pct)


def main():
    device = get_device()
    print(f"Using device: {device}")

    scale = load_soc_scale()
    print(f"SoC scale: {scale}")

    dataset_loader = get_dataset_loader()
    _, _, X_test, _, _, y_test = dataset_loader.load_soc_dataset()
    print(f"Test set: {X_test.shape[0]} samples, window {X_test.shape[1]}x{X_test.shape[2]}")

    rows = []

    lstm_cnn_model = load_lstm_cnn_attention_soc(device)
    if lstm_cnn_model is not None:
        metrics = evaluate_model(lstm_cnn_model, X_test, y_test, scale, device)
        rows.append({
            "model": "LSTM-CNN-Attention",
            "rmse_pct_soc": metrics["rmse"],
            "mae_pct_soc": metrics["mae"],
            "mape_pct": metrics["mape"],
            "n_test_samples": len(y_test),
        })
        print(f"\nLSTM-CNN-Attention SoC")
        print(f"  RMSE: {metrics['rmse']:.4f} % SoC")
        print(f"  MAE:  {metrics['mae']:.4f} % SoC")
        print(f"  MAPE: {metrics['mape']:.2f} %")

    if not rows:
        print("\nNo trained SoC models found - nothing to report.")
        return None

    table6 = pd.DataFrame(rows)
    output_path = SOC_MODEL_DIR / "table6.csv"
    table6.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    return table6


if __name__ == "__main__":
    main()
