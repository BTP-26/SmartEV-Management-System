
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
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from shared.dataset_loader import get_dataset_loader
from shared.train_utils import calculate_regression_metrics
from modules.soc.soc_scale import load_soc_scale, inverse_scale_soc, assert_unit_scale
from modules.soc.models.lstm_cnn_attention_soc import LSTMCNNAttentionSoC
from modules.soc.models.baselines_soc import BASELINE_REGISTRY
from modules.soc.models.coulomb_counting import evaluate_coulomb_counting

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

    metrics = calculate_regression_metrics(y_test_pct, preds_pct)
    # B5: additive keys (existing readers only read rmse/mae/mape by name) - lets callers
    # reuse these predictions for per-segment/per-SoC-level reporting without re-running
    # inference.
    metrics["preds_pct"] = preds_pct
    metrics["y_true_pct"] = y_test_pct
    return metrics


def build_baseline_ladder_rows(X_test: np.ndarray, y_test: np.ndarray,
                                scale: Dict[str, float]) -> Tuple[list, Dict[str, np.ndarray]]:
    """B3: the same-dataset baseline ladder (Coulomb Counting -> BASELINE_REGISTRY's
    learned baselines). Generic over BASELINE_REGISTRY - adding a new learned baseline in
    baselines_soc.py requires no changes here. Coulomb Counting is the one deliberate
    exception (see coulomb_counting.py docstring): it has no checkpoint and is evaluated
    over full real cycles, not the windowed X_test, so its `protocol` differs from every
    other row - callers must not treat n_test_samples as comparable across protocols (and
    it has no windowed predictions to return - excluded from the second return value).
    Returns ([] rows for any learned baseline whose checkpoint doesn't exist yet,
    {model_name: preds_pct} for models that were evaluated - B5: reused for per-segment/
    per-SoC-level reporting without re-running inference)."""
    rows = []
    predictions_by_model: Dict[str, np.ndarray] = {}

    cc_metrics = evaluate_coulomb_counting()
    rows.append({
        "model": "Coulomb Counting",
        "rmse_pct_soc": cc_metrics["rmse"],
        "mae_pct_soc": cc_metrics["mae"],
        "mape_pct": cc_metrics["mape"],
        "n_test_samples": cc_metrics["n_samples"],
        "protocol": "full-cycle",
    })

    for cls in BASELINE_REGISTRY:
        checkpoint_path = SOC_MODEL_DIR / cls.checkpoint_filename
        if not checkpoint_path.exists():
            print(f"warning: {checkpoint_path} not found, skipping {cls.name}")
            continue

        model = cls.load(checkpoint_path)
        preds_unit = model.predict(X_test)
        assert_unit_scale(preds_unit, f"{cls.name} predictions")

        preds_pct = inverse_scale_soc(preds_unit, scale)
        y_test_pct = inverse_scale_soc(y_test, scale)
        metrics = calculate_regression_metrics(y_test_pct, preds_pct)

        rows.append({
            "model": cls.name,
            "rmse_pct_soc": metrics["rmse"],
            "mae_pct_soc": metrics["mae"],
            "mape_pct": metrics["mape"],
            "n_test_samples": len(y_test),
            "protocol": "windowed-test-split",
        })
        predictions_by_model[cls.name] = preds_pct

    return rows, predictions_by_model


def compute_segment_and_soclevel_breakdown(model_name: str, y_true_pct: np.ndarray,
                                            preds_pct: np.ndarray, segment_type: np.ndarray,
                                            n_soc_bins: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """B5 (roadmap step 3): per-segment (charge/drive/regen) and error-vs-SoC-level
    breakdowns for one already-evaluated model, reusing its predictions rather than
    re-running inference. Small-n segments/bins (e.g. a rare "regen" slice, or a sparsely
    populated SoC bin) can have unstable or undefined R^2 (see
    calculate_regression_metrics's n<2/zero-variance guard) - not solved here, just not
    hidden."""
    segment_rows = []
    for seg in sorted(set(segment_type.tolist())):
        mask = segment_type == seg
        metrics = calculate_regression_metrics(y_true_pct[mask], preds_pct[mask])
        segment_rows.append({
            "model": model_name, "segment_type": seg,
            "rmse_pct_soc": metrics["rmse"], "mae_pct_soc": metrics["mae"], "r2": metrics["r2"],
            "n_samples": int(mask.sum()),
        })

    bin_edges = np.linspace(0.0, 100.0, n_soc_bins + 1)
    bin_idx = np.clip(np.digitize(y_true_pct, bin_edges) - 1, 0, n_soc_bins - 1)
    soc_level_rows = []
    for b in range(n_soc_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        metrics = calculate_regression_metrics(y_true_pct[mask], preds_pct[mask])
        soc_level_rows.append({
            "model": model_name, "soc_bin": f"{bin_edges[b]:.0f}-{bin_edges[b + 1]:.0f}%",
            "rmse_pct_soc": metrics["rmse"], "mae_pct_soc": metrics["mae"],
            "n_samples": int(mask.sum()),
        })

    return pd.DataFrame(segment_rows), pd.DataFrame(soc_level_rows)


def build_per_segment_and_soclevel_tables(predictions_by_model: Dict[str, np.ndarray],
                                           y_true_pct: np.ndarray, segment_type_test: np.ndarray,
                                           n_soc_bins: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """B5 deliverable: per-segment error table + error-vs-SoC-level table, across every
    model in `predictions_by_model` (deployed model + any evaluated baselines). Coulomb
    Counting is not included - it has no windowed predictions (see
    build_baseline_ladder_rows)."""
    segment_frames, soc_level_frames = [], []
    for model_name, preds_pct in predictions_by_model.items():
        seg_df, soc_df = compute_segment_and_soclevel_breakdown(
            model_name, y_true_pct, preds_pct, segment_type_test, n_soc_bins
        )
        segment_frames.append(seg_df)
        soc_level_frames.append(soc_df)
    return pd.concat(segment_frames, ignore_index=True), pd.concat(soc_level_frames, ignore_index=True)


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

    # B3: same-dataset baseline ladder (replaces the paper's cross-dataset Table 7).
    print("\n--- Baseline ladder (Table 7) ---")
    ladder_rows, _predictions_by_model = build_baseline_ladder_rows(X_test, y_test, scale)
    # Reuse the LSTM-CNN-Attention row already computed above, if available, so Table 7
    # includes the full ladder up to the deployed model without evaluating it twice.
    for row in rows:
        ladder_rows.append({**row, "protocol": "windowed-test-split"})

    table7 = pd.DataFrame(ladder_rows)
    table7_path = SOC_MODEL_DIR / "table7.csv"
    table7.to_csv(table7_path, index=False)
    print(f"Saved: {table7_path}")

    return table6


if __name__ == "__main__":
    main()
