"""The single SoC scale contract shared by preprocessing and every evaluation site.

SoC is recorded as a physical percentage (README: "SoC [%]"), so the [0,1] scale used by
the model's Sigmoid head is a fixed affine transform (divide by 100), not a data-driven
min-max fit. Using the observed min/max of a given split would make 0/1 mean different
physical SoC levels across subsets and would leak split-specific range into the scale.

modules/soc/data/preprocess_real_data.py (producer) writes soc_scale.json from the constants
below; every consumer that needs to report metrics in %SoC (evaluate_soc.py,
lstm_cnn_attention_soc.py, adaptive_ensemble.py, run_complete_pipeline.py) reads it back
through load_soc_scale()/inverse_scale_soc() here, so there is exactly one place the scale is
defined and exactly one place it is inverted.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np

SOC_MIN_PERCENT = 0.0
SOC_MAX_PERCENT = 100.0
DEFAULT_SCALE = {"soc_min_percent": SOC_MIN_PERCENT, "soc_max_percent": SOC_MAX_PERCENT}

SOC_SCALE_PATH = Path(__file__).parent / "data" / "soc_scale.json"


def scale_soc(soc_percent: np.ndarray) -> np.ndarray:
    """Map raw %SoC (0-100) to the [0,1] range consumed by the model's Sigmoid head."""
    return (soc_percent - SOC_MIN_PERCENT) / (SOC_MAX_PERCENT - SOC_MIN_PERCENT)


def load_soc_scale() -> Dict[str, float]:
    """Load the soc_min/soc_max metadata written by preprocess_real_data.py.

    Falls back to the documented default (0-100%) if the metadata file is missing, since
    that reflects the fixed physical definition of SoC, not a data-dependent fit.
    """
    if SOC_SCALE_PATH.exists():
        with open(SOC_SCALE_PATH) as f:
            return json.load(f)
    print(f"warning: {SOC_SCALE_PATH} not found, assuming default 0-100% scale")
    return DEFAULT_SCALE


def inverse_scale_soc(soc_unit: np.ndarray, scale: Dict[str, float] = None) -> np.ndarray:
    """Inverse of scale_soc: map [0,1] model output/target back to %SoC.

    Uses the fixed constants by default; pass an explicit `scale` (as returned by
    load_soc_scale()) to invert against the persisted metadata instead.
    """
    soc_min = SOC_MIN_PERCENT if scale is None else scale["soc_min_percent"]
    soc_max = SOC_MAX_PERCENT if scale is None else scale["soc_max_percent"]
    return soc_unit * (soc_max - soc_min) + soc_min


def assert_unit_scale(y: np.ndarray, name: str):
    """Guard against the exact class of scale bug B1 fixes: if targets/predictions land
    far outside [0,1], the model and data are on mismatched scales again."""
    lo, hi = float(np.min(y)), float(np.max(y))
    if lo < -0.05 or hi > 1.05:
        raise ValueError(
            f"{name} is outside the expected [0,1] range (got [{lo:.3f}, {hi:.3f}]). "
            "This usually means the model checkpoint was trained on a different SoC "
            "scale than the currently-loaded dataset - retrain before evaluating."
        )
