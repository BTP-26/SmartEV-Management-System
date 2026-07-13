
# Traditional Coulomb Counting baseline for SoC estimation.
#
# B3: previously this evaluated against a NASA-schema CSV convention
# (Current_measured/Time columns) that doesn't exist in this repo's actual dataset - no
# .csv files are produced anywhere in the pipeline - and never compared against real SoC
# at all (it compared CC-with-clean-current vs CC-with-injected-noise, not CC vs ground
# truth). capacity_ah also defaulted to 2.0, an order of magnitude off the real 240 Ah
# pack (reviewer_checklist.md BL-3). This rewrite evaluates CC against the real Mendeley
# cycles and the real pack capacity.
#
# CC is evaluated continuously over each real cycle (list_cycle_folders/load_cycle,
# reused from battery_rls_identification.py), seeded with that cycle's own true initial
# SoC, against the true SoC at every synchronized sample - not on the windowed
# train/test split the learned models use. This is intentional, not an oversight: CC has
# no learned parameters, so there is no train/test distinction for it, and the windowed
# .npy dataset doesn't retain per-window cycle identity to reconstruct a session-aligned
# test set anyway (same root gap as reviewer_checklist.md EV-1, to be closed under B5).
# See evaluate_soc.py's table7.csv `protocol` column, which makes this explicit.

import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from modules.soc.models.battery_rls_identification import list_cycle_folders, load_cycle
from modules.soc.data.preprocess_real_data import CAPACITY_AH


def coulomb_counting_soc(current: np.ndarray, time: np.ndarray, capacity_ah: float = CAPACITY_AH,
                          soc_init: float = None) -> np.ndarray:
    """Integrate current over time to estimate SoC (fraction, [0,1]). Positive current is
    charge, negative is discharge (this dataset's convention - see README.pdf)."""
    dt        = np.diff(time, prepend=time[0])
    dt        = np.clip(dt, 0, 60)
    charge_ah = current * dt / 3600.0
    is_discharge = np.mean(current) < 0
    if soc_init is None:
        soc_init = 1.0 if is_discharge else 0.0
    soc = soc_init + np.cumsum(charge_ah) / capacity_ah
    return np.clip(soc, 0.0, 1.0)


def evaluate_coulomb_counting(capacity_ah: float = CAPACITY_AH) -> Dict[str, float]:
    """Evaluate CC against real SoC across every readable Drive/Charge cycle."""
    all_true, all_pred = [], []
    n_cycles = 0

    for folder in list_cycle_folders():
        cycle = load_cycle(folder)
        if cycle is None:
            continue

        soc_true_unit = cycle["soc"] / 100.0  # cycle["soc"] is raw %, see load_cycle()
        soc_pred_unit = coulomb_counting_soc(
            cycle["curr"], cycle["t"], capacity_ah=capacity_ah, soc_init=soc_true_unit[0]
        )

        all_true.extend(soc_true_unit.tolist())
        all_pred.extend(soc_pred_unit.tolist())
        n_cycles += 1

    all_true = np.array(all_true) * 100.0  # report in % SoC
    all_pred = np.array(all_pred) * 100.0

    errors = np.abs(all_true - all_pred)
    rmse = float(np.sqrt(np.mean((all_true - all_pred) ** 2)))
    mae  = float(np.mean(errors))
    mape = float(np.mean(np.abs((all_true - all_pred) / (all_true + 1e-8))) * 100)

    print("Coulomb Counting Baseline (real cycles, real capacity):")
    print(f"  Cycles evaluated : {n_cycles}")
    print(f"  Capacity (Ah)    : {capacity_ah}")
    print(f"  Samples          : {len(all_true)}")
    print(f"  RMSE             : {rmse:.4f} % SoC")
    print(f"  MAE              : {mae:.4f} % SoC")
    print(f"  MAPE             : {mape:.2f} %")

    return {"rmse": rmse, "mae": mae, "mape": mape, "n_samples": len(all_true), "n_cycles": n_cycles}


if __name__ == "__main__":
    evaluate_coulomb_counting()
