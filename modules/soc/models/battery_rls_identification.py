
# B2: Battery parameter identification from real pack-level data.
#
# `BatteryPhysicsParams` previously hard-coded 18650-cell values (3.7V nominal, 100Ah,
# 0.05 Ohm) while this dataset is a real EV pack of a few hundred volts. Rather than
# guess replacement constants, this script derives them from the Mendeley "Real-world
# electric vehicle data driving and charging" recordings:
#   - Pack capacity: taken from the dataset authors' own analysis code
#     (`_code/fig_6_11_12/fig_6_11_12.m`, line "Cap = 240" [Ah]) - not published in the
#     README, but it is the value the dataset's own paper uses for its C-rate figures.
#   - Nominal/min/max voltage: observed extrema across all readable Drive+Charge cycles.
#   - R0, R1, C1 (first-order Thevenin ECM): identified via recursive least squares
#     (RLS) on real current/voltage/SoC traces, one estimate per drive/charge cycle,
#     aggregated as mean +/- std across cycles (roadmap B2 step 2-3, L-B4).
#
# Chemistry and series/parallel cell configuration are not disclosed anywhere in the
# dataset (README.pdf or _code/*.m); this script does not fabricate them.
#
# ECM: V(t) = OCV(SoC(t)) - R0*I(t) - Vc(t),  dVc/dt = I/C1 - Vc/(R1*C1)
# Discretized (ZOH, alpha = exp(-Ts/(R1*C1))) and rewritten in terms of the OCV residual
# e[k] = V[k] - OCV(SoC[k]) gives a linear ARX(1,1) form suitable for RLS:
#   e[k] = a1*e[k-1] + b0*I[k] + b1*I[k-1]
#   a1 = alpha,  b0 = -R0,  b1 = alpha*R0 - R1*(1-alpha)
# which is inverted after fitting: R0 = -b0, R1 = (a1*R0 - b1)/(1-a1), tau = -Ts/ln(a1),
# C1 = tau/R1. This ARX<->ECM mapping is standard in battery system identification
# (e.g. Plett, "Extended Kalman filtering for battery management systems...", 2004).

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import interpolate

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from modules.soc.data.preprocess_real_data import load_ev_data, synchronize_data, DATA_DIR, CAPACITY_AH

RESAMPLED_MAX_SAMPLES = 8000   # cap per-cycle RLS length for tractable runtime
IDLE_CURRENT_PERCENTILE = 2.0  # bottom percentile of |I| per cycle, pooled for the OCV fit
N_OCV_BINS = 25
FORGETTING_FACTOR = 0.999      # near-1: this is a one-shot per-cycle batch fit, not
                                # online tracking, so minimal forgetting gives the most
                                # stable estimate (aggressive forgetting, e.g. 0.99,
                                # let recent low-excitation windows dominate and produced
                                # unstable/non-physical poles in testing)
CURRENT_SCALE_A = 100.0        # condition the regressor: e[k] is O(1-10 V) but I is
                                # O(10-800 A) at pack level; without this the RLS
                                # covariance update is poorly scaled and diverges
OUTPUT_PATH = Path(__file__).parent / "battery_params_identified.json"


def list_cycle_folders() -> List[str]:
    folders = []
    for subdir in ("Drive", "Charge"):
        d = os.path.join(DATA_DIR, subdir)
        if not os.path.exists(d):
            continue
        for folder in sorted(os.listdir(d)):
            path = os.path.join(d, folder)
            if os.path.isdir(path):
                folders.append(path)
    return folders


def load_cycle(folder_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Load and synchronize one cycle; returns None for unreadable/corrupt files
    (e.g. Drive/Folder16, which fails to open - see preprocess_real_data.py)."""
    try:
        ev_data = load_ev_data(folder_path)
        features, soc, t = synchronize_data(ev_data)
    except Exception as e:
        print(f"  skipping {folder_path}: {e}")
        return None
    if len(t) < 50:
        return None
    volt, curr = features[:, 0], features[:, 1]
    if len(t) > RESAMPLED_MAX_SAMPLES:
        volt, curr, soc, t = volt[:RESAMPLED_MAX_SAMPLES], curr[:RESAMPLED_MAX_SAMPLES], \
            soc[:RESAMPLED_MAX_SAMPLES], t[:RESAMPLED_MAX_SAMPLES]
    Ts = float(np.median(np.diff(t)))
    return {"volt": volt, "curr": curr, "soc": soc, "t": t, "Ts": Ts, "folder": folder_path}


def fit_global_ocv_curve(cycles: List[Dict[str, np.ndarray]]):
    """Pool near-idle (low |I|) samples across all cycles and fit a monotonic OCV(SoC)
    lookup. A true rest-test OCV curve isn't available in this field dataset, so
    per-cycle low-current points are used as a practical proxy (documented, not a
    fabricated chemistry-based curve)."""
    soc_pts, volt_pts = [], []
    for c in cycles:
        thresh = np.percentile(np.abs(c["curr"]), IDLE_CURRENT_PERCENTILE)
        idle_mask = np.abs(c["curr"]) <= thresh
        soc_pts.append(c["soc"][idle_mask])
        volt_pts.append(c["volt"][idle_mask])
    soc_pts = np.concatenate(soc_pts)
    volt_pts = np.concatenate(volt_pts)

    bin_edges = np.linspace(soc_pts.min(), soc_pts.max(), N_OCV_BINS + 1)
    bin_idx = np.clip(np.digitize(soc_pts, bin_edges) - 1, 0, N_OCV_BINS - 1)
    bin_centers, bin_means = [], []
    for b in range(N_OCV_BINS):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_centers.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
        bin_means.append(volt_pts[mask].mean())
    bin_centers = np.array(bin_centers)
    bin_means = np.maximum.accumulate(np.array(bin_means))  # enforce OCV monotonicity

    ocv_interp = interpolate.interp1d(
        bin_centers, bin_means, kind="linear", bounds_error=False,
        fill_value=(bin_means[0], bin_means[-1]),
    )
    return ocv_interp, (bin_centers, bin_means)


def rls_arx(e: np.ndarray, curr: np.ndarray, lam: float = FORGETTING_FACTOR,
            curr_scale: float = CURRENT_SCALE_A) -> Optional[Tuple[float, float, float]]:
    """Recursive least squares for e[k] = a1*e[k-1] + b0*I[k] + b1*I[k-1].

    Current is scaled to O(1-10) before the recursion (matching e's own scale) and the
    b coefficients are un-scaled afterward - without this the covariance update is
    dominated by the ~100x magnitude mismatch between volts and amps and the identified
    pole (a1) becomes unstable/non-physical."""
    n = len(e)
    if n < 20:
        return None
    curr_s = curr / curr_scale
    theta = np.zeros(3)
    P = 1e2 * np.eye(3)
    for k in range(1, n):
        phi = np.array([e[k - 1], curr_s[k], curr_s[k - 1]])
        y = e[k]
        Pphi = P @ phi
        denom = lam + phi @ Pphi
        gain = Pphi / denom
        err = y - phi @ theta
        theta = theta + gain * err
        P = (P - np.outer(gain, Pphi)) / lam
    a1, b0_s, b1_s = theta
    return (a1, b0_s / curr_scale, b1_s / curr_scale)


def arx_to_ecm(a1: float, b0: float, b1: float, Ts: float) -> Optional[Dict[str, float]]:
    if not (0.0 < a1 < 1.0):
        return None
    R0 = -b0
    denom = 1.0 - a1
    R1 = (a1 * R0 - b1) / denom
    if R0 <= 0 or R1 <= 0:
        return None
    tau = -Ts / np.log(a1)
    C1 = tau / R1
    if not np.isfinite(tau) or not np.isfinite(C1):
        return None
    return {"R0_ohm": R0, "R1_ohm": R1, "C1_farad": C1, "tau_s": tau, "Ts_s": Ts}


def run_identification() -> Dict:
    print("Loading real EV cycles for battery parameter identification...")
    folders = list_cycle_folders()
    cycles = []
    for folder in folders:
        c = load_cycle(folder)
        if c is not None:
            cycles.append(c)
    print(f"Loaded {len(cycles)}/{len(folders)} cycles "
          f"(Drive/Folder16 is a known-corrupt .mat, skipped upstream too).")

    # Pack-level voltage/current envelope, observed across all cycles
    all_volt = np.concatenate([c["volt"] for c in cycles])
    drive_curr = np.concatenate([c["curr"] for c in cycles if "Drive" in c["folder"]])
    charge_curr = np.concatenate([c["curr"] for c in cycles if "Charge" in c["folder"]])

    max_discharge_current = float(np.percentile(drive_curr[drive_curr > 0], 99.9)) if np.any(drive_curr > 0) else None
    max_charge_current = float(np.percentile(-charge_curr[charge_curr < 0], 99.9)) if np.any(charge_curr < 0) else None

    print("Fitting global OCV(SoC) curve from pooled near-idle samples...")
    ocv_fn, ocv_curve = fit_global_ocv_curve(cycles)

    print(f"Running RLS per cycle (forgetting factor={FORGETTING_FACTOR})...")
    per_cycle_params = []
    for c in cycles:
        ocv = ocv_fn(c["soc"])
        e = c["volt"] - ocv
        result = rls_arx(e, c["curr"])
        if result is None:
            continue
        ecm = arx_to_ecm(*result, Ts=c["Ts"])
        if ecm is None:
            print(f"  {os.path.basename(c['folder'])}: non-physical fit, discarded")
            continue
        ecm["folder"] = os.path.basename(os.path.dirname(c["folder"])) + "/" + os.path.basename(c["folder"])
        per_cycle_params.append(ecm)
        print(f"  {ecm['folder']}: R0={ecm['R0_ohm']:.4f} Ohm, R1={ecm['R1_ohm']:.4f} Ohm, "
              f"C1={ecm['C1_farad']:.1f} F, tau={ecm['tau_s']:.1f} s")

    if not per_cycle_params:
        raise RuntimeError("RLS identification failed on every cycle - no usable fits.")

    def agg(key):
        vals = np.array([p[key] for p in per_cycle_params])
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n_cycles": int(len(vals)),
        }

    results = {
        "provenance": {
            "capacity_ah": CAPACITY_AH,
            "capacity_source": "_code/fig_6_11_12/fig_6_11_12.m line 'Cap = 240' "
                                "(dataset authors' own analysis script; not in README.pdf)",
            "voltage_source": "observed min/max/mean across all readable Drive+Charge cycles",
            "chemistry": None,
            "chemistry_note": "not disclosed anywhere in the public dataset metadata; not fabricated here",
            "n_cycles_used": len(cycles),
        },
        "voltage": {
            "min_v": float(all_volt.min()),
            "max_v": float(all_volt.max()),
            "mean_v": float(all_volt.mean()),
        },
        "current": {
            "max_discharge_current_a_p99_9": max_discharge_current,
            "max_charge_current_a_p99_9": max_charge_current,
            "max_discharge_c_rate": max_discharge_current / CAPACITY_AH if max_discharge_current else None,
            "max_charge_c_rate": max_charge_current / CAPACITY_AH if max_charge_current else None,
        },
        "ocv_curve_soc_percent": ocv_curve[0].tolist(),
        "ocv_curve_volt": ocv_curve[1].tolist(),
        "ecm_r0_ohm": agg("R0_ohm"),
        "ecm_r1_ohm": agg("R1_ohm"),
        "ecm_c1_farad": agg("C1_farad"),
        "ecm_tau_s": agg("tau_s"),
        "per_cycle": per_cycle_params,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_PATH}")

    r0, r1, c1, tau = results["ecm_r0_ohm"], results["ecm_r1_ohm"], results["ecm_c1_farad"], results["ecm_tau_s"]
    print(f"\nIdentified ECM (n={r0['n_cycles']} cycles):")
    print(f"  R0  = {r0['mean']:.4f} +/- {r0['std']:.4f} Ohm")
    print(f"  R1  = {r1['mean']:.4f} +/- {r1['std']:.4f} Ohm")
    print(f"  C1  = {c1['mean']:.1f} +/- {c1['std']:.1f} F")
    print(f"  tau = {tau['mean']:.1f} +/- {tau['std']:.1f} s")
    print(f"  Pack voltage: [{results['voltage']['min_v']:.1f}, {results['voltage']['max_v']:.1f}] V, "
          f"mean {results['voltage']['mean_v']:.1f} V")

    return results


if __name__ == "__main__":
    run_identification()
