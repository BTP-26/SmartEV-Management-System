"""U-3 - braking -> regen coupling wrapper (Track U). SIMULATION.

Runs a UAH-derived drive cycle (U-2) through the U-1 Mendeley-like BEV under two
regen strategies and returns standardized outputs (SoC, battery power, regen
energy, derived current).

The coupling, physically: FASTSim recovers regen from the speed trace. A late,
sharp brake demands more braking power than the motor's regen limit, so the
excess is lost to friction. If a brake is anticipated a few seconds early, the
vehicle can begin slowing sooner and more gently, keeping braking power under
the regen cap -> more energy recovered. So:
  * baseline     = run the raw cycle (reactive braking).
  * anticipatory = smooth/advance deceleration over predicted-braking windows.

Intent contract (so U-4 consumes this unchanged): `intent` is a per-timestep
binary array (1 = brake predicted). Until the real braking model feeds it, an
ORACLE placeholder is derived from the cycle itself. Every output is SIMULATION.

FASTSim-3 RES exposes no pack voltage/current, so battery current is DERIVED as
power / ASSUMED_PACK_V (Mendeley nominal), matching the U-0/U-1 convention.

Run:  .venv-fastsim/bin/python experiments/fastsim/u3_coupling.py
"""
import os
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import u2_cycle_adapter as u2   # cycle_from_arrays, cycle_arrays, CYCLES_DIR, _pick_trips

ASSUMED_PACK_V = 424.22               # Mendeley nominal; current = power / this (derived)
SOC_COL = "veh.pt_type.BEV.res.history.soc"
TIME_COL = "veh.history.time_seconds"
# RES electrical power (sign convention: + discharge / - regen). Exact name can
# vary across builds, so run_coupled_sim resolves it defensively from columns.
PWR_CANDIDATES = (
    "veh.pt_type.BEV.res.history.pwr_out_electrical_watts",
    "veh.pt_type.BEV.res.history.pwr_out_watts",
)


# --------------------------- intent + anticipatory transform (numpy only) ---------------------------
def derive_oracle_intent(speed_mps, lead_s=3, decel_thresh_mps2=0.5):
    """Placeholder intent: 1 where the trace is about to decelerate within the
    next `lead_s` seconds (an oracle standing in for the real braking model)."""
    v = np.asarray(speed_mps, float)
    n = len(v)
    intent = np.zeros(n, dtype=int)
    for i in range(n):
        j = min(n - 1, i + int(lead_s))
        if (v[i] - v[j]) > decel_thresh_mps2 * lead_s:   # meaningful upcoming drop
            intent[i] = 1
    return intent


def _dilate(mask, k):
    """Grow a binary mask by k samples on each side (covers the ramp region)."""
    if k <= 0:
        return mask.astype(bool)
    kern = np.ones(2 * k + 1)
    return np.convolve(mask.astype(float), kern, mode="same") > 0


def apply_anticipatory(speed_mps, intent, window_s=3):
    """Over predicted-braking windows, replace the sharp deceleration with a
    causal moving-average version: lowers peak braking power (more stays under
    the regen cap) and effectively begins slowing earlier. Endpoints preserved."""
    v = np.asarray(speed_mps, float)
    k = max(1, int(window_s))
    kern = np.ones(k) / k
    smoothed = np.convolve(v, kern, mode="same")
    mask = _dilate(np.asarray(intent, int), k)
    out = np.where(mask, smoothed, v)
    out[0], out[-1] = v[0], v[-1]                 # keep trip start/end speed
    return np.clip(out, 0.0, None)


# --------------------------- coupled simulation (needs fastsim) ---------------------------
def _resolve_pwr_col(df):
    for c in PWR_CANDIDATES:
        if c in df.columns:
            return c
    hits = [c for c in df.columns if "res" in c and "pwr" in c and "watt" in c]
    if hits:
        return hits[0]
    raise KeyError(f"no RES power column found; sample columns: {list(df.columns)[:12]}")


def run_coupled_sim(time_s, speed_mps, strategy="baseline", intent=None,
                    vehicle=None, assumed_voltage=ASSUMED_PACK_V):
    """Run one strategy through the U-1 BEV -> StandardizedResult dict (SIMULATION)."""
    import fastsim as fsim
    t = np.asarray(time_s, float)
    v = np.asarray(speed_mps, float)

    if strategy == "anticipatory":
        if intent is None:
            intent = derive_oracle_intent(v)
        v_used = apply_anticipatory(v, intent)
    elif strategy == "baseline":
        v_used = v
    else:
        raise ValueError(f"unknown strategy {strategy!r} (use 'baseline' or 'anticipatory')")

    if vehicle is None:
        from vehicle_config import build_mendeley_bev   # U-1 (imported, never modified)
        vehicle, _ = build_mendeley_bev()

    cyc, _ = u2.cycle_from_arrays(t, v_used)
    sd = fsim.SimDrive(vehicle, cyc)
    sd.walk()
    df = sd.to_dataframe()

    tt = np.asarray(df[TIME_COL].to_numpy(), float) if TIME_COL in df.columns else t
    soc = np.asarray(df[SOC_COL].to_numpy(), float)
    pwr = np.asarray(df[_resolve_pwr_col(df)].to_numpy(), float)   # + discharge / - regen
    dt = np.diff(tt, prepend=tt[0])
    regen_wh = float(-np.minimum(pwr, 0.0).dot(dt) / 3600.0)       # energy back into pack
    current = pwr / assumed_voltage
    cap_kwh = vehicle.to_pydict()["pt_type"]["BEV"]["res"]["energy_capacity_joules"] / 3.6e6

    return {
        "strategy": strategy,
        "time_s": tt.tolist(),
        "soc": soc.tolist(),
        "batt_power_w": pwr.tolist(),
        "batt_current_a": current.tolist(),
        "regen_energy_wh": round(regen_wh, 2),
        "soc_start": round(float(soc[0]), 5),
        "soc_end": round(float(soc[-1]), 5),
        "energy_used_kwh": round(float((soc[0] - soc[-1]) * cap_kwh), 3),
        "meta": {"assumed_voltage_v": assumed_voltage, "any_nan": bool(np.isnan(soc).any()),
                 "n_points": int(len(soc)), "simulation": True},
    }


def couple_ablation(time_s, speed_mps, intent=None, vehicle=None):
    """Baseline vs anticipatory on the same trip -> both results + regen delta."""
    if vehicle is None:
        from vehicle_config import build_mendeley_bev
        vehicle, _ = build_mendeley_bev()
    base = run_coupled_sim(time_s, speed_mps, "baseline", None, vehicle)
    anti = run_coupled_sim(time_s, speed_mps, "anticipatory", intent, vehicle)
    gain = anti["regen_energy_wh"] - base["regen_energy_wh"]
    return {
        "baseline": base,
        "anticipatory": anti,
        "regen_gain_wh": round(gain, 2),
        "regen_gain_pct": round(100.0 * gain / max(base["regen_energy_wh"], 1e-9), 2),
    }


# --------------------------- driver: smoke-run on one U-2 cycle ---------------------------
def _load_cycle_json(path):
    d = json.load(open(path))
    return np.asarray(d["time_seconds"], float), np.asarray(d["speed_meters_per_second"], float)


def _get_cycle():
    """Prefer a U-2-generated cycle; else build one from the first usable trip."""
    for style in ("normal", "aggressive", "drowsy"):
        p = u2.CYCLES_DIR / f"cycle_{style}.json"
        if p.exists():
            return style, _load_cycle_json(p)
    for style, trip in u2._pick_trips().items():
        arr = u2.cycle_arrays(trip)
        if arr is not None:
            return style.lower(), arr
    raise SystemExit("no usable cycle found - run u2_cycle_adapter.py first")


def main():
    style, (t, v) = _get_cycle()
    res = couple_ablation(t, v)
    summary = {
        "note": "SIMULATION - braking->regen coupling ablation (U-3)",
        "cycle": style,
        "regen_energy_wh": {"baseline": res["baseline"]["regen_energy_wh"],
                            "anticipatory": res["anticipatory"]["regen_energy_wh"],
                            "gain_wh": res["regen_gain_wh"], "gain_pct": res["regen_gain_pct"]},
        "soc_end": {"baseline": res["baseline"]["soc_end"],
                    "anticipatory": res["anticipatory"]["soc_end"]},
        "energy_used_kwh": {"baseline": res["baseline"]["energy_used_kwh"],
                            "anticipatory": res["anticipatory"]["energy_used_kwh"]},
        "any_nan": bool(res["baseline"]["meta"]["any_nan"] or res["anticipatory"]["meta"]["any_nan"]),
    }
    with open(HERE / "u3_validation.json", "w") as f:
        json.dump(summary, f, indent=2)
    r = summary["regen_energy_wh"]
    print(f"  cycle={style}  regen: baseline={r['baseline']}Wh "
          f"anticipatory={r['anticipatory']}Wh  gain={r['gain_wh']}Wh ({r['gain_pct']}%)  "
          f"any_nan={summary['any_nan']}")
    print(f"  wrote {HERE / 'u3_validation.json'}")


if __name__ == "__main__":
    main()
