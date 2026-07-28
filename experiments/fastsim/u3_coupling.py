"""U-3 - braking -> regen coupling wrapper (Track U). SIMULATION.

Runs a UAH-derived drive cycle (U-2) through the U-1 Mendeley-like BEV under two
strategies and returns standardized outputs (SoC, battery power, regen energy,
derived current, and net energy per km).

The coupling, physically: this vehicle is fully regen-capable (peak braking power
is far below the motor/pack limits), so FASTSim recovers essentially all braking
as regen and NO energy is lost to friction. The honest benefit of anticipation is
therefore NOT extra regen but lower net consumption: knowing a brake is coming, the
car lifts off and coasts down earlier instead of holding speed then braking, so it
draws less propulsion energy and avoids the regen round-trip loss. Metric = Wh/km
(distance-normalized). So:
  * baseline     = run the raw cycle (reactive braking).
  * anticipatory = advance/coast into predicted-braking windows; the controller
                   keeps it only when net Wh/km drops (else falls back to baseline).

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
DIST_COL = "veh.history.dist_meters"
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


def _decel_segments(v, min_drop=1.0):
    """Contiguous runs where speed decreases with total drop >= min_drop (m/s).
    Returns (start_idx, end_idx) pairs (start = cruise speed, end = target speed)."""
    n = len(v)
    segs = []
    i = 1
    while i < n:
        if v[i] < v[i - 1] - 1e-6:
            j = i
            while j < n and v[j] <= v[j - 1] + 1e-6:
                j += 1
            a, b = i - 1, j - 1
            if v[a] - v[b] >= min_drop:
                segs.append((a, b))
            i = j
        else:
            i += 1
    return segs


BRAKE_TRIGGER_W = u2.U1_MAX_PROP_W    # only anticipate meaningful brakes (peak power above this)
COAST_MASS_KG = u2.U1_MASS_KG


def apply_anticipatory(speed_mps, intent, lead_s=4, trigger_w=BRAKE_TRIGGER_W, mass_kg=COAST_MASS_KG):
    """For each PREDICTED, non-trivial braking event, begin coasting down lead_s
    seconds earlier and reach the SAME target speed - the vehicle sheds speed by
    coasting instead of holding speed then braking, which lowers net propulsion
    energy (see module docstring). Only events whose peak braking power exceeds
    trigger_w are anticipated (gentle micro-brakes are ignored); edits never
    overlap. The trip-level controller (couple_ablation) keeps the result only
    when net Wh/km actually drops."""
    v = np.asarray(speed_mps, float).copy()
    intent = np.asarray(intent, int)
    last = -1
    for a, b in _decel_segments(v):
        lo = max(0, a - int(lead_s), last + 1)           # no overlap with a prior edit
        if lo >= b or not intent[max(0, a - int(lead_s)):a + 1].any():
            continue
        seg = v[a:b + 1]
        peak_brake_w = float((mass_kg * -np.diff(seg, prepend=seg[0]) * seg).max())
        if peak_brake_w <= trigger_w:                    # trivial brake -> not worth anticipating
            continue
        v[lo:b + 1] = np.linspace(v[lo], v[b], b - lo + 1)   # coast in earlier
        last = b
    return np.clip(v, 0.0, None)


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
    net_kwh = float((soc[0] - soc[-1]) * cap_kwh)
    dist_km = (float(df[DIST_COL].to_numpy()[-1]) / 1000.0
               if DIST_COL in df.columns else float(v_used[1:].sum()) / 1000.0)
    wh_per_km = net_kwh * 1000.0 / max(dist_km, 1e-9)             # distance-normalized consumption

    return {
        "strategy": strategy,
        "time_s": tt.tolist(),
        "soc": soc.tolist(),
        "batt_power_w": pwr.tolist(),
        "batt_current_a": current.tolist(),
        "regen_energy_wh": round(regen_wh, 2),
        "soc_start": round(float(soc[0]), 5),
        "soc_end": round(float(soc[-1]), 5),
        "energy_used_kwh": round(net_kwh, 3),
        "distance_km": round(dist_km, 3),
        "wh_per_km": round(wh_per_km, 2),
        "meta": {"assumed_voltage_v": assumed_voltage, "any_nan": bool(np.isnan(soc).any()),
                 "n_points": int(len(soc)), "simulation": True},
    }


def couple_ablation(time_s, speed_mps, intent=None, vehicle=None):
    """Baseline vs anticipatory on the same trip. The benefit metric is net energy
    per km (distance-normalized), NOT regen: on a fully regen-capable vehicle no
    braking is lost to friction, so the honest gain from anticipation is that
    early coasting draws less propulsion energy (avoiding the regen round-trip
    loss). The controller intervenes only when it lowers Wh/km, so savings >= 0."""
    if vehicle is None:
        from vehicle_config import build_mendeley_bev
        vehicle, _ = build_mendeley_bev()
    base = run_coupled_sim(time_s, speed_mps, "baseline", None, vehicle)
    anti = run_coupled_sim(time_s, speed_mps, "anticipatory", intent, vehicle)

    intervened = anti["wh_per_km"] < base["wh_per_km"]
    if not intervened:                                    # coasting didn't help -> stay baseline
        anti = dict(base, strategy="anticipatory",
                    meta=dict(base["meta"], intervened=False))
    else:
        anti["meta"] = dict(anti["meta"], intervened=True)

    saved = base["wh_per_km"] - anti["wh_per_km"]
    return {
        "baseline": base,
        "anticipatory": anti,
        "intervened": intervened,
        "energy_saved_wh_per_km": round(saved, 2),
        "energy_saved_pct": round(100.0 * saved / max(base["wh_per_km"], 1e-9), 2),
        "regen_wh": {"baseline": base["regen_energy_wh"], "anticipatory": anti["regen_energy_wh"]},
    }


# --------------------------- driver: smoke-run on one U-2 cycle ---------------------------
def _load_cycle_json(path):
    d = json.load(open(path))
    return np.asarray(d["time_seconds"], float), np.asarray(d["speed_meters_per_second"], float)


def _all_cycles():
    """All U-2-generated cycles; else build from representative trips."""
    found = []
    for style in ("normal", "aggressive", "drowsy"):
        p = u2.CYCLES_DIR / f"cycle_{style}.json"
        if p.exists():
            found.append((style, _load_cycle_json(p)))
    if found:
        return found
    for style, trip in u2._pick_trips().items():
        arr = u2.cycle_arrays(trip)
        if arr is not None:
            found.append((style.lower(), arr))
    if not found:
        raise SystemExit("no usable cycle found - run u2_cycle_adapter.py first")
    return found


def main():
    report = {"note": "SIMULATION - braking->regen coupling ablation (U-3); "
                      "benefit metric = net energy per km (early-coast anticipation)", "cycles": {}}
    any_nan = False
    for style, (t, v) in _all_cycles():
        res = couple_ablation(t, v)
        b, a = res["baseline"], res["anticipatory"]
        nan = bool(b["meta"]["any_nan"] or a["meta"]["any_nan"])
        any_nan = any_nan or nan
        report["cycles"][style] = {
            "wh_per_km": {"baseline": b["wh_per_km"], "anticipatory": a["wh_per_km"],
                          "saved": res["energy_saved_wh_per_km"], "saved_pct": res["energy_saved_pct"]},
            "distance_km": {"baseline": b["distance_km"], "anticipatory": a["distance_km"]},
            "regen_energy_wh": res["regen_wh"],
            "soc_end": {"baseline": b["soc_end"], "anticipatory": a["soc_end"]},
            "intervened": res["intervened"],
            "any_nan": nan,
        }
        w = report["cycles"][style]["wh_per_km"]
        print(f"  {style:10} energy: baseline={w['baseline']} anticipatory={w['anticipatory']} Wh/km "
              f"saved={w['saved']} ({w['saved_pct']}%) intervened={res['intervened']} nan={nan}")
    report["any_nan"] = any_nan
    with open(HERE / "u3_validation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  wrote {HERE / 'u3_validation.json'}")


if __name__ == "__main__":
    main()
