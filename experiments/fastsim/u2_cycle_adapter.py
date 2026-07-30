"""U-2 - UAH GPS speed -> FASTSim drive-cycle adapter (Track U).

Turns a UAH `RAW_GPS.txt` trace into a uniform 1 Hz FASTSim `Cycle`. The GPS
reader is self-contained (mirrors the braking pipeline's `load_gps_data` column
layout) so this module stays inside the dedicated `.venv-fastsim` and imports
nothing from the braking/SoC pipelines. All outputs are SIMULATION inputs.

Run:  .venv-fastsim/bin/python experiments/fastsim/u2_cycle_adapter.py
"""
import os
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent
DATA_DIR = REPO / "UAH-DRIVESET-v1"
CYCLES_DIR = HERE / "cycles"

KMH_TO_MS = 1.0 / 3.6
TEMP_AMB_K = 295.15        # thermal (Tesla) vehicle needs an ambient-temp array
MAX_PLAUSIBLE_MPS = 60.0   # ~216 km/h, a generous sanity ceiling
MIN_POINTS = 10


# --------------------------- parsing + resampling (numpy only) ---------------------------
def read_uah_gps(trip_path):
    """Return (time_s, speed_kmh) from RAW_GPS.txt, or (None, None).

    Column layout matches load_gps_data: col0 = time (s), col1 = speed (km/h).
    """
    gps = os.path.join(trip_path, "RAW_GPS.txt")
    if not os.path.exists(gps):
        return None, None
    rows = []
    with open(gps) as fh:
        for line in fh:
            v = line.split()
            if len(v) >= 11:
                rows.append((float(v[0]), float(v[1])))
    if not rows:
        return None, None
    a = np.asarray(rows, dtype=float)
    return a[:, 0], a[:, 1]


def to_uniform_1hz(time_s, speed_mps):
    """Sort, drop non-monotonic timestamps, resample onto a uniform 1 Hz grid."""
    order = np.argsort(time_s)
    t, v = time_s[order], speed_mps[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-6])   # strictly increasing time
    t, v = t[keep], v[keep]
    grid = np.arange(0.0, float(t[-1] - t[0]) + 1.0, 1.0)  # 0..duration at 1 Hz
    vg = np.clip(np.interp(grid, t - t[0], v), 0.0, None)  # no negative speed
    return grid, vg


def cycle_arrays(trip_path):
    """UAH trip -> (time_s, speed_mps) on a uniform 1 Hz grid, or None."""
    t, kmh = read_uah_gps(trip_path)
    if t is None or len(t) < MIN_POINTS:
        return None
    return to_uniform_1hz(t, kmh * KMH_TO_MS)


def validate_arrays(time_s, speed_mps):
    """Structural checks on a cycle (no FASTSim run)."""
    dt = np.diff(time_s)
    dist_km = float((speed_mps[1:] * dt).sum() / 1000.0) if len(dt) else 0.0
    checks = {
        "n_points": int(len(time_s)),
        "uniform_1hz": bool(len(dt) and np.allclose(dt, 1.0)),
        "monotonic_time": bool(np.all(dt > 0)) if len(dt) else True,
        "nonneg_speed": bool(np.all(speed_mps >= 0)),
        "speed_plausible": bool(speed_mps.max() <= MAX_PLAUSIBLE_MPS),
        "max_speed_mps": round(float(speed_mps.max()), 2),
        "duration_s": float(time_s[-1] - time_s[0]) if len(time_s) else 0.0,
        "distance_km": round(dist_km, 3),
    }
    checks["valid"] = bool(
        checks["n_points"] >= MIN_POINTS and checks["uniform_1hz"]
        and checks["monotonic_time"] and checks["nonneg_speed"] and checks["speed_plausible"])
    return checks


# --------------------------- FASTSim cycle build (needs fastsim) ---------------------------
U1_MASS_KG = 2070.8        # U-1 BEV mass; used to size the acceleration budget (U-1 never modified).
PROP_BUDGET = 0.7          # fraction of the (speed-dependent) max forward propulsion power reserved
                           # for acceleration; the remainder covers drag / rolling / grade.
A_MAX_MPS2 = 3.0           # per-step acceleration cap (~0.3 g): the max a real vehicle/driver sustains,
                           # and what FASTSim's quasi-static solver can follow from a cold start. The
                           # power budget governs high speed; this governs the launch/low-speed ramp.

_PROP_CURVE = None         # cached (speed_grid, pwr_prop_fwd_max_grid), probed once from the U-1 vehicle


def _prop_power_curve():
    """Speed -> max forward propulsion power (W), read from FASTSim itself.

    Probes the U-1 vehicle with a gentle, full-range speed ramp and records
    `pwr_prop_fwd_max_watts` against achieved speed. This IS FASTSim's own
    speed-dependent propulsion capability (torque-limited at low speed, power-
    limited near/above base speed) - not a re-implemented torque curve, and not a
    constant. Cached; the gentle probe ramp needs no conditioning (condition=False),
    so there is no recursion with `condition_speed`."""
    global _PROP_CURVE
    if _PROP_CURVE is not None:
        return _PROP_CURVE
    import fastsim as fsim
    sys.path.insert(0, str(HERE))
    from vehicle_config import build_mendeley_bev          # U-1 (imported, never modified)
    veh, _ = build_mendeley_bev()
    n = int(MAX_PLAUSIBLE_MPS / 0.4) + 20
    t = np.arange(n, dtype=float)
    v = np.minimum(0.4 * t, MAX_PLAUSIBLE_MPS)             # gentle 0 -> vmax ramp (inherently feasible)
    cyc, _ = cycle_from_arrays(t, v, condition=False)      # gentle -> conditioning not needed
    sd = fsim.SimDrive(veh, cyc)
    sd.walk()
    df = sd.to_dataframe()
    v_ach = np.asarray(df["veh.history.speed_ach_meters_per_second"].to_numpy(), float)
    p_max = np.asarray(df["veh.history.pwr_prop_fwd_max_watts"].to_numpy(), float)
    order = np.argsort(v_ach)
    vg, idx = np.unique(np.round(v_ach[order], 2), return_index=True)  # strictly-increasing grid
    pg = p_max[order][idx]
    # Mechanical propulsion power is 0 at v=0 (P = F*v). Drop that degenerate point so the
    # interpolation clamps to the real launch capability (the first non-zero sample, ~45 kW,
    # which matches FASTSim's own launch limit) instead of 0 - otherwise the launch budget is
    # zero and the vehicle can never leave rest.
    keep = pg > 1000.0
    _PROP_CURVE = (vg[keep], pg[keep])
    return _PROP_CURVE


def _avail_prop_power_w(v):
    """FASTSim's max forward propulsion power (W) available at speed v (interp of the
    probed curve; clamps to the endpoint values outside the sampled range)."""
    vg, pg = _prop_power_curve()
    return float(np.interp(float(v), vg, pg))


def condition_speed(time_s, speed_mps, mass_kg=U1_MASS_KG):
    """Cap each 1 Hz step's acceleration to what the powertrain can actually deliver,
    using FASTSim's own speed-dependent propulsion capability (see `_prop_power_curve`).

    Each step keeps the largest next speed under BOTH limits:
      * propulsion power - `m*(v - prev)*v <= PROP_BUDGET * P_avail(prev)`. The limit is
        set by the speed at the START of the step (`prev`), mirroring FASTSim, whose
        `pwr_prop_fwd_max` is a function of the current speed (from rest only ~45 kW is
        available regardless of the target). This governs high speed.
      * acceleration - `v <= prev + A_MAX_MPS2`. FASTSim's quasi-static solver cannot
        follow the very steep launch ramp that the power limit alone permits from a cold
        start (it demands ~3.9 m/s^2, the solver stalls and the sim fails); capping at a
        realistic ~0.3 g keeps the ramp followable. This governs the launch/low-speed ramp.

    Why speed-dependent (review C-1 escalation, found in U-4): a constant budget capped
    high-speed accelerations far more than the vehicle requires, clipping ~23% of
    AGGRESSIVE-motorway steps vs ~9% of NORMAL and biasing the per-style energy table.
    The pair above removes that bias (clip: NORMAL ~0%, AGGRESSIVE ~2%) while staying
    feasible. Deceleration is untouched (friction braking is not propulsion-limited);
    only infeasible steps are lowered."""
    v = np.asarray(speed_mps, float).copy()
    prev = 0.0                                        # FASTSim starts the vehicle at rest
    for i in range(len(v)):
        budget = PROP_BUDGET * _avail_prop_power_w(prev)          # limit set by start-of-step speed
        v_power = (prev + np.sqrt(prev * prev + 4.0 * budget / mass_kg)) / 2.0
        v_max = min(v_power, prev + A_MAX_MPS2)                   # power OR acceleration, whichever binds
        if v[i] > v_max:
            v[i] = v_max
        prev = v[i]
    return np.asarray(time_s, float), v


def cycle_from_arrays(time_s, speed_mps, grade=0.0, condition=True):
    """(time, speed) arrays -> (fastsim.Cycle, pydict). Reused by U-3 for modified
    speed traces. Resizes every per-timestep field to our length (FASTSim needs
    grade/elev/pwr_* all equal length); flat road (grade=0). `condition=False`
    skips speed conditioning (used internally by the propulsion-curve probe, which
    is already feasible - avoids recursion with condition_speed)."""
    import fastsim as fsim
    if condition:
        time_s, speed_mps = condition_speed(time_s, speed_mps)
    else:
        time_s, speed_mps = np.asarray(time_s, float), np.asarray(speed_mps, float)
    d = fsim.Cycle.from_resource("udds.csv").to_pydict()   # schema-correct template
    n = len(time_s)
    tmpl_len = len(d["time_seconds"])
    dist = np.cumsum(speed_mps * np.diff(time_s, prepend=time_s[0]))
    for k, val in list(d.items()):
        if isinstance(val, list) and len(val) == tmpl_len:
            d[k] = [(val[0] if val else 0.0)] * n
    d["time_seconds"] = [float(x) for x in time_s]
    d["speed_meters_per_second"] = [float(x) for x in speed_mps]
    d["dist_meters"] = [float(x) for x in dist]
    d["grade"] = [float(grade)] * n
    d["elev_meters"] = [0.0] * n
    if "temp_amb_air_kelvin" in d:
        d["temp_amb_air_kelvin"] = [float(TEMP_AMB_K)] * n
    return fsim.Cycle.from_pydict(d), d


def build_cycle(trip_path, grade=0.0):
    """UAH trip -> (fastsim.Cycle, pydict). Returns None if the trip is unusable."""
    arr = cycle_arrays(trip_path)
    if arr is None:
        return None
    t, v = arr
    return cycle_from_arrays(t, v, grade)


def _sim_check(cyc):
    """Run the cycle through the U-1 Mendeley-like BEV; confirm no NaNs (SIMULATION)."""
    import fastsim as fsim
    sys.path.insert(0, str(HERE))
    from vehicle_config import build_mendeley_bev   # U-1 (imported, never modified)
    veh, _ = build_mendeley_bev()
    sd = fsim.SimDrive(veh, cyc)
    sd.walk()
    df = sd.to_dataframe()
    soc = np.asarray(df["veh.pt_type.BEV.res.history.soc"].to_numpy(), float)
    return {"ran": True, "any_nan": bool(np.isnan(soc).any()),
            "soc_start": round(float(soc[0]), 4), "soc_end": round(float(soc[-1]), 4),
            "soc_drop_pct": round(float((soc[0] - soc[-1]) * 100), 3)}


# --------------------------- driver: build representative + per-style cycles ---------------------------
def _pick_trips():
    """One representative trip per behavior tag (Normal/Aggressive/Drowsy) if available."""
    picks = {}
    for driver in sorted(os.listdir(DATA_DIR)):
        dp = DATA_DIR / driver
        if not dp.is_dir() or not driver.startswith("D"):
            continue
        for trip in sorted(os.listdir(dp)):
            tag = trip.split("-")[3].upper() if len(trip.split("-")) >= 5 else "?"
            style = "AGGRESSIVE" if "AGGRESSIVE" in tag else "DROWSY" if "DROWSY" in tag else \
                    "NORMAL" if "NORMAL" in tag else None
            if style and style not in picks and (dp / trip).is_dir():
                picks[style] = str(dp / trip)
    return picks


def main():
    CYCLES_DIR.mkdir(exist_ok=True)
    report = {"note": "SIMULATION inputs; UAH GPS -> uniform 1 Hz FASTSim cycles", "cycles": {}}
    for style, trip_path in _pick_trips().items():
        built = build_cycle(trip_path)
        if built is None:
            report["cycles"][style] = {"trip": os.path.basename(trip_path), "valid": False, "reason": "unusable"}
            continue
        cyc, d = built
        checks = validate_arrays(np.asarray(d["time_seconds"]), np.asarray(d["speed_meters_per_second"]))
        checks["trip"] = os.path.basename(trip_path)
        try:
            checks["sim"] = _sim_check(cyc)
        except Exception as exc:  # pragma: no cover
            checks["sim"] = {"ran": False, "error": str(exc)}
        with open(CYCLES_DIR / f"cycle_{style.lower()}.json", "w") as f:
            json.dump({"time_seconds": d["time_seconds"],
                       "speed_meters_per_second": d["speed_meters_per_second"]}, f)
        report["cycles"][style] = checks
        print(f"  {style:10} {checks['trip']}: valid={checks['valid']} "
              f"dur={checks['duration_s']:.0f}s dist={checks['distance_km']:.1f}km")
    with open(HERE / "u2_validation.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  wrote {HERE / 'u2_validation.json'} and cycles/ (git-ignored)")


if __name__ == "__main__":
    main()
