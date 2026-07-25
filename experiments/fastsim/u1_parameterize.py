"""U-1 driver — build the Mendeley-like BEV, confirm it simulates, and record a baseline-vs-
parameterized comparison. Writes only to experiments/fastsim/. Reuses vehicle_config.build_mendeley_bev.

Run:  .venv/bin/python experiments/fastsim/u1_parameterize.py
"""
import json
from pathlib import Path

import numpy as np
import fastsim as fsim

from vehicle_config import (
    build_mendeley_bev, save_vehicle, BASELINE_RESOURCE, MENDELEY_BEV_YAML, MENDELEY_CAP_KWH,
)

HERE = Path(__file__).parent
SOC = "veh.pt_type.BEV.res.history.soc"
PWR = "veh.pt_type.BEV.res.history.pwr_out_electrical_watts"
TIME = "veh.history.time_seconds"
DIST = "veh.history.dist_meters"


def cap_kwh(veh) -> float:
    return veh.to_pydict()["pt_type"]["BEV"]["res"]["energy_capacity_joules"] / 3.6e6


def sim_summary(veh, cyc) -> dict:
    sd = fsim.SimDrive(veh, cyc)
    sd.walk()
    df = sd.to_dataframe()
    t = np.asarray(df[TIME].to_numpy(), float)
    soc = np.asarray(df[SOC].to_numpy(), float)
    pwr = np.asarray(df[PWR].to_numpy(), float)
    dt = np.diff(t, prepend=t[0])
    dist_km = float(df[DIST].to_numpy()[-1] / 1000.0)
    kwh = (soc[0] - soc[-1]) * cap_kwh(veh)
    return {
        "distance_km": round(dist_km, 2),
        "soc_start": round(float(soc[0]), 4),
        "soc_end": round(float(soc[-1]), 4),
        "soc_drop_pct": round(float((soc[0] - soc[-1]) * 100), 3),
        "energy_kwh": round(float(kwh), 3),
        "km_per_kwh": round(dist_km / kwh, 2) if kwh > 0 else None,
        "regen_wh": round(float(-np.sum(np.minimum(pwr, 0.0) * dt) / 3600.0), 1),
        "batt_pwr_max_kw": round(float(pwr.max() / 1e3), 1),
        "batt_pwr_min_kw": round(float(pwr.min() / 1e3), 1),
        "any_nan": bool(np.isnan(soc).any()),
    }


def main():
    udds = fsim.Cycle.from_resource("udds.csv")
    base = fsim.Vehicle.from_resource(BASELINE_RESOURCE)
    param, meta = build_mendeley_bev()
    save_vehicle(param)

    out = {
        "meta": meta,
        "sanity_sim_udds": {
            "baseline_tesla_m3": sim_summary(base, udds),
            "mendeley_like_bev": sim_summary(param, udds),
        },
        "reusable_artifact": str(MENDELEY_BEV_YAML.relative_to(HERE.parent.parent)),
    }
    # confirm the serialized vehicle reloads with the right pack (reuse contract for U-2..U-4)
    reload = fsim.Vehicle.from_file(str(MENDELEY_BEV_YAML))
    out["reload_matches_target"] = abs(cap_kwh(reload) - MENDELEY_CAP_KWH) < 0.1
    out["simulates_ok"] = not out["sanity_sim_udds"]["mendeley_like_bev"]["any_nan"]

    (HERE / "u1_vehicle_params.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nsimulates_ok:", out["simulates_ok"], "| reload_matches_target:", out["reload_matches_target"])


if __name__ == "__main__":
    main()
