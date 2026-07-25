"""U-0 — FASTSim feasibility spike (Track U).

Self-contained feasibility check: confirms FASTSim can act as the physics coupling layer by
running a built-in BEV over (a) the built-in UDDS cycle and (b) a synthetic accel->cruise->brake
trace, then extracting SoC / battery power / (derived) current / regen energy and sanity-checking
against the Mendeley pack.

STRICT SCOPE (U-0 only): no vehicle parameterization (U-1), no UAH cycle adapter (U-2), no coupling
wrapper (U-3), no SoC/braking code. Writes ONLY to this folder. Touches no project modules.
ALL OUTPUTS ARE SIMULATION.

Run:  .venv/bin/python experiments/fastsim/u0_spike.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fastsim as fsim

OUT = Path(__file__).parent

# --- Mendeley pack reference (for the pack-comparison sanity check) ---
# From modules/soc/models/battery_params_identified.json: 240 Ah pack, mean pack voltage 424.22 V.
MENDELEY_CAP_AH = 240.0
MENDELEY_NOMINAL_V = 424.22
MENDELEY_CAP_KWH = MENDELEY_CAP_AH * MENDELEY_NOMINAL_V / 1000.0  # ~101.8 kWh
# FASTSim-3 has NO pack voltage; current is DERIVED as power / assumed voltage.
ASSUMED_PACK_V = MENDELEY_NOMINAL_V

BEV_RESOURCE = "2022 Tesla Model 3 RWD thrml.yaml"  # closest available ~400V-class BEV; U-1 will match the pack

# Confirmed fastsim-3.0.6 result columns (see fastsim_api_reference.md)
SOC_COL = "veh.pt_type.BEV.res.history.soc"
PWR_COL = "veh.pt_type.BEV.res.history.pwr_out_electrical_watts"   # +discharge / -regen(charge)
SPEED_COL = "veh.history.speed_ach_meters_per_second"
TIME_COL = "veh.history.time_seconds"
DIST_COL = "veh.history.dist_meters"


def battery_capacity_kwh(veh) -> float:
    d = veh.to_pydict()
    j = d["pt_type"]["BEV"]["res"]["energy_capacity_joules"]
    return float(j) / 3.6e6


def run(veh_resource: str, cyc) -> "pd.DataFrame":
    veh = fsim.Vehicle.from_resource(veh_resource)
    sd = fsim.SimDrive(veh, cyc)
    sd.walk()
    return veh, sd.to_dataframe()


def synthetic_brake_cycle():
    """accel(0->15 m/s) -> cruise -> hard brake(->0) -> stop, on flat road. Built from the UDDS
    pydict template so the schema is guaranteed correct; only the time/speed/derived arrays change."""
    tmpl = fsim.Cycle.from_resource("udds.csv").to_pydict()
    t = np.arange(0, 121, dtype=float)          # 0..120 s @ 1 Hz
    v = np.piecewise(
        t,
        [t < 15, (t >= 15) & (t < 90), (t >= 90) & (t < 100), t >= 100],
        [lambda x: 15.0 * (x / 15.0),           # accelerate to 15 m/s
         15.0,                                   # cruise
         lambda x: 15.0 - 1.5 * (x - 90),        # hard brake at 1.5 m/s^2
         0.0],                                   # stopped
    )
    v = np.clip(v, 0.0, None)
    n = len(t)
    dt = np.diff(t, prepend=t[0])
    dist = np.cumsum(v * dt)
    d = dict(tmpl)
    d["time_seconds"] = t.tolist()
    d["speed_meters_per_second"] = v.tolist()
    d["dist_meters"] = dist.tolist()
    d["grade"] = [0.0] * n                       # flat road (assumption)
    d["elev_meters"] = [d["init_elev_meters"]] * n
    d["pwr_max_chrg_watts"] = [0.0] * n
    d["temp_amb_air_kelvin"] = [295.15] * n      # 22 C ambient
    d["pwr_solar_load_watts"] = [0.0] * n
    return fsim.Cycle.from_pydict(d)


def metrics(df, label) -> dict:
    t = np.asarray(df[TIME_COL].to_numpy(), dtype=float)
    soc = np.asarray(df[SOC_COL].to_numpy(), dtype=float)
    pwr = np.asarray(df[PWR_COL].to_numpy(), dtype=float)   # W, +discharge / -regen
    dt = np.diff(t, prepend=t[0])
    regen_wh = float(-np.sum(np.minimum(pwr, 0.0) * dt) / 3600.0)
    disch_wh = float(np.sum(np.maximum(pwr, 0.0) * dt) / 3600.0)
    cur = pwr / ASSUMED_PACK_V                     # DERIVED current (A)
    dsoc = np.diff(soc)
    regen_mask = pwr[1:] < 0.0                     # steps where battery is charging (regen)
    soc_rises_during_regen = bool(regen_mask.any() and (dsoc[regen_mask] > 0).mean() > 0.5)
    return {
        "label": label,
        "steps": int(len(t)),
        "duration_s": float(t[-1] - t[0]),
        "distance_km": float(df[DIST_COL].to_numpy()[-1] / 1000.0) if DIST_COL in df.columns else None,
        "soc_start": float(soc[0]),
        "soc_end": float(soc[-1]),
        "soc_min": float(soc.min()),
        "soc_max": float(soc.max()),
        "batt_pwr_min_kw": float(pwr.min() / 1e3),
        "batt_pwr_max_kw": float(pwr.max() / 1e3),
        "regen_energy_wh": regen_wh,
        "discharge_energy_wh": disch_wh,
        "regen_fraction_of_discharge": float(regen_wh / disch_wh) if disch_wh > 0 else None,
        "derived_current_min_a": float(cur.min()),
        "derived_current_max_a": float(cur.max()),
        "soc_rises_during_regen": soc_rises_during_regen,
        "regen_energy_positive": regen_wh > 0,
        "any_nan": bool(np.isnan(soc).any() or np.isnan(pwr).any()),
    }


def main():
    print(f"FASTSim version: {fsim.__version__}")
    results = {"fastsim_version": fsim.__version__, "bev_resource": BEV_RESOURCE}

    # Step 2/3: built-in BEV + built-in UDDS example (confirms the engine runs end-to-end)
    udds = fsim.Cycle.from_resource("udds.csv")
    veh, df_udds = run(BEV_RESOURCE, udds)
    cap_kwh = battery_capacity_kwh(veh)

    m_udds = metrics(df_udds, "UDDS (built-in)")
    kwh_used = (m_udds["soc_start"] - m_udds["soc_end"]) * cap_kwh
    m_udds["energy_used_kwh"] = float(kwh_used)
    m_udds["efficiency_km_per_kwh"] = (
        float(m_udds["distance_km"] / kwh_used) if kwh_used > 0 and m_udds["distance_km"] else None
    )

    # Step 4b: synthetic accel->cruise->hard-brake trace (clearly exercises regen)
    _, df_syn = run(BEV_RESOURCE, synthetic_brake_cycle())
    m_syn = metrics(df_syn, "synthetic accel-brake")

    # Pack-comparison sanity check (requested): FASTSim BEV vs Mendeley pack
    pack = {
        "fastsim_bev_capacity_kwh": round(cap_kwh, 2),
        "fastsim_bev_nominal_voltage_v": None,  # NOT modeled by FASTSim-3 (energy reservoir only)
        "mendeley_capacity_kwh": round(MENDELEY_CAP_KWH, 2),
        "mendeley_nominal_voltage_v": MENDELEY_NOMINAL_V,
        "capacity_ratio_mendeley_over_fastsim": round(MENDELEY_CAP_KWH / cap_kwh, 2),
        "voltage_note": (
            "FASTSim-3 exposes no pack voltage or current (RES is an energy/power reservoir). "
            "Battery current is DERIVED as power / assumed_voltage using the Mendeley nominal "
            f"({MENDELEY_NOMINAL_V} V). This is a documented limitation, not a fundamental mismatch: "
            "the coupling runs on SoC + battery power + regen energy, all of which are native."
        ),
    }

    results["udds"] = m_udds
    results["synthetic"] = m_syn
    results["pack_comparison"] = pack

    # Exit-criteria evaluation
    checks = {
        "install_and_example_ran": True,
        "soc_trajectory_extractable": not m_udds["any_nan"],
        "battery_power_nontrivial": abs(m_udds["batt_pwr_min_kw"]) > 1 or abs(m_udds["batt_pwr_max_kw"]) > 1,
        "regen_energy_positive": m_udds["regen_energy_positive"] and m_syn["regen_energy_positive"],
        "soc_rises_during_regen": m_udds["soc_rises_during_regen"] or m_syn["soc_rises_during_regen"],
        "derived_current_sane": 0 < abs(m_udds["derived_current_max_a"]) < 2000,
    }
    results["exit_criteria"] = checks
    results["all_exit_criteria_met"] = all(checks.values())

    (OUT / "u0_results.json").write_text(json.dumps(results, indent=2))

    # Plot SoC + battery power for the synthetic trace (clearest regen signal)
    t = df_syn[TIME_COL].to_numpy()
    soc = df_syn[SOC_COL].to_numpy()
    pwr_kw = df_syn[PWR_COL].to_numpy() / 1e3
    spd = df_syn[SPEED_COL].to_numpy()
    fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t, spd, "b"); ax[0].set_ylabel("speed (m/s)"); ax[0].set_title("Synthetic accel->cruise->hard-brake (SIMULATION)")
    ax[1].plot(t, pwr_kw, "r"); ax[1].axhline(0, color="k", lw=0.6); ax[1].set_ylabel("batt power (kW)\n(+dis / -regen)")
    ax[2].plot(t, soc, "g"); ax[2].set_ylabel("SoC"); ax[2].set_xlabel("time (s)")
    for a in ax: a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "u0_soc_power.png", dpi=110); plt.close(fig)

    print(json.dumps(results, indent=2))
    print("\nAll exit criteria met:", results["all_exit_criteria_met"])


if __name__ == "__main__":
    main()
