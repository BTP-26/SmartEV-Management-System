"""Track U — Mendeley-like BEV builder (U-1).

Single source of truth for the FASTSim vehicle used across U-2..U-4. Later stages should either
call `build_mendeley_bev()` or load the serialized `MENDELEY_BEV_YAML` — no duplication.

Baseline: the FASTSim built-in **2022 Tesla Model 3 RWD** (closest available ~400 V-class BEV,
validated in U-0). We resize the battery pack to approximate the **Mendeley pack (~101.8 kWh)** and
adjust the two physical consequences of a bigger pack — vehicle mass and pack power — with
documented assumptions. Everything else (motor, chassis/aero, regen, thermal, aux) is inherited.

FASTSim-3 caveat: the battery (RES) is an **energy/power reservoir (Joules + power limit)** — it has
**no pack voltage or current**, so a "nominal voltage" cannot be set here (battery current stays
DERIVED as power / assumed_voltage downstream). Mendeley chemistry is unknown and is not modeled.
"""
from pathlib import Path

import fastsim as fsim

BASELINE_RESOURCE = "2022 Tesla Model 3 RWD thrml.yaml"
HERE = Path(__file__).parent
MENDELEY_BEV_YAML = HERE / "mendeley_like_bev.yaml"

# --- Mendeley pack targets (from modules/soc/models/battery_params_identified.json) ---
MENDELEY_CAP_AH = 240.0
MENDELEY_NOMINAL_V = 424.22
MENDELEY_CAP_KWH = MENDELEY_CAP_AH * MENDELEY_NOMINAL_V / 1000.0            # ~101.81 kWh
MENDELEY_PEAK_DISCHARGE_W = 828.476 * MENDELEY_NOMINAL_V                   # ~351 kW (observed p99.9)

# --- Documented engineering assumptions ---
# Pack-level specific energy used to convert added pack energy -> added vehicle mass.
# ~150 Wh/kg is a typical modern pack-level (not cell-level) figure; stated as an assumption.
PACK_SPECIFIC_ENERGY_WH_PER_KG = 150.0


def build_mendeley_bev(target_kwh: float = MENDELEY_CAP_KWH,
                       adjust_mass: bool = True,
                       scale_pack_power: bool = True):
    """Return (Vehicle, meta). `meta` records every base value, every modified value, every
    inherited block, and the assumptions — so downstream stages and the paper can cite it.

    Toggles let a reviewer fall back to a minimal-assumption variant:
      adjust_mass=False      -> keep the Tesla mass (100 kWh pack on a light chassis; unrealistic)
      scale_pack_power=False -> keep the Tesla pack power limit
    """
    veh = fsim.Vehicle.from_resource(BASELINE_RESOURCE)
    d = veh.to_pydict()
    res = d["pt_type"]["BEV"]["res"]

    base_kwh = res["energy_capacity_joules"] / 3.6e6
    base_mass = float(d["mass_kilograms"])
    base_res_pwr = float(res["pwr_out_max_watts"])
    motor_pwr = float(d["pt_type"]["BEV"]["em"]["pwr_out_max_watts"])
    ratio = target_kwh / base_kwh

    # (1) CORE: resize pack energy to the Mendeley target
    res["energy_capacity_joules"] = target_kwh * 3.6e6

    # (2) pack power scales with capacity (same chemistry -> ~constant C-rate capability)
    new_res_pwr = base_res_pwr
    if scale_pack_power:
        new_res_pwr = base_res_pwr * ratio
        res["pwr_out_max_watts"] = new_res_pwr

    # (3) vehicle mass increases with the larger pack (documented specific-energy assumption)
    new_mass = base_mass
    if adjust_mass:
        added_mass_kg = (target_kwh - base_kwh) * 1000.0 / PACK_SPECIFIC_ENERGY_WH_PER_KG
        new_mass = base_mass + added_mass_kg
        d["mass_kilograms"] = new_mass

    veh2 = fsim.Vehicle.from_pydict(d)

    meta = {
        "baseline_vehicle": BASELINE_RESOURCE,
        "assumptions": {
            "pack_specific_energy_wh_per_kg": PACK_SPECIFIC_ENERGY_WH_PER_KG,
            "pack_power_scales_with_capacity": scale_pack_power,
            "mass_adjusted_for_pack": adjust_mass,
            "nominal_voltage_settable": False,
            "voltage_note": "FASTSim-3 RES has no voltage/current; current stays derived (P/V).",
            "chemistry_modeled": False,
        },
        "modified": {
            "battery_capacity_kwh": {"base": round(base_kwh, 2), "new": round(target_kwh, 2),
                                     "ratio": round(ratio, 3)},
            "pack_max_power_kw": {"base": round(base_res_pwr / 1e3, 1),
                                  "new": round(new_res_pwr / 1e3, 1),
                                  "mendeley_observed_peak_kw": round(MENDELEY_PEAK_DISCHARGE_W / 1e3, 1)},
            "vehicle_mass_kg": {"base": round(base_mass, 1), "new": round(new_mass, 1),
                                "delta": round(new_mass - base_mass, 1)},
        },
        "inherited": {
            "motor_max_power_kw": round(motor_pwr / 1e3, 1),
            "drag_coef": d["chassis"]["drag_coef"],
            "frontal_area_m2": d["chassis"]["frontal_area_square_meters"],
            "wheel_rr_coef": d["chassis"]["wheel_rr_coef"],
            "drive_type": d["chassis"]["drive_type"],
            "aux_base_w": d.get("pwr_aux_base_watts"),
            "note": "motor, aero/rolling chassis, regen (electric-machine) map, thermal, aux — all "
                    "inherited unchanged from the Tesla Model 3 baseline.",
        },
        "not_modeled": ["pack nominal voltage", "pack current", "cell chemistry", "cell/module config"],
    }
    return veh2, meta


def max_fwd_propulsion_power_w(veh=None) -> float:
    """Maximum forward propulsion power (W) of the U-1 vehicle, read from the
    parameterized FASTSim vehicle itself (the electric machine's peak mechanical
    output). This is the single source of truth for downstream speed-trace
    conditioning (U-2/U-3) - callers must NOT hardcode it. Builds the vehicle if
    one is not supplied; the vehicle is never modified."""
    if veh is None:
        veh, _ = build_mendeley_bev()
    return float(veh.to_pydict()["pt_type"]["BEV"]["em"]["pwr_out_max_watts"])


def save_vehicle(veh, path: Path = MENDELEY_BEV_YAML) -> Path:
    """Serialize the configured vehicle for reuse by U-2..U-4 (load via Vehicle.from_file)."""
    veh.to_file(str(path))
    return path


if __name__ == "__main__":
    v, m = build_mendeley_bev()
    save_vehicle(v)
    import json
    print(json.dumps(m, indent=2))
    print(f"saved -> {MENDELEY_BEV_YAML}")
