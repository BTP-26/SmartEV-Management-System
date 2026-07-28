"""Track U — centralized simulation configuration (U-4).

Single documented home for every tunable that affects Track U simulation results, so a reviewer
can see all modeling choices in one place and the sensitivity sweep has one thing to vary.
Previously these were scattered as literals across u2/u3 (review item N-3).

Values here MIRROR the defaults currently compiled into U-2/U-3; this module does not monkey-patch
them. U-4 passes them explicitly where the API allows, and records them in every output artifact
so each result is traceable to the configuration that produced it.

ALL Track U outputs are SIMULATION.
"""
from dataclasses import dataclass, asdict, field
from typing import Dict, Any


@dataclass(frozen=True)
class SimConfig:
    # --- speed-trace conditioning (U-2 `condition_speed`) ---
    # Fraction of the U-1 vehicle's peak forward propulsion power usable when capping
    # acceleration. Peak power is only available near base speed; at the low speeds the
    # conditioner targets, the motor is torque-limited. NOTE (review C-1): the budget is applied
    # speed-INDEPENDENTLY, so high-speed accelerations are capped more than the vehicle truly
    # requires. Measured impact is small on reference cycles (UDDS 0.7% of steps, HWFET 0.1%),
    # but MUST be re-measured per driving style on UAH — see `clipping_stats()`.
    accel_power_fraction: float = 0.131

    # --- anticipatory-regen control (U-3) ---
    # Peak MECHANICAL braking power above which a deceleration is worth anticipating.
    # A property of braking salience (P = m*a*v), deliberately independent of propulsion (C-2).
    # ~45 kW ~= 1.5 m/s^2 at ~15 m/s for this ~2070 kg vehicle.
    brake_trigger_w: float = 45_000.0
    # How many seconds earlier the vehicle begins coasting into a predicted brake.
    lead_s: int = 4
    # Minimum speed drop (m/s) for a deceleration to count as a braking event.
    min_drop_mps: float = 1.0
    # Oracle-intent look-ahead and deceleration threshold (placeholder until the real
    # braking model supplies intent).
    intent_lead_s: int = 3
    intent_decel_thresh_mps2: float = 0.5

    # --- data quality gates (U-4; review item R-2) ---
    # Max tolerated gap (s) between consecutive raw GPS fixes. Longer gaps are linearly
    # interpolated by the adapter, which FABRICATES plausible-looking speed, so trips exceeding
    # this are flagged and excluded from the reported table.
    max_gps_gap_s: float = 10.0
    # Minimum usable trip duration (s) and distance (km).
    min_duration_s: float = 60.0
    min_distance_km: float = 0.5

    # --- provenance / honesty flags carried into every output row (R-1) ---
    # Intent is an ORACLE (derived from the true future trace) and the trip-level controller keeps
    # the anticipatory run only when it lowers Wh/km. Reported savings are therefore an UPPER
    # BOUND, not a deployable result. This flag must survive into the paper's tables.
    oracle_upper_bound: bool = True
    simulation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT = SimConfig()

# Sensitivity sweep grid (U-4 step 7): does the per-style conclusion survive these choices?
SENSITIVITY_GRID = {
    "accel_power_fraction": [0.131, 0.188, 0.30],   # 0.188 == the "~45 kW" reading; 0.30 == looser
    "brake_trigger_w": [25_000.0, 45_000.0, 80_000.0],
    "lead_s": [2, 4, 6],
}
