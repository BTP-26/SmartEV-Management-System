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
    # Each step is capped by TWO limits, whichever binds (review C-1 escalation, resolved):
    # (1) a SPEED-DEPENDENT propulsion-power budget, `prop_budget` * P_avail(v), where P_avail(v)
    #     is probed directly from FASTSim's own `pwr_prop_fwd_max_watts` (torque-limited at low
    #     speed, ~45 kW; power-limited near/above base speed, ~234 kW) — not a re-derived curve
    #     and not a constant, so high-speed accelerations are no longer capped more than the
    #     vehicle truly requires. This replaced a speed-INDEPENDENT constant budget that clipped
    #     ~23% of AGGRESSIVE-motorway steps vs ~9% of NORMAL, biasing the per-style table; the
    #     bias is gone with this model (re-verified per style/road — see `clipping_stats()`).
    # (2) a flat acceleration cap (`accel_cap_mps2`), which only binds during the low-speed launch
    #     ramp where the power limit alone would demand an infeasible ramp FASTSim's quasi-static
    #     solver cannot follow from a cold start.
    prop_budget: float = 0.7
    accel_cap_mps2: float = 3.0

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

# Sensitivity sweep grid (U-4 step 7, follow-up): does the per-style conclusion survive these choices?
SENSITIVITY_GRID = {
    "prop_budget": [0.5, 0.7, 0.9],
    "accel_cap_mps2": [2.0, 3.0, 4.0],
    "brake_trigger_w": [25_000.0, 45_000.0, 80_000.0],
    "lead_s": [2, 4, 6],
}
