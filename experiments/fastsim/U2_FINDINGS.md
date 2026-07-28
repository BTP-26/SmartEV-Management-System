# U-2 — UAH GPS → FASTSim Drive-Cycle Adapter: Findings

**Date:** 2026-07-28 · **Scope:** U-2 only (cycle adapter + the C-1 fix that touches it; no U-1
vehicle, no braking/SoC model, no U-4, no paper). **Verdict: ready → U-4 can consume the cycles.**
All outputs are **simulation** inputs.

---

## 1. Implementation summary

- Converts a UAH `RAW_GPS.txt` trace into a uniform **1 Hz FASTSim `Cycle`**: parse (time, km/h) →
  m/s → sort/drop non-monotonic timestamps → linear-interpolate onto a 1 Hz grid → clip negatives →
  build the cycle pydict.
- The GPS reader is **self-contained** (mirrors `load_gps_data`'s column layout) so the module runs
  inside the dedicated `.venv-fastsim` and imports nothing from the braking/SoC pipelines.
- `condition_speed()` makes every trace **powertrain-followable** before handing it to FASTSim
  (see §2) — this is what lets mid-motion starts and GPS spikes simulate without error.
- `main()` builds one representative cycle per behaviour tag (Normal / Aggressive / Drowsy), writes
  the regenerable cycles to `cycles/` (git-ignored) and a validation report to `u2_validation.json`.

## 2. What changed in this pass (review fix **C-1**)

- **Problem:** the acceleration budget used a hardcoded `U1_MAX_PROP_W = 44717.0` W, with a comment
  claiming it came from `u1_vehicle_params.json`. That was **wrong on both counts**: the file lists
  the motor peak as **239 kW**, and `44717` was actually FASTSim's *low-speed* forward-propulsion
  reading captured from an earlier runtime error — not a documented vehicle parameter.
- **Fix:** the max forward propulsion power is now **read from the U-1 vehicle at runtime** via a new
  single-source-of-truth helper `vehicle_config.max_fwd_propulsion_power_w()` (the electric machine's
  `pwr_out_max_watts`), cached once. The hardcoded constant and the false comment are removed.
- **Why the conditioner still uses a fraction of that peak:** peak motor power (239 kW) is only
  reached near base speed; at the near-zero speeds `condition_speed` targets (launches / spikes) the
  motor is torque-limited and only ~45 kW is actually available. `ACCEL_POWER_FRACTION = 0.131`
  encodes this, and is documented with tuning guidance (lower if a launch still fails, raise if
  cycles look over-smoothed).
- **Behaviour preserved:** effective budget = 0.131 × 239 kW = **31.3 kW**, identical to the
  previously-validated 0.7 × 44.7 kW. Regenerated conditioned speeds differ by **≤ 0.004 m/s** and
  distances are identical — so this is a provenance/traceability fix, not a behaviour change.

**Why `condition_speed` exists at all (unchanged design):** FASTSim launches the vehicle from rest
and must *exactly* meet the prescribed speed trace. Raw UAH GPS trips often (a) start mid-motion
(first sample already ~40 mph) or (b) contain single-sample speed spikes. Either demands impossible
tractive power and aborts the sim. The conditioner caps each 1 Hz step's acceleration to what the
powertrain can deliver at the achieved speed (solving `m·(v−v_prev)·v ≤ budget`); deceleration is
left untouched (friction braking is not propulsion-limited), so only infeasible steps are lowered.

## 3. Assumptions

1. **GPS column layout** matches the braking pipeline (`col0 = time s`, `col1 = speed km/h`); rows
   with < 11 columns are skipped.
2. **Uniform 1 Hz** resampling by linear interpolation is an adequate representation of UAH speed
   (native cadence ≈ 1 Hz); negatives clipped to 0.
3. **Flat road** (grade = 0, elevation = 0) — altitude-derived grade is out of scope (documented risk).
4. **Launch/spike conditioning** assumes only ~13% of peak motor power is available in the low-speed
   region; this matches FASTSim's reported low-speed propulsion limit for this vehicle and is
   tuneable per vehicle.
5. Ambient temperature array set to **295.15 K** (the thermal Tesla model needs one).

## 4. Validation

Regenerated from scratch (old `cycles/` deleted first), then re-run through the U-1 BEV:

| Cycle | Trip | Duration | Distance | `sim.ran` | `any_nan` | SoC drop |
|---|---|---|---|---|---|---|
| Normal | D1 …NORMAL1-SECONDARY | 623 s | 16.6 km | true | false | ~2.3 % |
| Aggressive | D1 …AGGRESSIVE-MOTORWAY | 736 s | 24.0 km | true | false | ~4.5 % |
| Drowsy | D1 …DROWSY-MOTORWAY | 938 s | 25.1 km | true | false | ~2–3 % |

- Structural checks pass for all three: uniform 1 Hz, monotonic time, non-negative & plausible speed.
- Distances match the trip-name km hints (e.g. the "16 km" Normal trip → 16.6 km).
- Post-C-1 conditioned speeds are numerically identical to the previously-validated run.

## 5. Limitations to carry into U-4

- Cycles are **counterfactual**: UAH drivers were not in this EV; absolute energy/SoC magnitudes are
  illustrative, not vehicle-accurate (inherited from U-1).
- **Flat-road** assumption biases energy; altitude-grade is a future refinement.
- Conditioning **lowers a handful of infeasible acceleration steps** (launch ramp + spikes); it never
  touches braking, so it does not affect the regen/coupling analysis.
- One representative trip per style is used; a multi-trip average is future work.

## 6. Files

**Modified (all in `experiments/fastsim/`):**
- `u2_cycle_adapter.py` — C-1: `condition_speed` reads max propulsion power from U-1 at runtime
  (`_max_fwd_prop_power_w`), hardcoded `U1_MAX_PROP_W` + false comment removed, `ACCEL_POWER_FRACTION`
  documented.
- `vehicle_config.py` — added `max_fwd_propulsion_power_w()` (single source of truth). *(U-1 builder
  itself unchanged.)*

**Regenerated:** `u2_validation.json`, `cycles/` (git-ignored). **Not modified:** U-0, U-1 vehicle,
braking/SoC models.
