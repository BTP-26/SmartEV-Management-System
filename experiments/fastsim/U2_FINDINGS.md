# U-2 — UAH GPS → FASTSim Drive-Cycle Adapter: Findings

**Date:** 2026-07-28 (rev. speed-dependent conditioning) · **Scope:** U-2 only (cycle adapter +
conditioning model; no U-1 vehicle, no braking/SoC model, no U-4, no paper). **Verdict: ready → U-4
can consume the cycles.** All outputs are **simulation** inputs.

> **Revision note (U-4 escalation of review item C-1):** `condition_speed` now uses a
> **speed-dependent** propulsion limit read from FASTSim's own `pwr_prop_fwd_max_watts`, replacing the
> constant-budget approach that clipped ~23% of AGGRESSIVE-motorway acceleration steps vs ~9% of
> NORMAL and biased the per-style energy table. See §2A.

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

## 2A. Speed-dependent conditioning (U-4 escalation of C-1)

- **Problem found in U-4.** The conditioner used a **speed-independent** power budget
  (`0.131 × 239 kW = 31.3 kW`). That is right at launch (motor torque-limited) but wrong at
  20–30 m/s, where the vehicle can really deliver ~234 kW. It therefore flattened exactly the strong
  accelerations that define aggressive driving: **~23% of AGGRESSIVE-motorway steps clipped vs ~9%
  of NORMAL (+14 pp)** — a style-correlated bias that understated AGGRESSIVE Wh/km.
- **New model.** Each 1 Hz step is capped by **two** limits, keeping the smaller next speed:
  1. **Propulsion power**, taken from **FASTSim's own capability**: a one-time probe runs the U-1
     vehicle up a gentle full-range speed ramp and records `pwr_prop_fwd_max_watts` vs achieved speed
     → `P_avail(v)` (cached). The step satisfies `m·(v−prev)·v ≤ PROP_BUDGET · P_avail(prev)`
     (`PROP_BUDGET = 0.7`), evaluated at **`prev`, the start-of-step speed** — mirroring FASTSim, whose
     `pwr_prop_fwd_max` is a function of the current speed (from rest only ~45 kW is available). This
     governs high speed.
  2. **Acceleration**, `v ≤ prev + A_MAX (3.0 m/s² ≈ 0.3 g)`. The power limit alone permits a very
     steep launch ramp (~3.9 m/s²) that FASTSim's quasi-static solver cannot follow from a cold start
     (it stalls → "failed to meet speed trace"); the cap keeps the launch ramp followable and is a
     realistic maximum sustained acceleration. This governs the launch/low-speed ramp.
  (The degenerate `P=0` point at `v=0` is dropped from the probed curve so the launch budget clamps to
  the real ~45 kW launch capability instead of 0.)
- **Effect.** Launches are followable and strong high-speed accelerations pass through. Per-trip
  acceleration clipping falls from the constant-budget baseline to **~0% (Normal)** and **~2.1%
  (Aggressive)** — excess ≈ 2 pp vs the original **+14.3 pp** style bias, which is removed.
  Conditioned distances match raw; the FASTSim sim now runs clean (no silent launch failure).
- **No duplicated logic / architecture preserved.** The probe reuses `cycle_from_arrays` (with a new
  `condition=False` flag to avoid recursion) and the U-1 `build_mendeley_bev()` vehicle; the public
  entry points (`condition_speed`, `cycle_from_arrays`) are unchanged, so U-3/U-4 consume it as before.
- **Comparison with the constant-budget approach:** at low speed the two agree (`0.7·P_avail(≈45 kW)
  ≈ 31 kW`), so launches/spike-handling are unchanged; they diverge at speed, where the new model
  correctly permits the higher power the vehicle actually has.

*(The prior constant-budget scheme — `ACCEL_POWER_FRACTION`, `_max_fwd_prop_power_w` — is removed. The
scalar `vehicle_config.max_fwd_propulsion_power_w()` is retained; U-4 still uses it for reporting.)*

## 2. Earlier pass (review fix **C-1**, superseded by §2A)

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
4. **Conditioning uses two limits** (§2A): a speed-dependent propulsion-power budget
   (`PROP_BUDGET = 0.7` of FASTSim's own `P_avail(prev)`, probed from the U-1 vehicle) governing high
   speed, and a `A_MAX = 3.0 m/s²` (~0.3 g) acceleration cap governing the launch/low-speed ramp so
   FASTSim's quasi-static solver can follow it. Both are documented, tuneable defaults.
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
- `u2_cycle_adapter.py` — **speed-dependent conditioning (§2A):** `_prop_power_curve()` probes
  FASTSim's `pwr_prop_fwd_max_watts`, `_avail_prop_power_w()` interpolates it, `condition_speed` caps
  each step against `PROP_BUDGET · P_avail(v)`; `cycle_from_arrays` gains a `condition=False` flag for
  the probe. Removed the constant-budget `ACCEL_POWER_FRACTION` / `_max_fwd_prop_power_w`.
- `vehicle_config.py` — `max_fwd_propulsion_power_w()` retained (used by U-4 reporting). *(U-1 builder
  unchanged.)*

**Regenerated:** `u2_validation.json`, `cycles/` (git-ignored). **Not modified:** U-0, U-1 vehicle,
braking/SoC models, U-4 analysis code (`u4_style_analysis.py`, `sim_config.py`).

> **Note for the U-4 rerun — done:** `sim_config.accel_power_fraction` was updated to
> `prop_budget`/`accel_cap_mps2` (mirroring this fix's real tunables) and the full 40-trip U-4
> sweep was regenerated against this fix. Result: the clipping gate now **passes** (AGGRESSIVE
> motorway clip 23.21%→0.96%, excess +14.31pp→+0.26pp). See `U4_FINDINGS.md` for the complete
> old-vs-new comparison.
