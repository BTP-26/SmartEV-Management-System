# U-4 — Per-Style Energy Analysis: Findings

**Date:** 2026-07-30 (rev. post U-2 speed-dependent conditioning fix) · **Scope:** U-4 (analysis
layer). No U-1 vehicle, braking/SoC model, or paper work. Every number here is **SIMULATION**.

> ## ✅ Headline: the publication gate now PASSES. The per-style table is reportable.
> U-2's speed-dependent conditioning fix (see `U2_FINDINGS.md` §2A) removed the style-correlated
> clipping bias this document previously blocked on. AGGRESSIVE-motorway clipping fell from
> **23.21% → 0.96%**; the excess over NORMAL fell from **+14.31 pp → +0.26 pp** — well inside the
> 5 pp gate threshold. This section documents the fix's effect precisely, old vs. new, rather
> than simply asserting the table is now fine.

---

## 1. What this revision verifies

This is a **regeneration**, not a new build: the same harness (`u4_style_analysis.py` +
`sim_config.py`) that produced the original provisional table was rerun unchanged against the
merged U-2 fix. Two stale references were corrected first (the old `ACCEL_POWER_FRACTION` mirror
in `sim_config.py`, replaced by `prop_budget`/`accel_cap_mps2`; two `CFG.accel_power_fraction`
lookups in `u4_style_analysis.py`'s reporting code) — neither affects the simulation itself, both
were needed only because the config field they referenced was renamed upstream.

**Independently re-verified before trusting the new numbers:**
- The regenerated `u4_trip_manifest.json`'s `clipping_gate` was read directly (not the console
  log): **`passed: true`** on both road types.
- **Full-pipeline reproducibility**: reran the entire 40-trip sweep twice; `u4_results.csv`,
  `u4_style_table.csv`, and `u4_trip_manifest.json` are **byte-identical** across runs.
- **Aggregation cross-check**: recomputed per-(style, road) mean/std independently from the raw
  `u4_results.csv` rows — matches `u4_style_table.csv` exactly.
- **Zero NaNs** across all 39 result rows (2 reference + 37 UAH).
- **No orphaned artifacts**: `u4_timeseries/` contains exactly 39 `.npz` files, matching 39 result
  rows — expected, since the exclusion set is unchanged (see §3).

---

## 2. Old vs. new — the complete comparison

### Clipping rate (the mechanism that changed) and gate outcome

| Road | Aggressive clip (old→new) | Normal clip (old→new) | Excess pp (old→new) | Gate |
|---|---|---|---|---|
| MOTORWAY | 23.21% → **0.96%** | 8.90% → 0.70% | +14.31 → **+0.26** | FAILED → **PASSED** |
| SECONDARY | 13.86% → **1.59%** | 5.71% → 1.02% | +8.15 → **+0.57** | FAILED → **PASSED** |

### Per-style energy table (baseline Wh/km, saved %, clip %) — old vs. new

| Style | Road | Wh/km (old→new) | Saved % (old→new) | Clip % (old→new) |
|---|---|---|---|---|
| NORMAL | MOTORWAY | 136.00 → 135.97 | 0.36 → 0.37 | 8.90 → 0.70 |
| NORMAL | SECONDARY | 122.40 → 122.62 | 0.36 → 0.41 | 5.71 → 1.02 |
| AGGRESSIVE | MOTORWAY | 163.38 → **164.84** | 2.77 → 2.92 | 23.21 → 0.96 |
| AGGRESSIVE | SECONDARY | 139.12 → **140.80** | 2.09 → 2.46 | 13.86 → 1.59 |
| DROWSY | MOTORWAY | 124.00 → 124.12 | 0.61 → 0.67 | 5.83 → 0.78 |
| DROWSY | SECONDARY | 131.95 → 131.96 | 0.54 → 0.56 | 6.30 → 1.05 |

### Reference cycles (dataset-independent anchor)

| Cycle | Clip % (old→new) | Wh/km (old→new) | Intervened (old→new) |
|---|---|---|---|
| UDDS | 0.73% → 0.00% | 105.20 → 105.22 | False → False |
| HWFET | 0.13% → 0.00% | 107.74 → 107.74 | False → False |

### Exclusion set

**Unchanged — same 3 of 40 trips, same reasons, same exact values:**

| Driver | Style | Road | Reason |
|---|---|---|---|
| D2 | DROWSY | MOTORWAY | powertrain limit: braking demanded −253 kW vs a 239 kW machine |
| D5 | AGGRESSIVE | SECONDARY | powertrain limit: braking demanded −262 kW |
| D3 | NORMAL | MOTORWAY | GPS gap 10.09 s > 10 s threshold |

This is exactly as predicted: the fix only touches **acceleration** conditioning, never
deceleration, so trips excluded for braking-power or GPS-gap reasons are unaffected. Confirmed
empirically (identical exclusion list and identical numeric values), not merely assumed.

### What changed, in plain terms

1. **The clipping bias is resolved.** AGGRESSIVE clip rates fell by roughly **24×** (motorway) and
   **9×** (secondary); the style-correlated excess collapsed from double digits to well under 1 pp.
2. **Point estimates barely moved, and moved in the predicted direction.** AGGRESSIVE Wh/km rose
   slightly (+0.9% motorway, +1.2% secondary) — consistent with the original finding that clipping
   was *understating* aggressive energy use by suppressing hard accelerations. NORMAL/DROWSY are
   essentially flat (≤0.2 Wh/km), as expected since their clip rates were already low.
3. **No directional conclusion changed.** Aggressive was already the highest-energy, highest-saving
   style before the fix, and remains so — the fix makes this trustworthy rather than reversing it.
4. **Reference-cycle clipping dropped to exactly 0%** — standard cycles are gentle enough that the
   corrected, higher-at-speed propulsion budget accommodates them with no clipping at all. No
   intervention on either cycle, unchanged from before (this is a braking-controller property,
   untouched by the acceleration fix).

---

## 3. Assumptions (unchanged from the provisional pass)

Counterfactual EV (UAH drivers were never in this vehicle) · flat road, grade = 0 · battery current
derived as P/424.22 V · **oracle intent** (derived from the true future trace) · **outcome-gated
controller** (anticipatory kept only when it lowers Wh/km ⇒ savings are an **upper bound**, ≥ 0 by
construction) · quasi-static 1 Hz · single vehicle parameterization · two trips excluded because
this vehicle's electric machine cannot absorb their peak braking demand (no friction-brake
blending modeled).

## 4. Status of review items

| Item | Status |
|---|---|
| **C-1 clipping bias** | ✅ **Resolved** — speed-dependent conditioning (U-2), gate passes on regenerated data |
| C-3 road-type confound | ✅ closed — matched enumeration, 6 drivers |
| R-1 upper-bound labeling | ✅ closed — flags on every row |
| R-2 GPS gap detection | ✅ closed — 1 trip excluded |
| R-5 reference cycles | ✅ closed — UDDS/HWFET anchor, now 0% clip |
| R-6 enumeration robustness | ✅ closed — graceful, auditable manifest |
| N-2 / N-3 | ✅ closed — `.npz` persistence, `sim_config.py` (fields now mirror the current U-2 model) |
| R-3 power-column fallback | ⬜ still open (cosmetic) |
| R-4 reproducibility | ✅ improved — `.venv-fastsim`; full-pipeline determinism now directly verified (byte-identical reruns) |

## 5. Remaining limitations / follow-up (not blocking this table)

- **Oracle intent** stands in for the real braking model — final numbers should be reproduced once
  the Track-A campaign lands and intent comes from an actual trained classifier.
- **Two hard-braking trips remain excluded** because the vehicle's electric machine cannot absorb
  their peak demand; friction-brake blending would recover them (Track-U follow-up, not attempted
  here to stay scoped to the conditioning-fix regeneration).
- **Sensitivity sweep** (`SENSITIVITY_GRID` in `sim_config.py`, now updated to mirror `prop_budget`/
  `accel_cap_mps2`) has not been executed — a natural next step to show the per-style ordering is
  stable under the new model's own tunables, not just under the single default setting reported here.
- **Single vehicle parameterization, flat road, counterfactual EV** — all inherited, unchanged,
  and already disclosed in U-1/U-2/U-3.

## 6. Readiness assessment

The regenerated evidence — a passing gate verified from the manifest (not asserted), full-pipeline
reproducibility confirmed by two independent byte-identical reruns, an unchanged and independently
re-verified exclusion set, and point estimates that moved in the direction the original diagnosis
predicted rather than arbitrarily — indicates Track U's per-style energy table is now
**scientifically ready to support the planned experimental results and paper draft**, subject to
the limitations in §5 (oracle intent in particular should be revisited once the real braking model
exists). This is a readiness assessment for the current evidence, not a claim that no further work
remains.

## 7. Files

**Modified:**
- `sim_config.py` — replaced the stale `accel_power_fraction` mirror (of a now-removed U-2
  constant) with `prop_budget`/`accel_cap_mps2`, matching U-2's actual tunables; updated
  `SENSITIVITY_GRID` to match.
- `u4_style_analysis.py` — updated two now-stale `CFG.accel_power_fraction` references (config
  rename) and the module docstring's C-1 bullet (past tense; gate reframed as a standing monitor,
  not an active-bug detector). No logic changes.

**Regenerated:** `u4_results.csv`, `u4_style_table.csv`, `u4_trip_manifest.json`, `u4_figures/*`,
`u4_timeseries/*.npz`.

**Not modified:** U-0/U-1/U-2/U-3 implementations, braking, SoC, `shared/`, `config/`,
`requirements.txt`.
