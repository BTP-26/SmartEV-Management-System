# U-4 — Per-Style Energy Analysis: Findings

**Date:** 2026-07-28 · **Scope:** U-4 (analysis layer only; no U-1/U-2/U-3 physics modified, no
braking/SoC/paper work). Every number here is **SIMULATION**.

> ## ⚠️ Headline: the harness is complete and working, but the per-style table is **NOT yet reportable**.
> The built-in **clipping gate failed**: `condition_speed` suppresses **23.2%** of acceleration steps
> on AGGRESSIVE motorway trips versus **8.9%** on NORMAL (**+14.3 pp**). The speed-trace conditioner
> is therefore compressing precisely the contrast this table exists to measure. Review item **C-1
> must be escalated to a speed-dependent power budget** before these numbers are published.

---

## 1. What was built

A complete, auditable analysis layer (`u4_style_analysis.py` + `sim_config.py`) that:

- **Enumerates its own trips** over (driver × style × road type) — 40 trips, 6 drivers — instead of
  `u2._pick_trips()`'s single arbitrary trip per style (closes **C-3 / R-6**).
- **Matches on road type**, never pooling MOTORWAY with SECONDARY (the confound C-3 identified).
- **Detects GPS gaps** before the adapter silently interpolates across them (closes **R-2**).
- **Quantifies conditioner clipping per trip and per style** — the C-1 gate (new).
- **Labels every row** `oracle_upper_bound=True` + `simulation=True` (closes **R-1**).
- **Anchors on built-in UDDS/HWFET** reference cycles, so the pipeline runs with no external data
  (closes **R-5**).
- Centralizes every tunable in `sim_config.py` (closes **N-3**) and persists per-run time series to
  `.npz` (closes **N-2**).

## 2. Sweep results (PROVISIONAL — see the gate)

40 trips enumerated · **37 included** · 3 excluded (below). Road-type matched, mean ± std.

| Style | Road | n | Wh/km (baseline) | Saved % | **Clip %** |
|---|---|---|---|---|---|
| NORMAL | MOTORWAY | 5 | 136.0 ± 10.2 | 0.36 | 8.90 |
| AGGRESSIVE | MOTORWAY | 6 | 163.4 ± 10.8 | 2.77 | **23.21** |
| DROWSY | MOTORWAY | 5 | 124.0 ± 8.3 | 0.61 | 5.83 |
| NORMAL | SECONDARY | 11 | 122.4 ± 9.9 | 0.36 | 5.71 |
| AGGRESSIVE | SECONDARY | 4 | 139.1 ± 14.3 | 2.09 | **13.86** |
| DROWSY | SECONDARY | 6 | 131.9 ± 9.2 | 0.54 | 6.30 |

Reference anchor (dataset-independent): **UDDS 105.20 Wh/km** (regen 774.9 Wh, clip 0.73%) and
**HWFET 107.74 Wh/km** (regen 270.7 Wh, clip 0.13%). On both, the controller **did not intervene**
(`intervened=False`, saving 0.00%) — the standardized cycles contain no braking event above the
45 kW significance trigger where early coasting also lowered Wh/km. That is an honest null result
and a useful contrast: the anticipation benefit appears on real aggressive driving, not on smooth
standard cycles.

Directionally the UAH ordering is sensible (aggressive > normal on both road types; ~+20% on
motorway), and anticipation helps aggressive driving most — but see §3 before citing any of it.

## 3. The blocking finding — clipping is style-correlated

| Road | AGGRESSIVE clip | NORMAL clip | Excess | Verdict |
|---|---|---|---|---|
| MOTORWAY | 23.21% | 8.90% | **+14.31 pp** | **BIASED** |
| SECONDARY | 13.86% | 5.71% | **+8.15 pp** | **BIASED** |

**Why this happens.** `condition_speed` caps acceleration using a **speed-independent** power budget
(`ACCEL_POWER_FRACTION 0.131 × 239 kW = 31.3 kW`). That budget is justifiable at launch, where the
motor is torque-limited, but it is also applied at 20–30 m/s where the vehicle can really deliver
~234 kW. Aggressive driving is defined by exactly the hard accelerations this suppresses, so roughly
**1 in 4** aggressive motorway steps is being flattened versus **1 in 11** normal steps.

**Effect on the numbers.** Clipping removes propulsion demand, so **AGGRESSIVE Wh/km is understated**
— the true aggressive-vs-normal gap is likely *larger* than the +20% reported. Saved-% is also
mildly inflated because its denominator is understated.

**Why the earlier estimate missed it.** My pre-implementation check measured 0.7% (UDDS) and 0.1%
(HWFET). Standardized cycles are far gentler than real aggressive human driving, so that proxy
under-estimated the real rate by roughly **30×**. This is precisely why the gate was made a
mandatory step rather than an assumption.

**Required fix (goes back to U-2 / Siddharth).** Replace the constant budget with a **speed-dependent**
limit — FASTSim already exposes `veh.history.pwr_prop_fwd_max_watts`, which varies with speed — or
model `P_avail(v) = min(T_max·v, P_max)`. Then re-run this sweep; the gate should pass.

## 4. Excluded trips (3 of 40) — auditable

| Driver | Style | Road | Reason |
|---|---|---|---|
| D2 | DROWSY | MOTORWAY | powertrain limit: braking demanded −253 kW vs a 239 kW machine |
| D5 | AGGRESSIVE | SECONDARY | powertrain limit: braking demanded −262 kW |
| D3 | NORMAL | MOTORWAY | GPS gap 10.09 s > 10 s threshold (R-2) |

**A second real finding.** Two trips brake harder than the electric machine can absorb. A real vehicle
blends **friction braking** for the excess; FASTSim-3's BEV model raises an error instead. We exclude
rather than distort the trace, and the exclusions are spread across styles (1 DROWSY, 1 AGGRESSIVE,
1 NORMAL), so no systematic style bias is introduced. **Friction-brake blending is a worthwhile
future refinement for U-2/U-3.**

## 5. Assumptions

Counterfactual EV (UAH drivers were never in this vehicle) · flat road, grade = 0 · battery current
derived as P/424.22 V · **oracle intent** (derived from the true future trace) · **outcome-gated
controller** (anticipatory kept only when it lowers Wh/km ⇒ savings are an **upper bound**, ≥ 0 by
construction) · speed-independent conditioning budget (**the failing gate**) · quasi-static 1 Hz ·
single vehicle parameterization.

## 6. Status of remaining review items

| Item | Status |
|---|---|
| C-3 road-type confound | ✅ closed — matched enumeration, 6 drivers |
| R-1 upper-bound labeling | ✅ closed — flags on every row |
| R-2 GPS gap detection | ✅ closed — 1 trip excluded |
| R-5 reference cycles | ✅ closed — UDDS/HWFET anchor |
| R-6 enumeration robustness | ✅ closed — graceful, auditable manifest |
| N-2 / N-3 | ✅ closed — `.npz` persistence, `sim_config.py` |
| **C-1 residual** | ❌ **escalated — gate failed, blocks publication** |
| R-3 power-column fallback | ⬜ still open (cosmetic) |
| R-4 reproducibility | ✅ improved — `.venv-fastsim` created and used |

## 7. Next steps

1. **Blocking:** speed-dependent conditioning budget in `u2_cycle_adapter.condition_speed` (Siddharth
   or coordinated), then re-run `u4_style_analysis.py` and confirm the gate passes.
2. Run the sensitivity sweep (`SENSITIVITY_GRID` in `sim_config.py`) to show conclusions are stable.
3. Replace oracle intent with the real braking model once the Track-A campaign lands.
4. Consider friction-brake blending so hard-braking trips need not be excluded.
5. Only then treat `u4_style_table.csv` as the Table 5 replacement.

## 8. Files

**Created (all in `experiments/fastsim/`):** `sim_config.py`, `u4_style_analysis.py`,
`u4_results.csv` (39 rows), `u4_style_table.csv` (6 groups), `u4_trip_manifest.json`,
`u4_figures/` (3 PNGs), `u4_timeseries/` (39 `.npz`, gitignored), `U4_FINDINGS.md`.

**Modified:** `.gitignore` (one entry for the regenerable `u4_timeseries/`).
**Not modified:** U-0/U-1/U-2/U-3 code, braking, SoC, `shared/`, `config/`, `requirements.txt`.
