# U-0 — FASTSim Feasibility Spike: Findings

**Date:** 2026-07-25 · **Scope:** U-0 only (no U-1/U-2/U-3; no project code touched).
**Verdict: FASTSim is suitable → PROCEED TO U-1** (with one setup note: isolate FASTSim in its
own venv). All outputs here are **simulation**.

## Implementation summary

- Installed FASTSim into `.venv`; ran a built-in BEV over the built-in **UDDS** cycle and over a
  **synthetic accel→cruise→hard-brake** trace; extracted SoC / battery power / derived current /
  regen energy; sanity-checked behaviour and compared the pack to Mendeley.
- Everything is in one script, `experiments/fastsim/u0_spike.py`; results in `u0_results.json`;
  plot in `u0_soc_power.png`; API captured in `fastsim_api_reference.md`.
- **All 6 exit criteria met** (`all_exit_criteria_met: true`).

## Exact version

- **`fastsim` 3.0.6** — this is **fastsim-3** (Rust-backed, polars outputs), **not** the 2.x assumed
  in the plan. Capabilities are a superset; only the API differs (captured in the API reference).

## Results (BEV = 2022 Tesla Model 3 RWD, closest available; pack match is U-1)

| Metric | UDDS (built-in) | Synthetic accel-brake |
|---|---|---|
| Steps / duration | 1370 / 1369 s | 121 / 120 s |
| Distance | 11.99 km | 1.31 km |
| SoC start → end | 0.980 → 0.959 | 0.980 → 0.978 |
| Battery power range | −25.2 … +41.0 kW | −29.0 … +31.8 kW |
| **Regen energy** | **646.7 Wh** (37% of discharge) | **42.5 Wh** (29%) |
| Derived current range (P/424 V) | −59 … +97 A | −68 … +75 A |
| SoC rises during regen? | ✅ | ✅ |
| Energy use / efficiency | 1.13 kWh / 10.6 km·kWh⁻¹ | — |
| NaNs / blow-ups | none | none |

Numbers are physically sane for a Model 3 (regen ≈ 1/3 of urban discharge; power tens of kW;
efficiency in the right range). This confirms FASTSim emits usable, plausible battery behaviour.

## API / functions we will rely on (U-1 → U-4)

Full detail in `fastsim_api_reference.md`. Essentials:

- Load: `fsim.Vehicle.from_resource(...)`, `fsim.Cycle.from_resource(...)` / `Cycle.from_pydict(...)`.
- Run: `fsim.SimDrive(veh, cyc).walk()` → `sd.to_dataframe()` (polars).
- Columns: SoC `veh.pt_type.BEV.res.history.soc`; battery power
  `veh.pt_type.BEV.res.history.pwr_out_electrical_watts` (+dis/−regen); speed/time under
  `veh.history.*`.
- Pack params: `veh.to_pydict()["pt_type"]["BEV"]["res"]["energy_capacity_joules"]` (÷3.6e6 = kWh).
- Derived: regen energy = −Σ min(power,0)·Δt; current = power ÷ assumed voltage.

## Pack comparison (FASTSim BEV vs Mendeley)

| | FASTSim Tesla M3 | Mendeley pack |
|---|---|---|
| Capacity | **54.0 kWh** | **~101.8 kWh** (240 Ah × 424.22 V) |
| Nominal voltage | **not modeled** | 424.22 V |
| Difference | Mendeley ≈ **1.89×** the capacity; FASTSim has no voltage concept | — |

→ U-1 must resize the FASTSim pack to ~100 kWh. Absolute SoC-swing magnitudes here are **not**
comparable to Mendeley yet (different pack) — expected, and out of U-0 scope.

## Assumptions made

- BEV = 2022 Tesla Model 3 RWD as a **stand-in** (pack match is U-1).
- Speed traces = built-in UDDS + a synthetic brake trace, as a **stand-in for UAH** (UAH data is
  **absent from the repo** → real UAH ingestion is U-2, needs the download).
- Road **grade = 0** (flat); ambient **22 °C**.
- Battery **current is derived** = power ÷ 424 V (FASTSim-3 models energy/power, no V/I).
- All results are **simulation**, not measured.

## Issues discovered

1. **fastsim-3, not fastsim-2** — different API; fully captured in `fastsim_api_reference.md`.
   Not a blocker.
2. **Install perturbed the shared venv** — downgraded numpy 2.4.6→2.3.2 and PyYAML 6.0.3→6.0.2 and
   pulled jupyter/polars. The existing project **still imports** (numpy/torch/yaml/sklearn verified),
   but **for U-1+ install FASTSim in a dedicated venv** to keep the main SoC/braking pipeline's
   dependencies pinned and untouched. (Nothing in `requirements.txt` was edited.)
3. **No native battery voltage/current** — current must be derived (P/V). A documented **limitation**,
   **not** a fundamental mismatch: the coupling runs on SoC + battery power + regen energy, all native.
   (It also reinforces that FASTSim's battery model is coarse and is never a substitute for Mendeley's
   real V/I/T.)
4. **polars, not pandas** outputs — minor code adaptation (documented).
5. **Observed, not caused by this spike:** `.gitignore` was modified during the session to add
   `analysis/` (mtime 12:34) — I did not edit it; flagged and left untouched.

## Exit criteria — all met

install+example ran ✅ · SoC extractable ✅ · battery power non-trivial ✅ · regen energy > 0 (both
cycles) ✅ · SoC rises during regen ✅ · derived current sane ✅.

## Recommendation

**PROCEED TO U-1.** FASTSim reproduces the physics we need (SoC, battery power, regen) with sane,
extractable outputs and a documented API. Carry two items into U-1: (a) install FASTSim in its own
venv; (b) resize the vehicle pack to ~100 kWh to match Mendeley. No fundamental mismatch found.

## Files created (all in `experiments/fastsim/`)

- `u0_spike.py` — the spike script.
- `u0_results.json` — machine-readable metrics + exit criteria.
- `u0_soc_power.png` — speed / battery power / SoC plot (synthetic trace).
- `fastsim_api_reference.md` — API for U-1 → U-4.
- `U0_FINDINGS.md` — this file.
