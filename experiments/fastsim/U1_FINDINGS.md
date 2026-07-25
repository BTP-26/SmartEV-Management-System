# U-1 — Vehicle Parameterization: Findings

**Date:** 2026-07-25 · **Scope:** U-1 only (no U-2 cycle adapter, no coupling wrapper, no braking/SoC/paper).
**Verdict: ready → PROCEED TO U-2.** A reusable, Mendeley-like BEV is built, physically coherent,
serialized, and validated. All outputs are **simulation**.

---

## 0. Requirements review (housekeeping)

- **`requirements.txt` was NOT modified.** No Track-U dependency was added to it.
- **Recommendation: FASTSim belongs in a dedicated venv, not the main requirements.** Reasons:
  installing `fastsim` downgrades the main pipeline's **numpy (2.4.6→2.3.2)** and **PyYAML
  (6.0.3→6.0.2)** and pulls a large notebook/dev toolchain (jupyter, ipykernel, debugpy, polars)
  the SoC/braking pipeline does not use. Mixing them risks silently changing the training pipeline's
  pinned deps.
- **Created** `experiments/fastsim/requirements-fastsim.txt` declaring the single genuine dependency
  `fastsim==3.0.6` (pip resolves its transitive deps). Transient jupyter/dev tooling is deliberately
  **not** declared — it is an install artifact, not a project dependency.

---

## 1. Implementation summary

- Selected the **2022 Tesla Model 3 RWD** built-in as the baseline BEV (closest ~400 V-class BEV;
  validated in U-0).
- Resized its battery pack to approximate the **Mendeley pack (~101.8 kWh)** and adjusted the two
  physical consequences of a larger pack — **mass** and **pack power** — with documented assumptions.
  Everything else (motor, aero/rolling chassis, regen map, thermal, aux) is inherited unchanged.
- Built a **reusable builder** (`vehicle_config.build_mendeley_bev()`) and **serialized** the
  configured vehicle to `mendeley_like_bev.yaml` so U-2..U-4 load one source of truth (no duplication).
- Verified the parameterized vehicle **simulates cleanly** on UDDS and **reloads** from YAML with the
  correct pack.

## 2. Parameter comparison (baseline → parameterized)

| Parameter | Baseline (Tesla M3 RWD) | Mendeley-like BEV | Status |
|---|---|---|---|
| Battery capacity | 54.0 kWh | **101.81 kWh** | **MODIFIED** (×1.885 → Mendeley 240 Ah × 424 V) |
| Pack max power | 201.2 kW | **379.4 kW** | **MODIFIED** (scaled ×1.885; cf. Mendeley observed peak ~351 kW) |
| Vehicle mass | 1752 kg | **2070.8 kg** | **MODIFIED** (+318.8 kg for the larger pack) |
| Motor max power | 239 kW | 239 kW | inherited |
| Drag coefficient | 0.23 | 0.23 | inherited |
| Frontal area | 2.22 m² | 2.22 m² | inherited |
| Rolling-resistance coef | 0.007 | 0.007 | inherited |
| Drive type | RWD | RWD | inherited |
| Aux base load | 250 W | 250 W | inherited |
| Regen / thermal model | Model 3 | Model 3 | inherited |
| Nominal voltage | — | — | **not modeled** (fastsim-3 RES = energy reservoir) |
| Cell chemistry / config | — | — | **not modeled** (unknown for Mendeley) |

**UDDS sanity check (baseline → parameterized), confirming physical coherence:**
SoC drop 2.09% → **1.24%** (≈½, bigger pack ✓); energy 1.13 → **1.26 kWh** (heavier ✓); efficiency
10.6 → **9.5 km/kWh** (heavier ✓); regen 647 → **775 Wh** (more KE recovered ✓); no NaNs; reload OK.

## 3. Assumptions made

1. **Baseline vehicle** = Tesla Model 3 RWD (closest built-in ~400 V BEV; validated U-0).
2. **Pack specific energy = 150 Wh/kg** → +47.8 kWh adds +318.8 kg (typical modern pack-level figure).
3. **Pack power scales linearly with capacity** (same chemistry ⇒ ~constant C-rate) → 379 kW, close
   to Mendeley's observed ~351 kW peak discharge.
4. **Motor inherited at 239 kW** → the vehicle is **motor-limited to 239 kW** (below the 379 kW pack).
5. **Aero/rolling inherited** from the Model 3 (drag 0.23) despite the heavier mass — reasonable for a
   large aero sedan, but mildly optimistic; documented.
6. **No nominal voltage/current** (fastsim-3) → battery current stays derived (P/V) downstream.
7. **Chemistry unknown** (Mendeley DS-4) — not modeled.
8. Both mass and power adjustments are **toggleable** (`adjust_mass`, `scale_pack_power`) for a
   minimal-assumption fallback variant.

## 4. Limitations to carry into U-2

- The vehicle is a **"100 kWh pack on a Model-3-class chassis/motor"** — a deliberate stand-in, not a
  specific real 100 kWh vehicle. Absolute energy/SoC magnitudes are **illustrative**, not
  vehicle-accurate. (Refinement: adopt a true large-EV chassis/motor, or raise the motor to reach
  Mendeley's full ~350 kW.)
- **Motor caps propulsion + regen at 239 kW**, so the vehicle never uses the pack's full 379 kW.
- **Current is derived**, not native — downstream current in U-3/U-4 is approximate (P/V).
- None of these **block** U-2, which only needs a working, reusable vehicle — which now exists.

## 5. Recommendation

**Proceed to U-2 (UAH GPS → drive-cycle adapter).** Carry forward: (a) install FASTSim in a dedicated
venv (`requirements-fastsim.txt`); (b) the `mendeley_like_bev.yaml` reusable vehicle; (c) the derived-
current + illustrative-magnitude caveats. U-2 will also need the **UAH dataset downloaded** (absent
from the repo, per U-0).

## 6. Files

**Created (all in `experiments/fastsim/`):**
- `requirements-fastsim.txt` — dedicated FASTSim env (requirements-review outcome).
- `vehicle_config.py` — reusable builder `build_mendeley_bev()` + `save_vehicle()` (source of truth).
- `u1_parameterize.py` — driver: build, sanity-sim, comparison.
- `mendeley_like_bev.yaml` — serialized configured vehicle (reused by U-2..U-4).
- `u1_vehicle_params.json` — machine-readable meta + baseline-vs-parameterized comparison.
- `U1_FINDINGS.md` — this file.

**Modified:** none. **`requirements.txt`:** unchanged (by recommendation).
