# Implementation Log

A chronological engineering journal. One entry per completed roadmap task. Newest at the
bottom. Every metric quoted here is reproducible from the repository unless explicitly
flagged as provisional.

---

## B1 — SoC scaling and metric reconciliation

- **Task ID:** B1 (Owner: Dhananjay)
- **Date:** 2026-07-05
- **Objective:** Make SoC targets, model output, and reported metrics live on one
  consistent scale, and produce a single, reproducible source for the SoC results table
  (paper Table 6). Eliminate the "paper says RMSE 0.088, pipeline prints RMSE ≈ 73"
  contradiction.

- **Files Modified:**
  - `modules/soc/data/preprocess_real_data.py` — scale SoC to [0,1]; persist scale; fix a
    blocking `PROJECT_ROOT` path bug; set `CAPACITY_AH = 240`.
  - `modules/soc/models/adaptive_ensemble.py` — load data via shared `DatasetLoader`.
  - `modules/soc/models/physics_informed_soc.py` — load data via `DatasetLoader`;
    parameterise sequence length (was hardcoded 25).
  - `modules/soc/models/multi_objective_ga_optimizer.py` — load data via `DatasetLoader`.
  - **New:** `modules/soc/evaluate_soc.py` — single source of Table 6.
  - **New:** `modules/soc/data/soc_scale.json` — persisted scale metadata (generated).

- **Summary of Changes:**
  1. SoC (recorded as a physical percentage, 0–100) is now divided by 100 once, at
     preprocessing, so the model's `Sigmoid` head ([0,1]) and the target live on the same
     scale. The transform is a fixed affine map (`/100`), **not** a data-driven min/max
     fit, so `0`/`1` mean the same physical SoC across every split. Scale is written to
     `soc_scale.json`.
  2. `evaluate_soc.py` loads the real test set, runs the deployed model, inverse-transforms
     **both** predictions and targets back to % SoC through the one shared definition, and
     writes `modules/soc/models/table6.csv`. It also asserts predictions/targets are in
     [0,1] to catch any future scale regression.
  3. The ensemble, constraint-regularised, and multi-objective-GA models previously loaded
     an orphaned `*_soc.npy` file convention (length-25, [0,1]) that **no preprocessing
     script in the repo produced** and that was not on disk. All three now load the same
     real `*_real.npy` set via `shared/dataset_loader.py`, and the length-25 truncations
     were removed (full window length 50 is preserved).

- **Why the Change Was Necessary:** Targets were saved on 0–100 while the model output was
  [0,1]; MSELoss against mismatched scales made the deployed model un-fittable (RMSE ≈ 73,
  MAPE ≈ 98%), while the GA/ensemble JSONs quoted RMSE ≈ 0.088–0.11 from a different,
  unreproducible dataset. Reviewers flagged this as a reproducibility failure. One scale +
  one evaluator makes every SoC number traceable.

- **Validation Performed (lightweight):**
  - Regenerated the dataset: 15/16 source folders processed (Folder16 `.mat` is corrupt —
    pre-existing, skipped gracefully); scaled SoC range verified in [0.143, 0.971].
  - Retrained `LSTMCNNAttentionSoC` on the corrected scale (verification pass, subset,
    15 epochs) and ran `evaluate_soc.py` → **RMSE 10.64 % SoC, MAE 9.50 % SoC, MAPE 14.54 %**.
  - Confirmed the saved checkpoint is a raw `state_dict` and loads the way
    `EnhancedEVPipeline._load_soc_model` expects.
  - Smoke-tested the rewired ensemble / physics / multi-objective-GA code paths end-to-end
    on the real data (no crashes, correct shapes at window length 50).

- **Remaining Limitations (must be honest in the paper):**
  - **The 10.64 % RMSE is PROVISIONAL.** It is measured on a random `train_test_split`
    over 50 %-overlapping windows, which leaks neighbouring windows across splits (fixed in
    **B5**). Treat it as a sanity figure, not a paper result.
  - The checked-in `.pth` was produced by a one-off verification script, not a documented
    repo command. `evaluate_soc.py` reproduces the *metric* from that `.pth`, but a clean
    checkout cannot yet regenerate the `.pth` itself (`train_soc.py` still needs the B3
    cleanup). Full clean-checkout reproducibility of the artifact is not yet satisfied.
  - Feature (voltage/current/temp) normalisation is still per-trip and not persisted;
    revisited under B5.

- **Paper Sections Affected:** §4.2 (SoC method), §5.4, Table 6, Fig. 4.

- **Future Follow-up Items:** B3 (make `train_soc.py` the canonical, correct trainer so the
  `.pth` is reproducible), B5 (leak-free split → final Table 6 numbers).

---

## B2 — Battery parameter correction

- **Task ID:** B2 (Owner: Dhananjay)
- **Date:** 2026-07-05
- **Objective:** Replace the impossible 18650-cell battery constants (100 Ah, 0.05 Ω,
  3.7 V) with values grounded in the real EV pack, identify the ECM parameters from data,
  and soften "PINN"/"physics-informed" language to what the model actually does.

- **Files Modified:**
  - `modules/soc/models/physics_informed_soc.py` — `BatteryPhysicsParams` now loads
    identified values; added `r1_ohm`, `c1_farad`; renamed PINN language to
    "constraint-regularised."
  - **New:** `modules/soc/models/battery_rls_identification.py` — RLS parameter ID.
  - **New:** `modules/soc/models/battery_params_identified.json` — identified params
    (generated).

- **Summary of Changes:**
  1. **Capacity:** `240 Ah`, taken from the dataset authors' own analysis script
     (`_code/fig_6_11_12/fig_6_11_12.m`, line `Cap = 240`). Not published in the README;
     this is the value the dataset's paper uses for its C-rate figures.
  2. **Voltage:** pack-level `[363.8, 456.2] V`, mean `424.2 V` — observed extrema/mean
     across all readable cycles (not a single-cell 2.5–4.2 V range).
  3. **ECM (first-order Thevenin):** identified by recursive least squares on each cycle.
     The ECM is rewritten in OCV-residual ARX(1,1) form and inverted to
     (R0, R1, C1). Result across 13/15 usable cycles:
     **R0 = 0.0246 ± 0.0148 Ω, R1 = 0.0513 ± 0.0528 Ω, C1 = 848 ± 1093 F.**
  4. Renamed "Physics-Informed"/PINN wording → "constraint-regularised" throughout the
     module (class names left intact to avoid touching importers).

- **Why the Change Was Necessary:** The hardcoded constants described an 18650 cell while
  the data is a ~400 V EV pack — a reviewer-flagged impossibility (3.5). Grounding the
  parameters in the real data (and identifying the ECM rather than guessing) is required
  for the physics claims to be defensible.

- **Validation Performed (lightweight):**
  - Ran `battery_rls_identification.py` end-to-end; 13/15 cycles gave physically valid
    (positive R0, R1; 0 < pole < 1) fits.
  - Sanity check: R0 ≈ 0.025 Ω × 800 A peak ≈ 20 V IR-drop, consistent with the ~90 V
    observed voltage swing.
  - `BatteryPhysicsParams()` loads the identified values; `test_physics_constraints()`
    runs; a short verification training of the constraint-regularised model completes
    without error.

- **Remaining Limitations (must be honest in the paper):**
  - **Only R0 is actually consumed** by `BatteryPhysicsConstraints` (IR drop). R1/C1 are
    identified and stored for the appendix but do **not** yet influence any prediction. Do
    not claim the model uses a 2nd-order ECM — it is first-order and only R0 is wired in.
  - **R1/C1 are weakly identified** (C1 std > mean): the RC branch is poorly excited in
    field data. Report with spread; do not present as precise.
  - **OCV(SoC) is a proxy** fit from pooled low-current samples (no rest-test OCV exists in
    this dataset) and forced monotonic. The identified params inherit this approximation.
  - **Chemistry / series-parallel configuration are unknown** — not disclosed anywhere in
    the public dataset metadata; deliberately left `null`, not fabricated.
  - SoH and thermal parameters remain literature-typical placeholders (flagged in code);
    they are not identifiable from this dataset.

- **Paper Sections Affected:** §3.5 (battery model), §4.2, parameter table + Appendix.

- **Future Follow-up Items:** decide whether to wire R1/C1 into a genuine 2nd-order ECM
  constraint (stretch), or keep them as reported appendix values; reconcile
  `coulomb_counting.py` capacity (still 2.0 Ah) under B3/B4.

---
