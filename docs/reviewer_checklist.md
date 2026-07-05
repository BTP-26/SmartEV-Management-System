# Reviewer Checklist

A living tracker of every reviewer concern raised so far, so we address them systematically
rather than losing them in conversation history. **This document is updated, not
recreated** — when a roadmap task changes an item's status, edit that item in place and
append to its Notes; do not duplicate or re-derive entries.

**Sources so far:** the B1/B2 implementation self-review (independent-reviewer pass
performed after B1/B2 landed) and the Reviewer #2 assessment (external-reviewer-style
critique of B1/B2, scoped to scientific methodology/experimental design/reproducibility).
Every item below traces to one of those two reviews — nothing here is invented.

**Snapshot at creation (2026-07-05):** 32 tracked concerns — 0 Publication Ready,
0 Experimentally Validated, 4 Implemented, 3 In Progress, 25 Not Started.

## Status legend

| Status | Meaning |
|---|---|
| Not Started | No code, experiment, or documentation change has addressed this yet |
| In Progress | Partially addressed (e.g., honestly documented as a limitation) but the underlying concern is not resolved |
| Implemented | The code/documentation change addressing this concern is complete |
| Experimentally Validated | The fix has been verified against real data/experiments, not just implemented |
| Publication Ready | Verified *and* reflected correctly in the manuscript draft |

## Section index

- [Dataset & Labels](#dataset--labels)
- [Evaluation Protocol](#evaluation-protocol)
- [Baselines](#baselines)
- [Physics Model](#physics-model)
- [Experimental Design](#experimental-design)
- [Statistical Validation](#statistical-validation)
- [Reproducibility](#reproducibility)
- [Documentation](#documentation)
- [Paper Writing](#paper-writing)

---

## Dataset & Labels

#### DS-1 — SoC label is a BMS-derived estimate, not independently verified ground truth
- **Reviewer Concern:** The dataset's `SoC [%]` is the vehicle's own on-board BMS estimate. Training against it means the model learns to reproduce a proprietary, unvalidated algorithm's output, not a verified physical quantity.
- **Why it matters scientifically:** Caps the meaning of any accuracy claim — "10.64% RMSE" measures agreement with an estimate of unknown accuracy, not error against true SoC. This is a validity threat to the entire estimation task, not a tuning issue.
- **Roadmap task(s):** None currently named in B1–B6 explicitly; closest is B1 (label handling). Flagging as a scope gap.
- **Status:** Not Started
- **Remaining work:** State label provenance explicitly wherever SoC accuracy is reported; consider an independent coulomb-counting-anchored check as a partial validation.
- **Notes:** Reviewer #2, concern M2.

#### DS-2 — Charge and drive cycles are structurally different regimes, pooled before splitting
- **Reviewer Concern:** Charge cycles (SoC 86–92%, current always negative) and drive cycles (SoC 14–97%, bidirectional current) are concatenated into one pool before any split.
- **Why it matters scientifically:** Aggregate error conflates two distinct physical regimes and hides which one the model actually handles; a single RMSE cannot be interpreted without knowing the regime mix.
- **Roadmap task(s):** B5 (leak-free evaluation and per-segment analysis)
- **Status:** Not Started
- **Remaining work:** Tag windows by segment type (charge/drive/regen); report per-segment RMSE/MAE/R².
- **Notes:** Reviewer #2, concern M12. Roadmap B5 text explicitly calls for charge/drive/regen segment reporting. Cross-referenced with ED-2 (same underlying issue, experimental-design framing).

#### DS-3 — Pack capacity (240 Ah) sourced from a plotting-script comment, not a cited spec
- **Reviewer Concern:** `Cap = 240` comes from a line in the dataset authors' MATLAB figure-generation script, not the README or any cited vehicle/pack specification. Whether it is nominal, usable, or beginning-of-life capacity is unstated.
- **Why it matters scientifically:** This single number scales every C-rate and coulombic calculation in the physics model; weak provenance undermines everything downstream of it.
- **Roadmap task(s):** B2
- **Status:** In Progress — source is now documented and flagged in-code (was previously silently wrong: `CAPACITY_AH = 2.0`, an 18650-cell-scale value)
- **Remaining work:** Search for a citable pack/vehicle spec; if none exists, state this provenance limitation explicitly in the manuscript.
- **Notes:** Reviewer #2, concern M9; self-review flagged the same provenance weakness independently. Documented in `modules/soc/models/battery_rls_identification.py` and `docs/soc_pipeline.md` §2.

#### DS-4 — Cell chemistry and series/parallel configuration undisclosed
- **Reviewer Concern:** Neither the README nor the authors' analysis code discloses chemistry or pack cell configuration.
- **Why it matters scientifically:** Undermines any electrochemical interpretation of the OCV curve or voltage limits; the "physics" grounding stops at capacity/voltage/resistance and cannot go deeper.
- **Roadmap task(s):** B2
- **Status:** In Progress — deliberately left as `null` with a provenance note rather than fabricated
- **Remaining work:** Reflect as an explicit limitation in the manuscript's methods/limitations section.
- **Notes:** Reviewer #2, concern M8. See `battery_params_identified.json` (`"chemistry": null`) and `docs/soc_pipeline.md` §2.

---

## Evaluation Protocol

#### EV-1 — Random split over 50%-overlapping windows leaks near-duplicate samples across train/test
- **Reviewer Concern:** SoC windows (50 samples, 25-sample stride) are concatenated across all cycles and then split with `train_test_split(random_state=42)`. Neighbouring, near-identical windows land on both sides of the split.
- **Why it matters scientifically:** The headline RMSE mostly measures interpolation between memorized neighbours, not generalization. This is the single most consequential outstanding issue in the SoC results.
- **Roadmap task(s):** B5
- **Status:** Not Started
- **Remaining work:** Split by trip/session; assert no window/session crosses splits; regenerate Table 6 from the corrected split.
- **Notes:** Reviewer #2, concern M1 (leakage half); self-review Major finding #1. Already flagged as **PROVISIONAL** in `docs/implementation_log.md` (B1 entry) and `docs/soc_pipeline.md` §5.

#### EV-2 — No naive/floor baseline reported (persistence, coulomb counting)
- **Reviewer Concern:** SoC is slowly varying and near-monotonic; without a trivial reference (predict-last-value persistence, or coulomb counting), a reader cannot tell whether the network exceeds simple temporal autocorrelation.
- **Why it matters scientifically:** A reported RMSE is scientifically uninterpretable without a floor to compare against — this is standard practice for any time-series estimation claim.
- **Roadmap task(s):** B3 (baseline ladder)
- **Status:** Not Started
- **Remaining work:** Wire `coulomb_counting.py` and a persistence baseline into `evaluate_soc.py` / Table 6.
- **Notes:** Reviewer #2, concern M1 (missing-floor half).

#### EV-3 — Per-trip feature normalization removes the absolute voltage signal and is non-causal
- **Reviewer Concern:** Features are z-scored per trip before windowing. This removes the *absolute* voltage level — the single most informative electrochemical indicator of SoC via the OCV relationship — and uses whole-trip statistics that would not be available causally at inference time.
- **Why it matters scientifically:** Handicaps the model on its most informative feature, and introduces a look-ahead information leak distinct from (and additional to) the split-leakage in EV-1.
- **Roadmap task(s):** Not explicitly named in B1–B6; closest existing task is B5. Flagging as a scope gap.
- **Status:** Not Started
- **Remaining work:** Decide whether to preserve absolute/globally-normalized voltage as a feature; if per-trip or windowed normalization is kept, recompute it from past-only statistics.
- **Notes:** Reviewer #2, concern M4.

#### EV-4 — MAPE is a poor SoC metric; R² is absent
- **Reviewer Concern:** MAPE is unstable near zero and range-dependent; SoC literature standardly reports RMSE/MAE in percentage points plus R², neither of which fully substitutes here.
- **Why it matters scientifically:** A nonstandard metric choice can obscure how the model performs at low/high SoC and makes cross-paper comparison harder.
- **Roadmap task(s):** B1 (evaluate_soc.py is the metric surface; follow-up)
- **Status:** Not Started
- **Remaining work:** Add R² and error-vs-SoC-level reporting to `evaluate_soc.py`.
- **Notes:** Reviewer #2, concern m13; self-review flagged the same MAPE concern independently (Suggestion-level).

---

## Baselines

#### BL-1 — No baseline ladder exists (coulomb counting, linear regression, MLP, single LSTM)
- **Reviewer Concern:** Without same-dataset baselines of increasing complexity, the proposed model's contribution cannot be isolated from how easy/hard the dataset itself is.
- **Why it matters scientifically:** Directly addresses EV-2's missing floor and gives the reader a same-dataset ladder to calibrate the headline result against.
- **Roadmap task(s):** B3
- **Status:** Not Started
- **Remaining work:** Implement `baselines_soc.py`; wire `coulomb_counting.py`; add `LinearRegression` + 2-layer MLP; re-implement a Chemali-style single LSTM (roadmap ref. L-B1) on our split.
- **Notes:** Roadmap B3 deliverable.

#### BL-2 — Corrected result (~10.6% RMSE) is ~10x worse than published SoC SotA, unconfronted
- **Reviewer Concern:** Chemali et al. (2018) and subsequent LSTM/transformer SoC work report ≈1% RMSE. A corrected 10.64% (on a leaky, optimistic split — see EV-1) is 10x off, and the manuscript does not confront this gap.
- **Why it matters scientifically:** An unconfronted 10x gap either signals an implementation deficiency or an unsubstantiated "real driving data is harder" claim; either way it must be addressed, not silently carried forward.
- **Roadmap task(s):** B3 (same-dataset comparator); Paper Writing (interpretation)
- **Status:** Not Started
- **Remaining work:** Run the Chemali-style baseline on our (eventually leak-free) split; write an explicit discussion of the gap once B3+B5 give a trustworthy number.
- **Notes:** Reviewer #2, concern M3. Do **not** attempt to close this gap by re-fitting to the current leaky split (EV-1) — the 10.64% is provisional and likely to change (probably worsen) once corrected; must be presented honestly regardless of direction.

#### BL-3 — coulomb_counting.py capacity mismatch (2.0 Ah default vs. real 240 Ah pack)
- **Reviewer Concern:** `coulomb_counting.py` still hardcodes `capacity_ah=2.0`, an order of magnitude off the real 240 Ah pack identified in B2.
- **Why it matters scientifically:** If wired into the baseline ladder (BL-1) without correction, this would silently produce a wrong coulomb-counting reference/baseline.
- **Roadmap task(s):** B3 / B4
- **Status:** Not Started
- **Remaining work:** Update the default (or parameterize) to the real pack capacity when wiring CC into `evaluate_soc.py`.
- **Notes:** Self-review Suggestion-level finding. Currently disconnected from the real pipeline, so no active harm yet — but must be fixed before BL-1/EV-2 land.

---

## Physics Model

#### PH-1 — ECM identification is never validated against held-out terminal voltage
- **Reviewer Concern:** Parameters (R0, R1, C1) are reported as mean ± std across cycles, but the ECM is never shown to actually reconstruct measured terminal voltage on held-out data.
- **Why it matters scientifically:** Without a voltage-reconstruction check, the identified parameters are unfalsifiable — there is no evidence they describe the real system rather than an artifact of the regression.
- **Roadmap task(s):** B2 (follow-up)
- **Status:** Not Started
- **Remaining work:** Add a voltage-reconstruction RMSE on held-out cycles using the identified (R0, R1, C1) + OCV curve.
- **Notes:** Reviewer #2, concern M5. Also folds in concern m15 (sign-convention correctness of the ECM, e.g. this dataset's discharge current being positive, is currently asserted rather than empirically checked) — a voltage-reconstruction plot would double as this sanity check.

#### PH-2 — OCV(SoC) curve is a crude proxy, and R0 identification is circular through it
- **Reviewer Concern:** The OCV curve used to form the RLS residual is fit from low-current (not true rest/relaxation) samples and forced monotonic via `maximum.accumulate`, because the field data contain no rest-test OCV. OCV error propagates directly into the identified R0.
- **Why it matters scientifically:** The identification is only as good as this proxy; presenting R0/R1/C1 without this caveat overstates their reliability.
- **Roadmap task(s):** B2
- **Status:** Implemented as a documented approximation (not hidden) — underlying limitation not resolved
- **Remaining work:** State as an explicit limitation in the manuscript; consider joint state-parameter estimation if pursued further.
- **Notes:** Reviewer #2, concern M5; self-review Minor finding #5. Documented in `modules/soc/models/battery_rls_identification.py` (module docstring) and `docs/soc_pipeline.md` §7.

#### PH-3 — R1/C1 are only weakly identified (C1 std > mean); reporting them as identified is misleading
- **Reviewer Concern:** C1 = 848 ± 1093 F has a coefficient of variation > 1, meaning the RC branch is not meaningfully excited by this field data — it is not "identified," only averaged.
- **Why it matters scientifically:** Reporting these as identified parameters implies a confidence the data does not support.
- **Roadmap task(s):** B2
- **Status:** In Progress — the spread is honestly reported (mean ± std, not a point estimate); underlying weak identifiability not resolved
- **Remaining work:** Either caveat R1/C1 explicitly as "weakly identified / not reliable" wherever quoted, or move them out of any headline table into appendix-only reporting.
- **Notes:** Self-review Major finding #4; Reviewer #2 independently confirms, concern M6.

#### PH-4 — ECM parameters fit as single scalars across a wide (SoC, temperature) envelope, conflating operating-condition variation with parameter uncertainty
- **Reviewer Concern:** R0/R1/C1 are known to be strong functions of SoC and temperature; fitting one scalar per parameter across 15–33°C and 14–97% SoC means the large cross-cycle spread (PH-3) is at least partly confounded operating-condition variation, not just estimation noise.
- **Why it matters scientifically:** Misattributes real physical variation to "uncertainty," and a scalar average over this envelope is not a well-defined ECM parameter in the way battery-modeling reviewers expect.
- **Roadmap task(s):** B2 (follow-up)
- **Status:** Not Started
- **Remaining work:** Stratify identification by SoC bin and/or temperature bin, or explicitly state the reported values are a coarse operating-envelope average, not condition-resolved parameters.
- **Notes:** Reviewer #2, concern M6.

#### PH-5 — Only R0 is consumed by the model; it is a first-order ECM, not second-order
- **Reviewer Concern:** R1/C1 are identified and stored but do not influence any prediction; `BatteryPhysicsConstraints` only uses R0 (IR-drop). The roadmap's loose "2nd-order ECM" phrasing does not match what is implemented (one RC branch = first-order).
- **Why it matters scientifically:** The manuscript must not claim the model uses a 2nd-order ECM or that physics constraints beyond simple IR-drop are active — this would be an unsupported claim of exactly the kind already flagged elsewhere in this repo (roadmap intro).
- **Roadmap task(s):** B2
- **Status:** Implemented — code and docs already correctly scope this as first-order, R0-only
- **Remaining work:** Ensure the paper draft matches this exactly; separately decide (not yet decided) whether to wire R1/C1 into an actual prediction pathway (stretch goal) or keep them as appendix-only reporting.
- **Notes:** Self-review Major finding #3. Documented in `docs/soc_pipeline.md` §6–7 and `docs/implementation_log.md` (B2 entry). Directly related to PH-3/PH-4 (why R1/C1 aren't trustworthy enough to wire in yet).

#### PH-6 — "Constraint-regularised" gating is not derived from a conservation law and is never ablated
- **Reviewer Concern:** The sigmoid-based constraint gates in `PhysicsInformedSoCModel`/`EnhancedPhysicsInformedSoC` are not derived from any conservation law, and there is no ablation comparing the constrained model against an unconstrained equivalent.
- **Why it matters scientifically:** Without a constrained-vs-unconstrained ablation and an R0-sensitivity analysis, the physical framing cannot be shown to contribute anything beyond a cosmetic label — this is the same category of unsupported-methodology concern the roadmap already exists to fix elsewhere (e.g., B6's GA honesty).
- **Roadmap task(s):** B2 (follow-up) / Stage 4 ablations (roadmap Final Checklist: "Ablations complete")
- **Status:** Not Started
- **Remaining work:** Run a same-architecture ablation with constraints removed; report the accuracy delta and an R0-sensitivity analysis.
- **Notes:** Reviewer #2, concern M7. Cross-referenced with ED-1 (same fix, experimental-design framing).

#### PH-7 — Observed current percentiles reported as rated charge/discharge limits
- **Reviewer Concern:** `max_charge_rate`/`max_discharge_rate` are the 99.9th percentile of *observed* current divided by capacity — an empirical envelope, not a manufacturer/design rating — but the field names and framing imply a rating.
- **Why it matters scientifically:** Presenting observed data as a nameplate rating is a category error that could mislead a reader about the pack's actual design limits.
- **Roadmap task(s):** B2
- **Status:** Not Started
- **Remaining work:** Rename/relabel fields and any paper text to "observed 99.9th-percentile current," not implying a rated limit.
- **Notes:** Reviewer #2, concern M10.

---

## Experimental Design

#### ED-1 — No ablation studies exist anywhere in the SoC pipeline
- **Reviewer Concern:** No ablations exist yet for the physics constraints (PH-6), the adaptive ensemble, or the GA components.
- **Why it matters scientifically:** Without ablations, no individual component's contribution is isolated; a reviewer cannot tell what is actually doing the work in any multi-part system.
- **Roadmap task(s):** Stage 4 (roadmap Final Checklist: "Ablations complete (multitask, gyro, GA, ensemble)")
- **Status:** Not Started
- **Remaining work:** Constrained-vs-unconstrained physics ablation (PH-6), fixed-vs-adaptive ensemble ablation (B4), GA weight-sensitivity sweep (B6).
- **Notes:** Cross-cutting item — several B-tasks feed into this; tracked here so it isn't lost across tasks.

#### ED-2 — Charge/drive regime mixing is an experimental-design gap, not just a data-labeling gap
- **Reviewer Concern:** See DS-2. Distinctly from the labeling/segmentation issue, the *experiment* as currently designed never reports results split by regime — only one pooled number exists.
- **Why it matters scientifically:** Even after DS-2's fix (segment tagging), the experimental protocol must actually *use* that tagging to report per-regime results, or the fix is incomplete.
- **Roadmap task(s):** B5
- **Status:** Not Started
- **Remaining work:** See DS-2's remaining work; this item exists to ensure the per-segment reporting step is not skipped once tagging exists.
- **Notes:** Reviewer #2, concern M12. Intentionally cross-referenced with DS-2 rather than duplicated in detail.

---

## Statistical Validation

#### SV-1 — Single training run, single split, single random seed; no confidence intervals or significance testing
- **Reviewer Concern:** Every number reported so far (including the provisional 10.64% RMSE) comes from one run on one seed with no dispersion measure.
- **Why it matters scientifically:** A point estimate with no confidence interval cannot support any comparative or quantitative claim at a top venue — this is table-stakes for experimental ML/EV research.
- **Roadmap task(s):** Stage 4 (roadmap: "≥5 seeds, error bars, significance")
- **Status:** Not Started
- **Remaining work:** Rerun final experiments (after B3 makes training canonical and B5 fixes the split) across ≥5 seeds; report mean ± std or CI; add significance tests for any baseline comparison (BL-2).
- **Notes:** Reviewer #2, concern M11; explicitly listed in the roadmap's Final Checklist ("Statistics complete").

---

## Reproducibility

#### RP-1 — Evaluated checkpoint is not regenerable from a documented repo command
- **Reviewer Concern:** The `lstm_cnn_attention_soc.pth` behind the current Table 6 number was produced by an ad-hoc verification script, not a documented entry point in the repo.
- **Why it matters scientifically:** "Run X, get Table 6" must hold from a clean checkout for any reproducibility claim; an undocumented training run breaks that chain entirely.
- **Roadmap task(s):** B3 (make `train_soc.py` the canonical, correct trainer)
- **Status:** Not Started
- **Remaining work:** Fix `train_soc.py`'s existing bugs; make it the single documented entry point; regenerate the checkpoint from it.
- **Notes:** Self-review Major finding #2; Reviewer #2 concern m14 (checkpoint half). Flagged in `docs/implementation_log.md` (B1 entry, "Remaining Limitations").

#### RP-2 — Training subset size is arbitrary and unjustified
- **Reviewer Concern:** Verification runs so far used e.g. 30,000 of 283,278 available training windows, with no stated justification for the subsample size.
- **Why it matters scientifically:** An unexplained subsampling choice is a rigor gap — unclear whether results would hold, improve, or degrade on the full dataset.
- **Roadmap task(s):** B3
- **Status:** Not Started
- **Remaining work:** Justify or remove subsampling in the final (post-B3/B5) training protocol.
- **Notes:** Reviewer #2, concern m14 (subset half). Current subsets were explicit, documented quick-verification passes, not final experiments (see `docs/implementation_log.md`) — but this must not be the final protocol.

#### RP-3 — SoC inverse-transform is implemented twice (preprocessing and evaluation)
- **Reviewer Concern:** `inverse_scale_soc` exists independently in both `preprocess_real_data.py` and `evaluate_soc.py` rather than being imported from one place.
- **Why it matters scientifically:** Two independent implementations of the same transform can silently drift out of sync, reintroducing exactly the kind of scale bug B1 fixed.
- **Roadmap task(s):** B1 (follow-up)
- **Status:** Not Started — documented as a known minor issue, not fixed
- **Remaining work:** Import the transform from one shared location, or explicitly justify the duplication (current rationale: `evaluate_soc.py` avoids importing h5py-dependent preprocessing code).
- **Notes:** Self-review Minor finding #6.

#### RP-4 — BatteryPhysicsParams fallback constants can silently drift from the identified JSON
- **Reviewer Concern:** `_load_identified_battery_params()`'s hardcoded fallback values (used only when `battery_params_identified.json` is absent) are a snapshot of one identification run and are not auto-synced if the script is rerun with different results.
- **Why it matters scientifically:** A stale, silently-used fallback could misrepresent the identified parameters in a future run where the JSON is regenerated but the code isn't updated.
- **Roadmap task(s):** B2 (follow-up)
- **Status:** Not Started — low risk noted, not fixed (JSON is source-of-truth whenever present)
- **Remaining work:** Consider failing loudly instead of silently falling back, or auto-syncing the fallback from the JSON at commit time.
- **Notes:** Self-review Minor finding #7.

---

## Documentation

#### DC-1 — Provisional status of Table 6 / the 10.64% RMSE must stay visibly flagged
- **Reviewer Concern:** An unflagged provisional number can be mistaken for a final result at any later stage (paper draft, slides, follow-up discussion).
- **Why it matters scientifically:** Directly prevents a known-leaky number (EV-1) from silently becoming "the" reported result.
- **Roadmap task(s):** B1 / B5 (cross-cutting)
- **Status:** Implemented — flagged in `docs/implementation_log.md` (B1 entry) and `docs/soc_pipeline.md` §5
- **Remaining work:** Keep this flag in place until B5 lands and Table 6 is regenerated on a leak-free split; then update all docs and remove the caveat.
- **Notes:** N/A.

#### DC-2 — "First-order, R0-only" model status must stay explicit in all documentation
- **Reviewer Concern:** Prevents future contributors or paper drafts from silently overclaiming a 2nd-order or fully physics-driven model (PH-5).
- **Why it matters scientifically:** Same category of concern the roadmap already exists to fix (unsupported architectural claims) — must not be reintroduced by omission in later docs/paper edits.
- **Roadmap task(s):** B2 (cross-cutting)
- **Status:** Implemented — flagged in `docs/soc_pipeline.md` §6–7 and `docs/implementation_log.md` (B2 entry)
- **Remaining work:** Keep synchronized if PH-5/PH-6 change this in the future (i.e., if R1/C1 are ever wired into a prediction pathway).
- **Notes:** N/A.

---

## Paper Writing

#### PW-1 — Manuscript must state the SoC label is a BMS-derived estimate, not verified ground truth
- **Reviewer Concern:** See DS-1.
- **Why it matters scientifically:** The entire SoC estimation claim's framing depends on this being explicit.
- **Roadmap task(s):** None in B1–B6 explicitly (paper-writing stage, roadmap Stage 5)
- **Status:** Not Started
- **Remaining work:** Add an explicit label-provenance statement to the methods/limitations section.
- **Notes:** Reviewer #2, concern M2.

#### PW-2 — Manuscript must not claim a 2nd-order physics-informed model
- **Reviewer Concern:** See PH-5. Manuscript text must match the "constraint-regularised," first-order, R0-only reality.
- **Why it matters scientifically:** Overclaiming architecture sophistication is exactly the class of issue the roadmap exists to eliminate before resubmission.
- **Roadmap task(s):** B2 / Stage 5 (paper rewrite)
- **Status:** Not Started — code/docs are already correct; paper draft itself not yet audited
- **Remaining work:** Audit and update paper draft language to match code reality.
- **Notes:** Mirrors the code-level PINN→constraint-regularised rename already completed in B2.

#### PW-3 — Manuscript must disclose unknown chemistry/cell configuration as a limitation
- **Reviewer Concern:** See DS-4.
- **Why it matters scientifically:** An undisclosed gap in physical grounding, left unstated in the manuscript, reads as an oversight rather than an honest limitation.
- **Roadmap task(s):** Stage 5
- **Status:** Not Started
- **Remaining work:** Add to the manuscript's limitations section.
- **Notes:** Reviewer #2, concern M8.

#### PW-4 — Manuscript must not present observed current percentiles as rated charge/discharge limits
- **Reviewer Concern:** See PH-7.
- **Why it matters scientifically:** A category error (empirical envelope vs. design rating) that could mislead a reader if it reaches the manuscript unlabeled.
- **Roadmap task(s):** Stage 5
- **Status:** Not Started
- **Remaining work:** Audit any table/figure using these fields for correct labeling before drafting.
- **Notes:** Reviewer #2, concern M10.

#### PW-5 — Manuscript must confront the ~10x gap to published SoC SotA explicitly
- **Reviewer Concern:** See BL-2.
- **Why it matters scientifically:** Omitting this discussion would itself be a form of the unsupported-claims problem the whole revision is meant to fix.
- **Roadmap task(s):** Stage 5 (after B3 baseline + B5 leak-free split give a trustworthy number)
- **Status:** Not Started
- **Remaining work:** Write an explicit discussion paragraph once final (post-B3/B5/Stage-4) numbers exist.
- **Notes:** Reviewer #2, concern M3.

---

## Change log

- **2026-07-05** — Document created. Populated from the B1/B2 implementation self-review and the Reviewer #2 assessment (both performed same day). 32 items across 9 sections.
