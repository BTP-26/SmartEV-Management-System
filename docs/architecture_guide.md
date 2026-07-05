# Architecture Guide

A from-first-principles tour of the EV Smart Management System. Assumes you are a strong
engineer who has never seen this repo. Read this before touching code.

> This is a **research** codebase supporting a paper under major revision. The guiding rule
> is: *every number in the paper must be reproducible from a clean checkout.* Correctness
> and honesty beat "nicer-looking" results.

---

## 1. What the system claims to do

Think of an electric vehicle that wants to brake intelligently and know how full its
battery is. Four pieces:

1. **Braking-intention prediction** — from recent motion sensors, predict whether the
   driver is about to brake *Light / Normal / Emergency*.
2. **State-of-Charge (SoC) estimation** — from recent battery voltage/current/temperature,
   estimate how full the pack is (0–100 %).
3. **Cognitive layer** — profile the driver's style (eco/normal/aggressive/conservative)
   and pick a regenerative-braking strategy.
4. **Unified pipeline + UI** — glue the above into one inference call and a dashboard.

An analogy: the two ML models are two specialists (a "driving-behaviour doctor" and a
"battery doctor"). They never operate on the same patient data; a **coordinator**
(`EnhancedEVPipeline`) asks each one a question and combines their answers into a single
recommendation. This matters because the paper must **not** claim the models are "jointly
trained" — they are co-designed, run side by side.

---

## 2. Repository structure

```
SmartEV-Management-System/
├─ run_complete_pipeline.py     # top-level orchestrator (check data → train → eval → demo)
├─ config/
│  ├─ default.yaml              # model dims, training knobs, paths, perf constants
│  └─ dataset_config.yaml       # real-vs-simulated switch + logical-name → file map
├─ shared/                      # cross-module "common ground" — announce before editing
│  ├─ config.py                 # Config singleton + get_config()
│  ├─ dataset_loader.py         # DatasetLoader: loads soc/braking .npy per dataset_config
│  ├─ train_utils.py            # set_seed, metrics, checkpoints, early stopping
│  ├─ enhanced_utils.py         # EnhancedEVPipeline: unified inference (braking+SoC+cog.)
│  └─ cognitive_manager.py      # driver profiling + regen strategy (rule-based)
├─ modules/
│  ├─ braking/   (Owner A — Siddharth)
│  │  ├─ data/preprocess_real_data.py      # UAH-DriveSet → windows + scaler.pkl
│  │  └─ models/{multitask_lstm_cnn_attention, genetic_algorithm_optimizer}.py
│  ├─ soc/       (Owner B — Dhananjay)   ← our area, see docs/soc_pipeline.md
│  │  ├─ data/preprocess_real_data.py      # Mendeley .mat → windows (SoC scaled [0,1])
│  │  ├─ evaluate_soc.py                    # single source of Table 6
│  │  └─ models/{lstm_cnn_attention_soc, physics_informed_soc, adaptive_ensemble,
│  │             multi_objective_ga_optimizer, coulomb_counting,
│  │             battery_rls_identification}.py
│  └─ train/{train_all_models, train_braking, train_soc}.py
├─ ui/app.py                    # Streamlit dashboard (synthetic inputs → pipeline)
├─ docs/                        # this knowledge base
└─ Real-world electric vehicle data driving and charging/   # Mendeley source (.mat)
```

**Ownership:** A = braking (Siddharth), **B = SoC (us, Dhananjay)**, C = unification/
cognitive (SaaD). Do not edit another owner's module. `shared/*` and
`run_complete_pipeline.py` are common ground — announce before editing.

---

## 3. The two datasets

| | Braking | SoC (ours) |
|---|---|---|
| Source | UAH-DriveSet v1 (phone sensors) | Mendeley "Real-world EV driving and charging" (Stanford) |
| Raw format | `.txt` sensor logs | `.mat` (HDF5) per drive/charge cycle |
| Window | 75 × 7 (acc/gyro/speed) | 50 × 3 (voltage, current, temperature) |
| Target | braking class + intensity | SoC (pack-level %, scaled to [0,1]) |

Both raw datasets are gitignored; the `.npy` windows are regenerated on demand. **Neither
`.npy` set is committed**, so reproducibility depends on the preprocessing scripts.

---

## 4. Data flow (end to end)

```
raw source ──► preprocess_real_data.py ──► *_real.npy ──► DatasetLoader ──► model ──► pipeline ──► UI
                    (windows, labels,          (config/                    (.pth)      (fuse)
                     scaling, split)         dataset_config.yaml)
```

`config/dataset_config.yaml` decides which files a module loads (real vs simulated). Every
consumer that needs data should go through `shared/dataset_loader.py` — as of B1, the SoC
ensemble/physics/GA models all do (they previously bypassed it with an orphaned file set).

---

## 5. Training flow

```
train_all_models.py ── subprocess ──► train_braking.py (baseline / multitask / GA)
                    ── subprocess ──► train_soc.py       (baseline / LSTM-CNN-Attention)
```

- The reusable, clean SoC trainer is `train_soc_model()` inside
  `modules/soc/models/lstm_cnn_attention_soc.py` (Adam + MSELoss + grad-clip + early stop +
  best-checkpoint restore). The ensemble and GA modules call it.
- `config/default.yaml` holds the intended knobs (epochs, batch size, lr, patience), but
  several trainers still hard-code them — a reproducibility gap being cleaned up per task.
- **Policy:** training is *experimental* work. Do not run full training as part of routine
  implementation; use lightweight smoke tests unless a real experiment is explicitly
  requested.

---

## 6. Evaluation flow

For SoC there is now **one** evaluator: `modules/soc/evaluate_soc.py`. It loads the real
test set, runs the model, inverse-transforms predictions and targets to % SoC through the
single `soc_scale.json` definition, and writes `table6.csv`. Any SoC number in the paper
should come from here. (Historically three different evaluators disagreed — see the B1 log
entry.)

---

## 7. Inference / unification

`shared/enhanced_utils.EnhancedEVPipeline`:
1. loads both `.pth` models (+ optional dynamic quantization),
2. validates inputs (shape/type/NaN),
3. runs braking → `(class, intensity)` and SoC → `soc ∈ [0,1]`,
4. fuses them into a `system_action` (`_process_results`),
5. optionally calls the cognitive layer for a driver-style-adjusted strategy.

The models do not share tensors — the pipeline is the only meeting point.

---

## 8. Configuration system

- `config/default.yaml` — architecture dims, training hyperparameters, paths, and
  `performance.regen_efficiency` (the source of the "65 % energy" figure).
- `config/dataset_config.yaml` — the real/simulated switch and the logical-name → path map.
- Both are read through `shared/config.py` (`get_config()` returns a process-wide singleton).

---

## 9. Where to go next

- SoC deep dive (our area, tasks B1–B6): **`docs/soc_pipeline.md`**.
- Chronological change journal: **`docs/implementation_log.md`**.
- The authoritative task plan: `REVISION_ROADMAP.md` (workstreams A/B/C, dependency matrix).
