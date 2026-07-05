# SoC Pipeline (Deep Dive)

Everything about the State-of-Charge estimation module (`modules/soc/`), which is our
ownership (roadmap tasks B1–B6). Written to teach: read top to bottom and you should be
able to explain and modify the pipeline without reopening every file.

---

## 1. The one-sentence goal

Given the last 50 timesteps of pack **voltage, current, temperature**, predict the battery
**State of Charge** (how full it is), and report the error in **% SoC** in a way any
reviewer can reproduce.

---

## 2. The data source

Mendeley "Real-world electric vehicle data driving and charging" (Stanford Energy Control
Lab). Real EV, not a lab bench. Each cycle is a `Raw.mat` (HDF5) under `Drive/` or
`Charge/` with pack-level channels sampled on their own timelines:

```
Curr [A]   Volt [V]   Temp [°C]   SoC [%]   + a Time* array per channel
```

Key physical facts (verified from the data and the authors' own MATLAB scripts):

| Quantity | Value | Source |
|---|---|---|
| SoC | 0–100 % (observed 14–97) | README: "SoC [%]" |
| Pack voltage | ~364–456 V (nominal ~424) | observed across cycles |
| Pack current | up to +835 A (drive), −239 A (charge) | observed |
| Pack capacity | **240 Ah** | `_code/fig_6_11_12/fig_6_11_12.m`, `Cap = 240` |
| Chemistry / cell config | **unknown** | not disclosed anywhere — do not fabricate |

Sign convention (important, unusual): **driving current is positive** (discharge),
**braking and charging are negative**. This is why the ECM math (§7) has R0 come out
positive with a negative regression coefficient.

---

## 3. Preprocessing — `modules/soc/data/preprocess_real_data.py`

```
load_ev_data(.mat)          # h5py → dict of channels
   │
synchronize_data            # downsample current 10×; interp V/T/SoC onto current's clock
   │                        # → features [volt, curr, temp], label soc (still %)
normalize_features          # z-score features PER TRIP (mean/std NOT persisted — see B5)
   │
create_sliding_windows      # window=50, step=25 (50% overlap); label = soc at window end
   │
scale_soc  (B1)             # soc% → [0,1] via fixed /100 (NOT a data-driven min/max fit)
   │
train_test_split (random)   # ← LEAKAGE: overlapping windows split randomly (B5 will fix)
   │
save *_real.npy + soc_scale.json
```

Why `/100` and not min-max? Because SoC is a *physical* percentage. A min-max fit would
make `0` and `1` mean different physical SoC levels in different subsets and leak
split-specific range into the scale. `soc_scale.json` records the transform so every
consumer inverts it identically.

**Gotcha fixed in B1:** `PROJECT_ROOT` was resolved one directory too shallow, so
`DATA_DIR` pointed nowhere and preprocessing always raised "No valid data found." Corrected
to three levels up.

---

## 4. The models (and how they relate)

```
                       lstm_cnn_attention_soc.py
                       ┌───────────────────────────┐
                       │ LSTMCNNAttentionSoC        │  ← the DEPLOYED model
                       │ train_soc_model()          │  ← reused everywhere
                       │ evaluate_soc_model()       │
                       └───────────┬───────────────┘
             imported by           │ imported by            imported by
        ┌──────────────────────────┼───────────────────────────────┐
        ▼                          ▼                                ▼
 adaptive_ensemble.py     multi_objective_ga_optimizer.py   physics_informed_soc.py
 LSTM+Transformer+Physics  weighted-objective HP GA (B6)     constraint-regularised NN (B2)
 + weight GA (B4)                                            + BatteryPhysicsParams
        ▲
        │  coulomb_counting.py  — classical CC baseline (B3/B4; currently disconnected)
        │  battery_rls_identification.py — offline pack-parameter ID (B2)
```

`LSTMCNNAttentionSoC` is the hub: it is the deployed model **and** a sub-model of the
ensemble, and its `train_soc_model`/`evaluate_soc_model` are reused by the GA and physics
files. Treat its interface as a shared contract — changing it ripples outward.

Model shape: `Conv1d → BatchNorm → LSTM → additive Attention → MLP → Sigmoid`. The final
`Sigmoid` is why targets must be in [0,1] (the whole point of B1).

---

## 5. Evaluation — `modules/soc/evaluate_soc.py` (B1)

The **single source of Table 6**. Loads real test data via `DatasetLoader`, runs the model
in batches, inverse-transforms predictions **and** targets to % SoC via `soc_scale.json`,
and reports RMSE/MAE/MAPE through the shared `calculate_regression_metrics`. It asserts
values stay in [0,1] before inverting, so a future scale regression fails loudly instead of
silently printing RMSE ≈ 73 again.

Current provisional result (see caveat): **RMSE 10.64 % SoC, MAE 9.50, MAPE 14.54 %**.

> ⚠️ **This number is provisional.** It is measured on the leaky random split. It is a
> sanity figure, not a paper result. The final Table 6 comes after B5's leak-free split.

---

## 6. Battery parameters — `physics_informed_soc.py` + `battery_rls_identification.py` (B2)

`BatteryPhysicsParams` used to hardcode an 18650 cell (100 Ah, 0.05 Ω, 3.7 V) — impossible
for a 400 V pack. Now:

- **Electrochemical/voltage fields** are loaded from `battery_params_identified.json`
  (generated by `battery_rls_identification.py`) with a code fallback to the last-known
  identified values.
- **SoH/thermal fields** stay literature-typical placeholders, explicitly flagged as
  *not identifiable from this dataset* (no aging cycles, no direct thermal test).
- Language changed from "Physics-Informed"/PINN → **"constraint-regularised"**, because
  only the electrochemical params are calibrated; it is not a physically-derived PINN.

---

## 7. RLS parameter identification — the math

We fit a **first-order Thevenin ECM**:

```
V(t) = OCV(SoC(t)) − R0·I(t) − Vc(t),     dVc/dt = I/C1 − Vc/(R1·C1)
```

Discretising (ZOH, α = exp(−Ts/(R1·C1))) and writing everything in terms of the OCV
residual `e[k] = V[k] − OCV(SoC[k])` gives a linear ARX(1,1):

```
e[k] = a1·e[k−1] + b0·I[k] + b1·I[k−1]
       a1 = α,   b0 = −R0,   b1 = α·R0 − R1·(1−α)
```

which is solved by recursive least squares and inverted:

```
R0 = −b0,   R1 = (a1·R0 − b1)/(1−a1),   τ = −Ts/ln(a1),   C1 = τ/R1
```

Two numerical-conditioning choices matter (both learned by debugging real cycles):
- **Scale current by 100 A** inside the recursion — `e` is O(1–10 V) but I is O(10–800 A);
  without scaling the covariance update is ill-conditioned and the pole `a1` goes unstable.
- **Forgetting factor ≈ 0.999** — this is a one-shot per-cycle batch fit, not online
  tracking, so minimal forgetting is the most stable.

**Result (13/15 cycles converge to physical fits):**

| Param | Mean ± std | Notes |
|---|---|---|
| R0 | 0.0246 ± 0.0148 Ω | well-identified; ≈20 V IR-drop at 800 A, sane |
| R1 | 0.0513 ± 0.0528 Ω | weakly identified (RC branch poorly excited) |
| C1 | 848 ± 1093 F | weakly identified (std > mean) — report with spread |

OCV(SoC) is a **proxy** fit from pooled low-current samples (no rest-test data exists) and
forced monotonic. The identified params inherit this approximation.

> ⚠️ **Only R0 is actually consumed** by the model (IR-drop in `BatteryPhysicsConstraints`).
> R1/C1 are identified and reported (appendix), but do not yet influence any prediction. It
> is a first-order ECM; do not claim 2nd-order.

---

## 8. Task status (B1–B6)

| Task | What | Status |
|---|---|---|
| **B1** | SoC scaling + single evaluator + one dataset convention | ✅ done (result provisional until B5) |
| **B2** | Real pack params + RLS ECM + rename PINN | ✅ done (R1/C1 not yet wired in) |
| B3 | Baseline ladder (CC → LinReg → MLP → LSTM → ensemble) | ⬜ next; also make `train_soc.py` canonical |
| B4 | CC-residual adaptation signal + real weight trajectory | ⬜ |
| B5 | Trip/session split + per-segment error (fixes provisional B1 number) | ⬜ |
| B6 | Rename to weighted-objective GA + weight-sensitivity sweep | ⬜ |

---

## 9. Open reproducibility debts to clear

1. The deployed `.pth` is not yet regenerable from a documented repo command (B3 will make
   `train_soc.py` the canonical trainer).
2. Table 6 is on a leaky split until B5.
3. Feature normalisation is per-trip and unpersisted (B5).
4. `coulomb_counting.py` still assumes 2.0 Ah vs the real 240 Ah (B3/B4).
