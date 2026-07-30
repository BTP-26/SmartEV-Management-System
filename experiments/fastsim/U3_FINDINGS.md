# U-3 — Braking → Regen Coupling Wrapper: Findings

**Date:** 2026-07-28 · **Scope:** U-3 only (coupling wrapper + the C-2 fix; no U-1 vehicle, no
braking/SoC model, no U-4, no paper). **Verdict: ready → U-4 can consume the coupling layer.**
Every output is **simulation**.

---

## 1. Implementation summary

- Runs a U-2 drive cycle through the U-1 Mendeley-like BEV under two strategies and returns a
  **standardized result** (SoC trajectory, battery power, derived current, regen energy, distance,
  and net Wh/km) so U-4 can consume it unchanged.
  - **baseline** — the raw (reactive) cycle.
  - **anticipatory** — knowing a brake is coming, the vehicle lifts off and **coasts down earlier**
    into predicted-braking windows, reaching the same target speed over more time.
- **Intent contract (for U-4):** `intent` is a per-timestep binary array (1 = brake predicted). Until
  the real braking model feeds it, an **oracle placeholder** is derived from the cycle. Swapping in
  the real model requires no interface change.
- Battery **current is derived** as power ÷ assumed pack voltage (424.22 V) — FASTSim-3 RES exposes
  no native voltage/current (U-0/U-1 convention).
- The trip-level controller keeps the anticipatory run **only when it lowers net Wh/km**, else falls
  back to baseline — so the reported saving is ≥ 0.

## 2. What changed in this pass (review fix **C-2**)

- **Problem:** `BRAKE_TRIGGER_W = U1_MAX_PROP_W` **aliased two unrelated physical quantities** — the
  braking-event significance threshold and the vehicle's propulsion limit. They have no physical
  relationship; coupling them made the trigger silently depend on the powertrain rating.
- **Fix:** introduced an **independent** `BRAKE_TRIGGER_W = 45000.0` W — the peak *mechanical braking
  power* above which a deceleration is worth anticipating. It is a property of **braking salience**
  (`P = m·a·v`; ~45 kW ≈ moderate braking, order of 1.5 m/s² at ~15 m/s for this ~2070 kg vehicle),
  **not** of propulsion, so it is deliberately decoupled from the vehicle and documented with tuning
  guidance (lower → anticipate gentler brakes; raise → only hard stops).
- **Behaviour preserved:** 45 kW vs the previous 44.7 kW is a < 1 % change, so the same braking
  events are anticipated (Aggressive still modifies 55 samples) — coupling continues to activate.

## 3. Key physical finding (why the metric is Wh/km, not regen)

The U-1 vehicle is **fully regen-capable**: peak braking power on these cycles is ~123 kW against a
239 kW motor / 379 kW pack limit, so FASTSim recovers **~100 % of braking as regen** and mechanical
friction braking is **0 Wh**. There is therefore **no friction loss for anticipation to recover** —
a "more regen" framing would be false here. The honest benefit is **lower net consumption**: coasting
in early draws less propulsion energy and avoids the regen round-trip (< 100 %) loss. Accordingly the
ablation metric is **net energy per km** (distance-normalized), and regen is reported for completeness
(it actually *drops* slightly under anticipation, which is expected).

## 4. Assumptions

1. **Coasting model:** an anticipated brake is represented by advancing/linearizing the speed drop
   over a lead window (default 4 s); the target speed of each brake is preserved.
2. **Oracle intent** stands in for the real braking model until U-4 wires it in.
3. **Derived current** (P / 424.22 V) — no native pack V/I in FASTSim-3.
4. **Regen cap never binds** on these cycles (verified), so results reflect coasting efficiency, not
   friction recovery.
5. Fully regen-capable vehicle → the coupling's energy lever is **round-trip efficiency**, which is
   modest and cycle-dependent.

## 5. Validation

Regenerated cycles (post C-1/C-2), baseline vs anticipatory through the U-1 BEV:

Numbers below are with the corrected **speed-dependent U-2 conditioning** (regen values logged in
`u3_validation.json`). Aggressive Wh/km rose vs the earlier constant-budget run (179 → 182) because
strong accelerations are no longer clipped — the intended de-biasing.

| Cycle | Net energy (baseline → anticipatory) | Saved | Intervened | NaN |
|---|---|---|---|---|
| Normal | 139.41 → 138.09 Wh/km | −1.32 (−0.95 %) | yes | no |
| Aggressive | 181.98 → 179.22 Wh/km | −2.76 (−1.52 %) | yes | no |
| Drowsy | 137.23 → 135.59 Wh/km | −1.64 (−1.20 %) | yes | no |

- **Coupling activates** and **anticipatory behaviour occurs** (speed traces modified; controller
  intervened on all three).
- **SoC physically reasonable:** monotone within a trip, small per-trip drop consistent with a
  ~100 kWh pack; no NaNs.
- **Battery power reasonable:** peak regen ~123 kW < motor/pack limits; discharge within pack rating.
- **Regen reasonable:** hundreds of Wh recovered per trip; drops slightly under anticipation, as
  expected when the vehicle carries less speed into each brake.
- Distances are effectively equal between strategies (≤ 0.1 km), so Wh/km is a fair comparison.

## 6. Limitations to carry into U-4

- **Modest, cycle-dependent** effect (~0.7–1.5 %), largest on motorway/aggressive driving. Frame as an
  honest efficiency finding, not "large energy savings".
- The benefit is **coasting round-trip efficiency**, contingent on the fully-regen-capable vehicle; a
  weaker/regen-limited vehicle would show a different (friction-recovery) mechanism.
- **Oracle intent** — final numbers should be reproduced once the **real braking model** feeds intent.
- **Counterfactual EV + flat road** (inherited from U-1/U-2): magnitudes are illustrative.
- Quasi-static FASTSim; single representative trip per style.

## 7. Files

**Modified (all in `experiments/fastsim/`):**
- `u3_coupling.py` — C-2: independent, documented `BRAKE_TRIGGER_W` (no longer aliased to propulsion);
  ablation metric is net Wh/km with a fall-back-to-baseline controller.

**Regenerated:** `u3_validation.json`. **Not modified:** U-0, U-1 vehicle, braking/SoC models, U-4.
