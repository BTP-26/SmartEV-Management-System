"""L-A3: anticipation lead-time distribution.

Replaces the old single-example figure with a distribution: for every test
window the model correctly predicts as a brake, how many seconds ahead of the
actual braking sample did it fire (the `ttb` computed by the dataset builder)?
Reports a histogram + summary stats across all correctly-anticipated events.

Writes results/lead_time.csv and results/lead_time.png.

Usage:
  python modules/braking/experiments/lead_time_analysis.py            # full
  python modules/braking/experiments/lead_time_analysis.py --debug    # smoke test
"""
import os
import sys
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import horizon_sweep as hs

DATA_DIR = hs.DATA_DIR
RESULTS_DIR = hs.RESULTS_DIR


def run(horizon_idx, seed, epochs, debug, model_name='ours'):
    horizon = hs.load_manifest()['horizons_s'][horizon_idx]
    X_tr = hs.load_split('train')[0]
    X_te = hs.load_split('test')[0]
    y_tr = hs.load_labels('train', horizon_idx)
    y_te = hs.load_labels('test', horizon_idx)
    ttb = np.load(os.path.join(DATA_DIR, 'ttb_test.npy'))   # seconds to first brake
    if debug:
        X_tr, y_tr = X_tr[:512], y_tr[:512]
        epochs = 1

    try:
        preds, _, _ = hs._train_and_predict(
            X_tr, y_tr, X_te, hs._device(), epochs, 32, 1e-3, seed, model_name=model_name)
    except Exception as exc:  # pragma: no cover - surfaces torch/setup issues
        print(f"model training failed: {exc}")
        return np.array([])
    if len(preds) != len(y_te):
        return np.array([])

    # correctly anticipated events: predicted brake, real brake ahead (finite ttb)
    tp = (preds == hs.POS_LABEL) & (y_te == 1) & np.isfinite(ttb)
    leads = ttb[tp]
    caught = float((preds[y_te == 1] == hs.POS_LABEL).mean()) if (y_te == 1).any() else float('nan')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'lead_time.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['horizon_s', 'n_events', 'median_lead_s', 'mean_lead_s', 'recall_events'])
        w.writerow([horizon, len(leads),
                    round(float(np.median(leads)), 3) if len(leads) else '',
                    round(float(np.mean(leads)), 3) if len(leads) else '',
                    round(caught, 3)])
    print(f"lead time @ {horizon}s: n={len(leads)} "
          f"median={np.median(leads):.2f}s mean={np.mean(leads):.2f}s recall={caught:.2f}"
          if len(leads) else "no correctly-anticipated events (try more epochs)")

    _plot(leads, horizon)
    return leads


def _plot(leads, horizon):
    if len(leads) == 0:
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("  matplotlib unavailable - skipping histogram (csv still written)")
        return
    plt.figure(figsize=(6, 4))
    plt.hist(leads, bins=15, edgecolor='black', alpha=0.8)
    plt.axvline(np.median(leads), color='red', linestyle='--',
                label=f'median {np.median(leads):.2f}s')
    plt.xlabel('Anticipation lead time (s before braking)')
    plt.ylabel('Correctly-anticipated events')
    plt.title(f'Braking anticipation lead-time distribution (horizon {horizon}s)')
    plt.legend(); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'lead_time.png')
    plt.savefig(path, dpi=150)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description="Anticipation lead-time distribution (L-A3)")
    ap.add_argument('--horizon-idx', type=int, default=3, help="index into horizons_s (default 3.0s)")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--debug', action='store_true')
    a = ap.parse_args()
    epochs = a.epochs if a.epochs is not None else hs._default_epochs()
    run(a.horizon_idx, a.seed, epochs, a.debug)


if __name__ == "__main__":
    main()
