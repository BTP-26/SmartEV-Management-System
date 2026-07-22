"""L-A2: compare our braking model against the reproduced base-paper architecture
(Yang et al., 2024) under identical conditions - same task, same trip-level split,
same seeds - with a paired significance test. Also lists the rule/persistence
floors. This is the valid same-dataset comparison (not a cross-dataset one).

Writes results/base_comparison.csv.

Usage:
  python modules/braking/experiments/base_comparison.py            # full (5 seeds)
  python modules/braking/experiments/base_comparison.py --debug    # fast smoke test
"""
import os
import sys
import csv
import argparse
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
import horizon_sweep as hs   # reuse loaders, metrics, and the shared trainer


def run(models, seeds, horizon_idx, epochs, debug):
    manifest = hs.load_manifest()
    horizon = manifest['horizons_s'][horizon_idx]
    print(f"Base comparison @ {horizon}s | models={models} | seeds={seeds}")

    X_tr, rule_tr, persist_tr = hs.load_split('train')
    X_te, rule_te, persist_te = hs.load_split('test')
    y_tr = hs.load_labels('train', horizon_idx)
    y_te = hs.load_labels('test', horizon_idx)
    if debug:
        X_tr, y_tr = X_tr[:512], y_tr[:512]
        epochs, seeds = 1, seeds[:1]

    rows = []
    f1_by_model = {name: {} for name in models}   # name -> {seed: f1_pos}, kept seed-aligned

    # non-ML floors (single value, no seeds)
    for name, pred in [('rule', rule_te), ('persist', persist_te)]:
        m = hs.score_hard(y_te, pred[:len(y_te)])
        rows.append({'model': name, 'seed': '-', **{k: m[k] for k in
                     ('f1_pos', 'precision_pos', 'recall_pos', 'pr_auc')}})

    # ML models across seeds (results kept keyed by seed so the paired test aligns)
    for name in models:
        for s in seeds:
            try:
                preds, probs, _ = hs._train_and_predict(
                    X_tr, y_tr, X_te, hs._device(), epochs, 32, 1e-3, s, model_name=name)
            except Exception as exc:  # pragma: no cover - surfaces torch/setup issues
                print(f"  {name} seed {s} failed: {exc}")
                continue
            m = hs.score_hard(y_te, preds, probs)
            f1_by_model[name][s] = m['f1_pos']
            rows.append({'model': name, 'seed': s, **{k: m[k] for k in
                         ('f1_pos', 'precision_pos', 'recall_pos', 'pr_auc')}})
        vals = list(f1_by_model[name].values())
        if vals:
            print(f"  {name:6}: F1_pos = {np.mean(vals):.3f} +/- {np.std(vals):.3f} (n={len(vals)})")

    # paired significance: ours vs base-paper, only on seeds where BOTH succeeded
    if 'ours' in f1_by_model and 'yang' in f1_by_model:
        common = [s for s in seeds if s in f1_by_model['ours'] and s in f1_by_model['yang']]
        if len(common) > 1:
            a = [f1_by_model['ours'][s] for s in common]
            b = [f1_by_model['yang'][s] for s in common]
            t, p = stats.ttest_rel(a, b)
            print(f"  paired t-test ours vs yang (F1_pos, n={len(common)}): t={t:.3f}, p={p:.4f}")
            rows.append({'model': 'ours_vs_yang_ttest', 'seed': '-',
                         'f1_pos': round(float(t), 4), 'pr_auc': round(float(p), 4),
                         'precision_pos': '', 'recall_pos': ''})

    os.makedirs(hs.RESULTS_DIR, exist_ok=True)
    path = os.path.join(hs.RESULTS_DIR, 'base_comparison.csv')
    cols = ['model', 'seed', 'f1_pos', 'precision_pos', 'recall_pos', 'pr_auc']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})
    print(f"  wrote {path}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Ours vs base-paper (Yang 2024) comparison")
    ap.add_argument('--models', nargs='+', default=['ours', 'yang'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 999])
    ap.add_argument('--horizon-idx', type=int, default=1, help="index into horizons_s (default 1.0s)")
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--data-dir', default=None, help="dataset dir (for window-size sweep)")
    ap.add_argument('--debug', action='store_true')
    a = ap.parse_args()
    if a.data_dir:
        hs.DATA_DIR = a.data_dir
    epochs = a.epochs if a.epochs is not None else hs._default_epochs()
    run(a.models, a.seeds, a.horizon_idx, epochs, a.debug)


if __name__ == "__main__":
    main()
