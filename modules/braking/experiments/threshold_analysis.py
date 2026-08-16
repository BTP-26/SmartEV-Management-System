"""Threshold / precision-recall calibration analysis for the Track A horizon
sweep (due-diligence follow-up to the Track A decision memo, remedy 1).

This does NOT change the dataset, labels, train/test split, model
architecture, or training procedure. It reuses horizon_sweep.py's existing
`_train_and_predict` unchanged (same seed, same hyperparameters, same model
class) and simply keeps the probability scores it already computes and
discards, instead of throwing them away after the single argmax@0.5 metric.

For each horizon, this:
  1. trains the same model horizon_sweep.py trains (identical call),
  2. sweeps every decision threshold along the ROC/PR curve via
     sklearn.metrics.precision_recall_curve,
  3. finds the F1-optimal threshold and reports P/R/F1 there,
  4. compares that optimum against the existing argmax@0.5 result and
     against the rule/persist floor baselines,
  5. writes CSV + PR-curve artifacts.

Because horizon_sweep.py does not checkpoint trained models, "keeping the
existing model's predictions" means re-running training with the same
seed/hyperparameters/architecture used in the original campaign, not
retraining a new one — no methodology change, no new campaign.

Outputs (git-ignored, regenerated here):
  modules/braking/experiments/results/threshold_sweep_h{horizon}.csv
  modules/braking/experiments/results/threshold_optimal_summary.csv
  modules/braking/experiments/results/pr_curves.png

Usage:
  python modules/braking/experiments/threshold_analysis.py               # multitask, matches ours+mt
  python modules/braking/experiments/threshold_analysis.py --debug       # fast smoke test
"""
import os
import sys
import csv
import argparse
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from modules.braking.experiments.horizon_sweep import (  # noqa: E402
    load_manifest, load_split, load_labels, load_intensity,
    _train_and_predict, _device, _default_epochs, RESULTS_DIR, POS_LABEL,
)


def sweep_thresholds(y_true, y_prob):
    """Full precision/recall/F1 curve across every threshold implied by the
    sorted probability scores (sklearn's precision_recall_curve already does
    this efficiently instead of a manual grid search)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=POS_LABEL)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1;
    # last precision/recall point has no corresponding finite threshold.
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where((precision + recall) > 0,
                  2 * precision * recall / np.maximum(precision + recall, 1e-12),
                  0.0)
    return thresholds, precision, recall, f1


def f1_at_fixed_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def rule_persist_f1(y_true, y_hard):
    return f1_at_fixed_threshold(y_true, y_hard.astype(float), 0.5)


def run(epochs, seed, debug, eval_split, multitask, lam, model_name):
    manifest = load_manifest()
    horizons = manifest['horizons_s']

    X_tr, rule_tr, persist_tr = load_split('train')
    X_ev, rule_ev, persist_ev = load_split(eval_split)
    if debug:
        X_tr = X_tr[:512]
        epochs = 1

    summary_rows = []
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pr_curve_data = {}

    for hi, h in enumerate(horizons):
        y_tr = load_labels('train', hi)[:len(X_tr)]
        y_ev = load_labels(eval_split, hi)
        yint_tr = load_intensity('train', hi)[:len(X_tr)] if multitask else None

        print(f"[{h}s] training (same config as horizon_sweep.py: seed={seed}, "
              f"epochs={epochs}, multitask={multitask}, model={model_name}) ...")
        model_trained, preds_argmax, probs, _ = _train_and_predict(
            X_tr, y_tr, X_ev, _device(), epochs, 32, 1e-3, seed,
            yint_tr=yint_tr, lam=lam, model_name=model_name, return_model=True)

        # Save the trained weights so this never needs to be re-run just to get
        # probability scores again - future threshold/PR analysis, error
        # auditing, or the eventual FASTSim real-intent wiring can load this
        # directly instead of retraining.
        import torch
        ckpt_dir = os.path.join(RESULTS_DIR, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'{model_name}{"_mt" if multitask else ""}_h{h}_seed{seed}.pt')
        torch.save({
            'state_dict': model_trained.state_dict(),
            'horizon_s': h, 'seed': seed, 'epochs': epochs, 'multitask': multitask,
            'model_name': model_name, 'input_dim': X_tr.shape[2],
        }, ckpt_path)
        print(f"  saved checkpoint: {ckpt_path}")

        # --- existing argmax@0.5 result (unchanged behavior, for comparison) ---
        p05, r05, f105 = f1_at_fixed_threshold(y_ev, probs, 0.5)

        # --- full threshold sweep ---
        thresholds, precision, recall, f1 = sweep_thresholds(y_ev, probs)
        if len(f1) == 0:
            print(f"[{h}s] WARNING: empty threshold sweep (degenerate probs); skipping")
            continue
        best_idx = int(np.argmax(f1))
        best_thr = float(thresholds[best_idx]) if best_idx < len(thresholds) else 1.0
        best_p, best_r, best_f1 = float(precision[best_idx]), float(recall[best_idx]), float(f1[best_idx])
        pr_auc = float(average_precision_score(y_ev, probs))

        # --- floor baselines, same horizon, same eval split ---
        rp, rr, rf1 = rule_persist_f1(y_ev, rule_ev[:len(y_ev)])
        pp, pr_, pf1 = rule_persist_f1(y_ev, persist_ev[:len(y_ev)])

        gap_closed = best_f1 >= pf1
        print(f"  argmax@0.5: F1={f105:.3f} P={p05:.3f} R={r05:.3f}")
        print(f"  F1-optimal @ thr={best_thr:.3f}: F1={best_f1:.3f} P={best_p:.3f} R={best_r:.3f}")
        print(f"  PR-AUC={pr_auc:.3f} | rule F1={rf1:.3f} | persist F1={pf1:.3f} | "
              f"optimal beats persist: {gap_closed}")

        # per-horizon full threshold curve, for anyone who wants the raw sweep
        curve_path = os.path.join(RESULTS_DIR, f'threshold_sweep_h{h}.csv')
        with open(curve_path, 'w', newline='') as f_out:
            w = csv.writer(f_out)
            w.writerow(['threshold', 'precision', 'recall', 'f1'])
            for t, pv, rv, fv in zip(thresholds, precision, recall, f1):
                w.writerow([float(t), float(pv), float(rv), float(fv)])
        print(f"  wrote {curve_path}")

        pr_curve_data[h] = (recall, precision)

        summary_rows.append({
            'horizon_s': h,
            'method': f"{model_name}+mt" if multitask else model_name,
            'f1_argmax_0.5': round(f105, 4),
            'precision_argmax_0.5': round(p05, 4),
            'recall_argmax_0.5': round(r05, 4),
            'threshold_optimal': round(best_thr, 4),
            'f1_optimal': round(best_f1, 4),
            'precision_optimal': round(best_p, 4),
            'recall_optimal': round(best_r, 4),
            'pr_auc': round(pr_auc, 4),
            'rule_f1': round(rf1, 4),
            'persist_f1': round(pf1, 4),
            'f1_improvement_from_calibration': round(best_f1 - f105, 4),
            'optimal_beats_persist': gap_closed,
            'optimal_beats_rule': best_f1 >= rf1,
        })

    _write_summary(summary_rows)
    _plot_pr_curves(pr_curve_data, eval_split)
    return summary_rows


def _write_summary(rows):
    path = os.path.join(RESULTS_DIR, 'threshold_optimal_summary.csv')
    cols = ['horizon_s', 'method', 'f1_argmax_0.5', 'precision_argmax_0.5', 'recall_argmax_0.5',
            'threshold_optimal', 'f1_optimal', 'precision_optimal', 'recall_optimal', 'pr_auc',
            'rule_f1', 'persist_f1', 'f1_improvement_from_calibration',
            'optimal_beats_persist', 'optimal_beats_rule']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {path}")


def _plot_pr_curves(pr_curve_data, split):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable - skipping PR-curve plot (CSVs still written)")
        return
    plt.figure(figsize=(6, 5))
    for h, (recall, precision) in sorted(pr_curve_data.items()):
        plt.plot(recall, precision, label=f'{h}s')
    plt.xlabel('Recall (positive class)')
    plt.ylabel('Precision (positive class)')
    plt.title(f'Track A precision-recall curves by horizon ({split})')
    plt.legend(title='Horizon')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'pr_curves.png')
    plt.savefig(path, dpi=150)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Threshold/PR-curve calibration analysis for Track A (due diligence, no methodology change)")
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval-split', default='test', choices=['val', 'test'])
    ap.add_argument('--multitask', action='store_true', default=True,
                     help="match the campaign's reported ours+mt model (default: on)")
    ap.add_argument('--no-multitask', dest='multitask', action='store_false')
    ap.add_argument('--lam', type=float, default=0.3)
    ap.add_argument('--model', default='ours', choices=['ours', 'yang'])
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()
    epochs = args.epochs if args.epochs is not None else _default_epochs()
    print(f"Threshold analysis | model={args.model} epochs={epochs} seed={args.seed} "
          f"multitask={args.multitask} eval_split={args.eval_split}")
    run(epochs, args.seed, args.debug, args.eval_split, args.multitask, args.lam, args.model)


if __name__ == "__main__":
    main()
