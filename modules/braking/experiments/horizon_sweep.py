"""Anticipation-horizon sweep for binary braking prediction (task A2.2).

For each look-ahead Delta produced by make_horizon_dataset.py, this trains the
braking model to predict "brake within the next Delta s" and scores it against
two non-ML floor baselines (rule = already braking; persist = last sample
braking). Because braking is a rare event, the headline metrics are positive-
class precision / recall / F1 and PR-AUC (average precision) - plain accuracy is
~96% by always predicting NoBrake and is reported only for completeness.

Outputs (git-ignored; regenerated in the experiments stage):
  modules/braking/experiments/results/horizon_sweep.csv
  modules/braking/experiments/results/horizon_sweep.png   (if matplotlib present)

Usage:
  python modules/braking/experiments/horizon_sweep.py            # full run (config epochs)
  python modules/braking/experiments/horizon_sweep.py --debug    # fast smoke test
  python modules/braking/experiments/horizon_sweep.py --epochs 30 --seed 42

torch is imported lazily inside training so the metric helpers stay importable
without torch installed.
"""
import os
import sys
import csv
import json
import argparse
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             average_precision_score)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, 'modules', 'braking', 'data', 'horizon')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
POS_LABEL = 1  # "Brake"


# --------------------------- data + metrics (torch-free) ---------------------------
def load_manifest():
    with open(os.path.join(DATA_DIR, 'horizons.json')) as f:
        return json.load(f)


def load_split(split):
    X = np.load(os.path.join(DATA_DIR, f'X_{split}.npy'))
    rule = np.load(os.path.join(DATA_DIR, f'rule_pred_{split}.npy'))
    persist = np.load(os.path.join(DATA_DIR, f'persist_pred_{split}.npy'))
    return X, rule, persist


def load_labels(split, hi):
    return np.load(os.path.join(DATA_DIR, f'y_h{hi}_{split}.npy'))


def score_hard(y_true, y_pred, y_prob=None):
    """Positive-class (Brake) metrics for a set of hard predictions."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[POS_LABEL], average=None, zero_division=0)
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_pos': float(p[0]),
        'recall_pos': float(r[0]),
        'f1_pos': float(f1[0]),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out['pr_auc'] = float(average_precision_score(y_true, y_prob))
    else:
        out['pr_auc'] = float('nan')
    return out


# --------------------------- model training (needs torch) ---------------------------
def _train_and_predict(X_tr, y_tr, X_eval, device, epochs, batch_size, lr, seed):
    """Train a binary braking classifier and return (preds, prob_pos) for X_eval."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from modules.braking.models.multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention
    from modules.train.train_braking import compute_class_weights

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MultitaskLSTMCNNAttention(input_dim=X_tr.shape[2])
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)  # binary head
    model = model.to(device)

    weights = compute_class_weights(y_tr, 2, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float(),
                      torch.from_numpy(y_tr.astype(np.int64))),
        batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = model(bx)
            criterion(logits, by).backward()
            optimizer.step()

    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        eval_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_eval).float()),
            batch_size=256, shuffle=False)
        for (bx,) in eval_loader:
            logits, _ = model(bx.to(device))
            prob = torch.softmax(logits, dim=1)[:, POS_LABEL]
            preds.append(logits.argmax(dim=1).cpu().numpy())
            probs.append(prob.cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


# --------------------------- sweep orchestration ---------------------------
def run_sweep(epochs, seed, debug, eval_split='test'):
    manifest = load_manifest()
    horizons = manifest['horizons_s']

    X_tr, rule_tr, persist_tr = load_split('train')
    X_ev, rule_ev, persist_ev = load_split(eval_split)
    if debug:
        X_tr, rule_tr, persist_tr = X_tr[:512], rule_tr[:512], persist_tr[:512]
        epochs = 1

    rows = []
    for hi, h in enumerate(horizons):
        y_tr = load_labels('train', hi)
        y_ev = load_labels(eval_split, hi)
        if debug:
            y_tr = y_tr[:512]

        # model
        try:
            preds, probs = _train_and_predict(
                X_tr, y_tr, X_ev, _device(), epochs, 32, 1e-3, seed)
            rows.append(_row(h, 'model', y_ev, preds, probs))
        except Exception as exc:  # pragma: no cover - surfaces torch/setup issues
            print(f"[{h}s] model training failed: {exc}")

        # floor baselines (hard predictions, horizon-independent inputs)
        rows.append(_row(h, 'rule', y_ev, rule_ev[:len(y_ev)]))
        rows.append(_row(h, 'persist', y_ev, persist_ev[:len(y_ev)]))

    _write_csv(rows, eval_split)
    _plot(rows, horizons, eval_split)
    return rows


def _row(h, method, y_true, y_pred, y_prob=None):
    m = score_hard(y_true, y_pred, y_prob)
    m.update({'horizon_s': h, 'method': method})
    print(f"  {method:8} @ {h}s : F1_pos={m['f1_pos']:.3f} "
          f"P={m['precision_pos']:.3f} R={m['recall_pos']:.3f} PR-AUC={m['pr_auc']:.3f}")
    return m


def _device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _write_csv(rows, split):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'horizon_sweep.csv')
    cols = ['horizon_s', 'method', 'accuracy', 'precision_pos', 'recall_pos', 'f1_pos', 'pr_auc']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"  wrote {path}  (eval split: {split})")


def _plot(rows, horizons, split):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("  matplotlib unavailable - skipping curve (csv still written)")
        return
    plt.figure(figsize=(6, 4))
    for method in ['model', 'rule', 'persist']:
        xs = [r['horizon_s'] for r in rows if r['method'] == method]
        ys = [r['f1_pos'] for r in rows if r['method'] == method]
        if xs:
            plt.plot(xs, ys, marker='o', label=method)
    plt.xlabel('Anticipation horizon Delta (s)')
    plt.ylabel('Positive-class F1 (Brake)')
    plt.title(f'Braking anticipation vs horizon ({split})')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'horizon_sweep.png')
    plt.savefig(path, dpi=150)
    print(f"  wrote {path}")


def _default_epochs():
    try:
        from shared.config import get_config
        return int(get_config().get('training.epochs.braking_multitask', 15) or 15)
    except Exception:
        return 15


def main():
    ap = argparse.ArgumentParser(description="Braking anticipation-horizon sweep (A2.2)")
    ap.add_argument('--epochs', type=int, default=None, help="training epochs per horizon")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval-split', default='test', choices=['val', 'test'])
    ap.add_argument('--debug', action='store_true', help="tiny subset + 1 epoch smoke test")
    args = ap.parse_args()
    epochs = args.epochs if args.epochs is not None else _default_epochs()
    print(f"Horizon sweep | epochs={epochs} seed={args.seed} debug={args.debug} "
          f"eval_split={args.eval_split}")
    run_sweep(epochs, args.seed, args.debug, args.eval_split)


if __name__ == "__main__":
    main()
