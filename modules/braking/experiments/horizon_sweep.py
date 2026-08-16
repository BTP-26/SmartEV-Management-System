"""Anticipation-horizon sweep for binary braking prediction (tasks A2.2 / A3.2).

For each look-ahead Delta produced by make_horizon_dataset.py, this trains the
braking model to predict "brake within the next Delta s" and scores it against
two non-ML floor baselines (rule = already braking; persist = last sample
braking). Because braking is a rare event, the headline metrics are positive-
class precision / recall / F1 and PR-AUC (average precision) - plain accuracy is
~96% by always predicting NoBrake and is reported only for completeness.

With --multitask, the model also trains the regression head on the A3 intensity
target (peak deceleration magnitude), loss = CE + lam * MSE(intensity), and the
intensity MAE is reported. This gives the single-task vs multitask comparison.

Outputs (git-ignored; regenerated in the experiments stage):
  modules/braking/experiments/results/horizon_sweep.csv
  modules/braking/experiments/results/horizon_sweep.png   (if matplotlib present)

Usage:
  python modules/braking/experiments/horizon_sweep.py               # single-task
  python modules/braking/experiments/horizon_sweep.py --multitask   # + intensity head
  python modules/braking/experiments/horizon_sweep.py --debug       # fast smoke test

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
from sklearn.preprocessing import StandardScaler

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


def load_intensity(split, hi):
    return np.load(os.path.join(DATA_DIR, f'yint_h{hi}_{split}.npy'))


def load_all():
    """Concatenate train+val+test for leave-one-driver-out (ignores the split)."""
    def cat(name):
        return np.concatenate([np.load(os.path.join(DATA_DIR, f'{name}_{sp}.npy'))
                               for sp in ('train', 'val', 'test')])
    return cat('X'), cat('rule_pred'), cat('persist_pred'), cat('driver')


def load_pooled_raw():
    """Pooled UNSCALED windows + driver ids for leave-one-driver-out, so the
    scaler is fit per fold (a global scaler would leak the held-out driver)."""
    def cat(name):
        arrs = []
        for sp in ('train', 'val', 'test'):
            path = os.path.join(DATA_DIR, f'{name}_{sp}.npy')
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing {path}. Run `python modules/braking/data/make_horizon_dataset.py` "
                    f"to generate the Xraw_*/driver_* arrays before using --lodo.")
            arrs.append(np.load(path))
        return np.concatenate(arrs)
    return cat('Xraw'), cat('driver')


def _scale_fold(X_tr, X_te):
    """Fit a StandardScaler on the fold's training windows only, apply to both."""
    nf = X_tr.shape[2]
    scaler = StandardScaler().fit(X_tr.reshape(-1, nf))

    def apply(a):
        return scaler.transform(a.reshape(-1, nf)).reshape(a.shape).astype(np.float32)
    return apply(X_tr), apply(X_te)


def load_labels_all(hi):
    return np.concatenate([load_labels(sp, hi) for sp in ('train', 'val', 'test')])


def load_intensity_all(hi):
    return np.concatenate([load_intensity(sp, hi) for sp in ('train', 'val', 'test')])


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
def _build_model(model_name, input_dim, num_classes=2):
    """Return a braking model with a `num_classes` head. Both models expose the
    same (logits, intensity) forward interface so the harness is identical."""
    if model_name not in ('ours', 'yang'):   # fail fast on typos before loading torch
        raise ValueError(f"unknown model_name '{model_name}' (expected 'ours' or 'yang')")
    import torch.nn as nn
    if model_name == 'yang':   # base-paper reproduction (Yang et al., 2024)
        from modules.braking.models.yang_lstm_cnn_attention import YangLSTMCNNAttention
        return YangLSTMCNNAttention(input_dim=input_dim, num_classes=num_classes)
    from modules.braking.models.multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention
    model = MultitaskLSTMCNNAttention(input_dim=input_dim)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _train_and_predict(X_tr, y_tr, X_eval, device, epochs, batch_size, lr, seed,
                       yint_tr=None, lam=0.0, model_name='ours', return_model=False):
    """Train the braking model and predict on X_eval.

    Single-task by default (classification only). If yint_tr is given, trains the
    multitask model with loss = CE(class) + lam * MSE(intensity) and also returns
    intensity predictions. Returns (preds, prob_pos, int_preds); int_preds is None
    in single-task mode. model_name selects 'ours' or the 'yang' base architecture.

    return_model=True additionally returns the trained model as the first element
    (model, preds, prob_pos, int_preds), for callers that want to checkpoint it -
    default is False, so existing call sites (run_sweep, run_lodo) are unaffected.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from modules.train.train_braking import compute_class_weights

    torch.manual_seed(seed)
    np.random.seed(seed)
    multitask = yint_tr is not None

    model = _build_model(model_name, X_tr.shape[2]).to(device)

    cls_criterion = nn.CrossEntropyLoss(weight=compute_class_weights(y_tr, 2, device))
    reg_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    tensors = [torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr.astype(np.int64))]
    if multitask:
        tensors.append(torch.from_numpy(yint_tr.astype(np.float32)))
    loader = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits, intensity = model(batch[0])
            loss = cls_criterion(logits, batch[1])
            if multitask:
                loss = loss + lam * reg_criterion(intensity, batch[2])
            loss.backward()
            optimizer.step()

    model.eval()
    preds, probs, ints = [], [], []
    with torch.no_grad():
        eval_loader = DataLoader(TensorDataset(torch.from_numpy(X_eval).float()),
                                 batch_size=256, shuffle=False)
        for (bx,) in eval_loader:
            logits, intensity = model(bx.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, POS_LABEL].cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            ints.append(intensity.cpu().numpy())
    int_preds = np.concatenate(ints) if multitask else None
    if return_model:
        return model, np.concatenate(preds), np.concatenate(probs), int_preds
    return np.concatenate(preds), np.concatenate(probs), int_preds


# --------------------------- sweep orchestration ---------------------------
def run_sweep(epochs, seed, debug, eval_split='test', multitask=False, lam=0.3, model_name='ours'):
    manifest = load_manifest()
    horizons = manifest['horizons_s']

    X_tr, rule_tr, persist_tr = load_split('train')
    X_ev, rule_ev, persist_ev = load_split(eval_split)
    if debug:
        X_tr = X_tr[:512]
        epochs = 1

    if model_name == 'yang' and multitask:   # Yang has no real intensity head
        print("  note: --multitask ignored for the Yang base model (no intensity head)")
        multitask = False
    rows = []
    method = f"{model_name}+mt" if multitask else model_name
    for hi, h in enumerate(horizons):
        y_tr = load_labels('train', hi)[:len(X_tr)]
        y_ev = load_labels(eval_split, hi)
        yint_tr = load_intensity('train', hi)[:len(X_tr)] if multitask else None
        yint_ev = load_intensity(eval_split, hi) if multitask else None

        try:
            preds, probs, int_preds = _train_and_predict(
                X_tr, y_tr, X_ev, _device(), epochs, 32, 1e-3, seed,
                yint_tr=yint_tr, lam=lam, model_name=model_name)
            int_mae = float(np.mean(np.abs(int_preds - yint_ev))) if multitask else float('nan')
            rows.append(_row(h, method, y_ev, preds, probs, int_mae))
        except Exception as exc:  # pragma: no cover - surfaces torch/setup issues
            print(f"[{h}s] model training failed: {exc}")

        # floor baselines (hard predictions, horizon-independent inputs)
        rows.append(_row(h, 'rule', y_ev, rule_ev[:len(y_ev)]))
        rows.append(_row(h, 'persist', y_ev, persist_ev[:len(y_ev)]))

    _write_csv(rows, eval_split)
    _plot(rows, horizons, eval_split, method)
    return rows


def _row(h, method, y_true, y_pred, y_prob=None, int_mae=float('nan')):
    m = score_hard(y_true, y_pred, y_prob)
    m.update({'horizon_s': h, 'method': method, 'int_mae': int_mae})
    print(f"  {method:9} @ {h}s : F1_pos={m['f1_pos']:.3f} P={m['precision_pos']:.3f} "
          f"R={m['recall_pos']:.3f} PR-AUC={m['pr_auc']:.3f} int_MAE={int_mae:.3f}")
    return m


def _device():
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _write_csv(rows, split):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'horizon_sweep.csv')
    cols = ['horizon_s', 'method', 'accuracy', 'precision_pos', 'recall_pos',
            'f1_pos', 'pr_auc', 'int_mae']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"  wrote {path}  (eval split: {split})")


def _plot(rows, horizons, split, model_method):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("  matplotlib unavailable - skipping curve (csv still written)")
        return
    plt.figure(figsize=(6, 4))
    for method in [model_method, 'rule', 'persist']:
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


def run_lodo(epochs, seed, debug, multitask=False, lam=0.3, model_name='ours'):
    """Leave-one-driver-out: train on 5 drivers, test on the held-out one, per
    horizon, and report mean +/- std positive-class F1 across drivers (Reviewer 3.2)."""
    horizons = load_manifest()['horizons_s']
    X_all, driver_all = load_pooled_raw()   # unscaled; scaled per fold below
    drivers = sorted(int(d) for d in np.unique(driver_all))
    if model_name == 'yang' and multitask:   # Yang has no real intensity head
        print("  note: --multitask ignored for the Yang base model (no intensity head)")
        multitask = False
    method = f"{model_name}+mt" if multitask else model_name
    print(f"LODO over drivers {drivers} | method={method}")

    # labels don't depend on the fold, so load every horizon's arrays once
    labels = [load_labels_all(hi) for hi in range(len(horizons))]
    intens = [load_intensity_all(hi) for hi in range(len(horizons))] if multitask else None

    rows = []
    f1_by_h = {h: [] for h in horizons}
    for d in drivers:
        te = driver_all == d
        tr = ~te
        # scale once per held-out driver (the split is identical across horizons)
        X_tr_raw = X_all[tr][:512] if debug else X_all[tr]
        X_tr, X_te = _scale_fold(X_tr_raw, X_all[te])
        n_tr = len(X_tr)
        for hi, h in enumerate(horizons):
            y_tr = labels[hi][tr][:n_tr]
            y_ev = labels[hi][te]
            yint_tr = intens[hi][tr][:n_tr] if multitask else None
            try:
                preds, probs, _ = _train_and_predict(
                    X_tr, y_tr, X_te, _device(), 1 if debug else epochs,
                    32, 1e-3, seed, yint_tr=yint_tr, lam=lam, model_name=model_name)
                m = score_hard(y_ev, preds, probs)
                m.update({'horizon_s': h, 'method': method, 'driver': d})
                rows.append(m); f1_by_h[h].append(m['f1_pos'])
                print(f"  D{d} [{h}s]: F1_pos={m['f1_pos']:.3f} "
                      f"P={m['precision_pos']:.3f} R={m['recall_pos']:.3f}")
            except Exception as exc:  # pragma: no cover
                print(f"  D{d} [{h}s] failed: {exc}")

    summary = []
    for h in horizons:
        f1s = f1_by_h[h]
        if f1s:
            mean, std = float(np.mean(f1s)), float(np.std(f1s))
            summary.append({'horizon_s': h, 'method': method, 'f1_pos_mean': round(mean, 4),
                            'f1_pos_std': round(std, 4), 'n_folds': len(f1s)})
            print(f"  {h}s LODO {method}: F1_pos = {mean:.3f} +/- {std:.3f} (n={len(f1s)})")
    _write_lodo_csv(rows, summary)
    return rows, summary


def _write_lodo_csv(rows, summary):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fold_cols = ['horizon_s', 'method', 'driver', 'accuracy',
                 'precision_pos', 'recall_pos', 'f1_pos', 'pr_auc']
    with open(os.path.join(RESULTS_DIR, 'lodo_folds.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fold_cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fold_cols})
    sum_cols = ['horizon_s', 'method', 'f1_pos_mean', 'f1_pos_std', 'n_folds']
    with open(os.path.join(RESULTS_DIR, 'lodo_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sum_cols); w.writeheader()
        for r in summary:
            w.writerow(r)
    print(f"  wrote {RESULTS_DIR}/lodo_folds.csv and lodo_summary.csv")


def _default_epochs():
    try:
        from shared.config import get_config
        return int(get_config().get('training.epochs.braking_multitask', 15) or 15)
    except Exception:
        return 15


def main():
    ap = argparse.ArgumentParser(description="Braking anticipation-horizon sweep (A2.2 / A3.2)")
    ap.add_argument('--epochs', type=int, default=None, help="training epochs per horizon")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eval-split', default='test', choices=['val', 'test'])
    ap.add_argument('--multitask', action='store_true', help="train the intensity head too")
    ap.add_argument('--lam', type=float, default=0.3, help="weight on the intensity MSE loss")
    ap.add_argument('--lodo', action='store_true', help="leave-one-driver-out evaluation")
    ap.add_argument('--model', default='ours', choices=['ours', 'yang'], help="model to train")
    ap.add_argument('--debug', action='store_true', help="tiny subset + 1 epoch smoke test")
    args = ap.parse_args()
    epochs = args.epochs if args.epochs is not None else _default_epochs()
    print(f"Horizon sweep | model={args.model} epochs={epochs} seed={args.seed} "
          f"multitask={args.multitask} lam={args.lam} lodo={args.lodo} "
          f"debug={args.debug} eval_split={args.eval_split}")
    if args.lodo:
        run_lodo(epochs, args.seed, args.debug, args.multitask, args.lam, args.model)
    else:
        run_sweep(epochs, args.seed, args.debug, args.eval_split, args.multitask, args.lam, args.model)


if __name__ == "__main__":
    main()
