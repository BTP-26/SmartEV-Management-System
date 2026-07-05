"""Build the binary braking-anticipation dataset for the horizon sweep (task A2/A3).

Task definition (decided after data analysis of UAH-DriveSet, which contains no
graded/emergency braking): for a window whose input ends at time t, predict
whether a braking event occurs within the next Delta seconds ("brake within the
next Delta s"). A braking event is a sample with longitudinal deceleration
< -1.0 m/s^2 (GPS-derived; orientation-free). The input window strictly precedes
the look-ahead, so the task is leak-free (Reviewer 3.1).

A3 adds an auxiliary regression target per window: the normalised peak
deceleration magnitude in the look-ahead window (how hard braking will be) -
the continuous quantity the binary label is a threshold of. This replaces the
all-zero intensity target so the multitask regression head has something to learn.

Per window we also emit two non-ML floor baselines (Reviewer 3.1 / 4.2):
  * rule    : 1 if the vehicle is already braking anywhere in the INPUT window.
  * persist : 1 if the last input sample is a braking sample ("assume no change").

Outputs (regenerable; .npy git-ignored) -> modules/braking/data/horizon/:
  X_{split}.npy              scaled input windows (shared across horizons)
  y_h{idx}_{split}.npy       binary label per horizon (brake within Delta s)
  yint_h{idx}_{split}.npy    intensity target per horizon (normalised peak decel)
  rule_pred_{split}.npy      threshold-rule floor prediction
  persist_pred_{split}.npy   persistence floor prediction
  scaler.pkl                 StandardScaler fit on train
  horizons.json              manifest (horizons, positive rates, distributions)

Additive: does NOT touch the Stage-1 y_class_*_real.npy files.
"""
import os
import sys
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from modules.braking.data.preprocess_real_data import (  # noqa: E402
    DATA_DIR, WINDOW_SIZE, STEP_SIZE,
    load_accelerometer_data, load_gps_data,
    synchronize_sensor_data, compute_deceleration,
)

# --- A2/A3 configuration (explicit; can move to config/default.yaml later) ---
HORIZONS_S = [0.5, 1.0, 2.0, 3.0]   # look-ahead lead times Delta (seconds)
BRAKE_THRESHOLD = -1.0              # deceleration (m/s^2) that counts as braking
MAX_DECEL_MS2 = 3.0                 # scale for the intensity target (m/s^2 -> [0,1])
RANDOM_STATE = 42
CLASS_NAMES = {0: "NoBrake", 1: "Brake"}
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'horizon')


def _binary_distribution(y):
    y = np.asarray(y).astype(int)
    counts = np.bincount(y, minlength=2)
    return {CLASS_NAMES[0]: int(counts[0]), CLASS_NAMES[1]: int(counts[1])}


def _peak_intensity(decel_segment):
    """Normalised peak deceleration magnitude in a look-ahead segment -> [0,1]."""
    if len(decel_segment) == 0:
        return 0.0
    magnitude = max(0.0, -float(decel_segment.min()))   # strongest deceleration
    return min(magnitude / MAX_DECEL_MS2, 1.0)


def _load_trip(trip_path):
    """Return (features[N,7], decel[N], braking_bool[N], median_dt) or None."""
    time_acc, acc_data, gyro_data = load_accelerometer_data(trip_path)
    time_gps, gps_speed = load_gps_data(trip_path)
    if time_acc is None or len(time_acc) < WINDOW_SIZE + 10:
        return None
    features, time_clean = synchronize_sensor_data(time_acc, acc_data, gyro_data, time_gps, gps_speed)
    if len(features) < WINDOW_SIZE + 10:
        return None
    decel = compute_deceleration(gps_speed, time_gps, time_clean)   # m/s^2 per sample
    braking = (decel < BRAKE_THRESHOLD).astype(np.int64)
    dt = float(np.median(np.diff(time_clean)))
    if not np.isfinite(dt) or dt <= 0:
        return None
    return features, decel, braking, dt


def _windows_for_trip(features, decel, braking, horizon_samps):
    """Emit (window, [label], [intensity], rule, persist) per window.

    A window is kept only if its longest look-ahead fits inside the trip, so the
    window set is identical across horizons.
    """
    n = len(features)
    max_h = max(horizon_samps)
    out_X, out_y, out_int, out_rule, out_persist = [], [], [], [], []
    for s in range(0, n - WINDOW_SIZE + 1, STEP_SIZE):
        e = s + WINDOW_SIZE - 1
        if e + max_h >= n:
            break
        y_per_h = [int(braking[e + 1:e + 1 + h].any()) for h in horizon_samps]
        int_per_h = [_peak_intensity(decel[e + 1:e + 1 + h]) for h in horizon_samps]
        out_X.append(features[s:s + WINDOW_SIZE])
        out_y.append(y_per_h)
        out_int.append(int_per_h)
        out_rule.append(int(braking[s:e + 1].any()))   # already braking in input window
        out_persist.append(int(braking[e]))            # last input sample braking
    return out_X, out_y, out_int, out_rule, out_persist


def build():
    print("Building binary braking-anticipation dataset (A2/A3)...")
    print(f"  look-ahead horizons (s): {HORIZONS_S}")

    all_X, all_y, all_int, all_rule, all_persist, dts = [], [], [], [], [], []
    for driver in sorted(os.listdir(DATA_DIR)):
        driver_path = os.path.join(DATA_DIR, driver)
        if not os.path.isdir(driver_path) or not driver.startswith('D'):
            continue
        for trip in sorted(os.listdir(driver_path)):
            trip_path = os.path.join(driver_path, trip)
            if not os.path.isdir(trip_path):
                continue
            loaded = _load_trip(trip_path)
            if loaded is None:
                continue
            features, decel, braking, dt = loaded
            horizon_samps = [max(1, int(round(h / dt))) for h in HORIZONS_S]
            Xw, yw, iw, rw, pw = _windows_for_trip(features, decel, braking, horizon_samps)
            if Xw:
                all_X.extend(Xw); all_y.extend(yw); all_int.extend(iw)
                all_rule.extend(rw); all_persist.extend(pw)
                dts.append(dt)

    if not all_X:
        raise ValueError("No windows produced - check dataset path / horizon settings.")

    X = np.asarray(all_X, dtype=np.float32)
    y = np.asarray(all_y, dtype=np.int64)              # (Nw, n_horizons)
    yint = np.asarray(all_int, dtype=np.float32)       # (Nw, n_horizons)
    rule = np.asarray(all_rule, dtype=np.int64)
    persist = np.asarray(all_persist, dtype=np.int64)
    print(f"  total windows: {len(X)}  |  median trip dt: {np.median(dts):.4f}s")

    # Stratify on the longest-horizon label (most positives) so both classes
    # appear in every split. A5 will replace this with trip-level splitting.
    strat_ref = y[:, -1]
    stratify = strat_ref if np.bincount(strat_ref, minlength=2).min() >= 2 else None
    idx = np.arange(len(X))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.30, random_state=RANDOM_STATE, stratify=stratify)
    strat_tmp = strat_ref[idx_tmp] if stratify is not None else None
    if strat_tmp is not None and np.bincount(strat_tmp, minlength=2).min() < 2:
        strat_tmp = None
    idx_val, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=strat_tmp)
    splits = {'train': idx_tr, 'val': idx_val, 'test': idx_te}

    scaler = StandardScaler()
    n_feat = X.shape[2]
    scaler.fit(X[idx_tr].reshape(-1, n_feat))

    def scale(arr):
        ns, nt, nf = arr.shape
        return scaler.transform(arr.reshape(-1, nf)).reshape(ns, nt, nf).astype(np.float32)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = {
        'task': 'binary_braking_anticipation',
        'positive_definition': f'braking event (decel < {BRAKE_THRESHOLD} m/s^2) within next Delta s',
        'intensity_definition': f'peak deceleration magnitude in look-ahead, normalised by {MAX_DECEL_MS2} m/s^2',
        'horizons_s': HORIZONS_S,
        'class_names': CLASS_NAMES,
        'window_size': WINDOW_SIZE,
        'step_size': STEP_SIZE,
        'total_windows': int(len(X)),
        'label_distribution': {},
    }
    for split, sidx in splits.items():
        np.save(os.path.join(OUTPUT_DIR, f'X_{split}.npy'), scale(X[sidx]))
        np.save(os.path.join(OUTPUT_DIR, f'rule_pred_{split}.npy'), rule[sidx])
        np.save(os.path.join(OUTPUT_DIR, f'persist_pred_{split}.npy'), persist[sidx])
        manifest['label_distribution'][split] = {}
        for hi, h in enumerate(HORIZONS_S):
            yh = y[sidx, hi]
            ih = yint[sidx, hi]
            np.save(os.path.join(OUTPUT_DIR, f'y_h{hi}_{split}.npy'), yh)
            np.save(os.path.join(OUTPUT_DIR, f'yint_h{hi}_{split}.npy'), ih)
            dist = _binary_distribution(yh)
            dist['positive_rate'] = round(float(yh.mean()), 4)
            dist['intensity_mean'] = round(float(ih.mean()), 4)
            manifest['label_distribution'][split][f'{h}s'] = dist

    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUT_DIR, 'horizons.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print("  Brake rate / mean intensity by look-ahead (train):")
    for h in HORIZONS_S:
        d = manifest['label_distribution']['train'][f'{h}s']
        print(f"    {h}s: rate={d['positive_rate']:.3f}  intensity_mean={d['intensity_mean']:.3f}")
    print(f"  saved to: {OUTPUT_DIR}")
    print("Done.")
    return True


if __name__ == "__main__":
    build()
