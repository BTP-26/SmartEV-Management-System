import os
import sys
import json
import numpy as np
import pandas as pd
import h5py
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# project root is 3 levels up from modules/soc/data/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(PROJECT_ROOT)

from modules.soc.soc_scale import SOC_MIN_PERCENT, SOC_MAX_PERCENT, scale_soc, inverse_scale_soc

DATA_DIR = os.path.join(PROJECT_ROOT, "Real-world electric vehicle data driving and charging")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
WINDOW_SIZE = 50
STEP_SIZE = 25  # larger step size for faster processing
CAPACITY_AH = 240.0  # Mendeley pack capacity (see fig_6_11_12.m: `Cap = 240`), not a single-cell value


def load_ev_data(folder_path: str) -> Dict[str, np.ndarray]:
    """load ev data from matlab .mat file using h5py"""
    mat_file = None
    for file in os.listdir(folder_path):
        if file.endswith('.mat'):
            mat_file = os.path.join(folder_path, file)
            break
    
    if not mat_file:
        raise FileNotFoundError(f"No .mat file found in {folder_path}")
    
    data = h5py.File(mat_file, 'r')
    raw_data = data['Raw']
    
    result = {}
    for key in ['Curr', 'Volt', 'Temp', 'SoC', 'TimeCurr', 'TimeVolt', 'TimeTemp', 'TimeSoC']:
        if key in raw_data:
            result[key] = np.array(raw_data[key]).flatten()
        else:
            print(f"warning: {key} not found in data")
            result[key] = np.array([])
    
    data.close()
    return result

def synchronize_data(ev_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """synchronize voltage, current, temperature, and soc data"""
    curr = ev_data['Curr']
    volt = ev_data['Volt']
    temp = ev_data['Temp']
    soc = ev_data['SoC']
    time_curr = ev_data['TimeCurr']
    time_volt = ev_data['TimeVolt']
    time_temp = ev_data['TimeTemp']
    time_soc = ev_data['TimeSoC']
    
    # Use current timestamps as reference (highest frequency)
    ref_time = time_curr
    
    # Downsample for faster processing (take every 10th sample)
    downsample_factor = 10
    indices = np.arange(0, len(ref_time), downsample_factor)
    ref_time = ref_time[indices]
    curr = curr[indices]
    
    # Interpolate other signals to downsampled timestamps
    from scipy import interpolate
    
    if len(volt) > 1 and len(time_volt) > 1:
        volt_interp = interpolate.interp1d(time_volt, volt, kind='linear', bounds_error=False, fill_value='extrapolate')
        volt_sync = volt_interp(ref_time)
    else:
        volt_sync = np.full_like(ref_time, np.nan)
    
    if len(temp) > 1 and len(time_temp) > 1:
        temp_interp = interpolate.interp1d(time_temp, temp, kind='linear', bounds_error=False, fill_value='extrapolate')
        temp_sync = temp_interp(ref_time)
    else:
        temp_sync = np.full_like(ref_time, np.nan)
    
    if len(soc) > 1 and len(time_soc) > 1:
        soc_interp = interpolate.interp1d(time_soc, soc, kind='linear', bounds_error=False, fill_value='extrapolate')
        soc_sync = soc_interp(ref_time)
    else:
        soc_sync = np.full_like(ref_time, np.nan)
    
    # Create feature matrix
    features = np.column_stack([volt_sync, curr, temp_sync])
    labels = soc_sync
    
    # Remove NaN values
    valid_mask = ~(np.isnan(features).any(axis=1) | np.isnan(labels))
    features = features[valid_mask]
    labels = labels[valid_mask]
    
    return features, labels, ref_time[valid_mask]

def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to zero mean and unit variance."""
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (features - mean) / std, mean, std

def create_sliding_windows(features: np.ndarray, labels: np.ndarray, 
                        window_size: int = WINDOW_SIZE, 
                        step_size: int = STEP_SIZE) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows from time series data."""
    n_samples = (len(features) - window_size) // step_size + 1
    
    X = np.zeros((n_samples, window_size, features.shape[1]))
    y = np.zeros(n_samples)
    
    for i in range(n_samples):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        X[i] = features[start_idx:end_idx]
        y[i] = labels[end_idx - 1]  # Use last value in window as label
    
    return X, y

def process_single_trip(trip_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Process a single trip folder."""
    try:
        ev_data = load_ev_data(trip_path)
        features, labels, _ = synchronize_data(ev_data)
        
        if len(features) < WINDOW_SIZE + 5:
            print(f"Skipping {trip_path}: insufficient data ({len(features)} samples)")
            return None, None
        
        # Normalize features
        features_norm, _, _ = normalize_features(features)
        
        # Create sliding windows
        X, y = create_sliding_windows(features_norm, labels)
        
        return X, y
    except Exception as e:
        print(f"Error processing {trip_path}: {e}")
        return None, None

def assign_sessions_to_splits(session_sizes: Dict[str, Tuple[str, int]], val_frac: float = 0.2,
                               test_frac: float = 0.2, seed: int = 42) -> Dict[str, str]:
    """B5: assign whole sessions (not windows) to train/val/test, so no session's windows
    can appear in more than one split (the leak roadmap EV-1/B5 exists to close).

    `session_sizes` maps session_id -> (folder_type, n_windows). Sessions are assigned
    independently *within* each folder_type ("Drive"/"Charge") via greedy largest-first
    bin-packing toward the target window-count share, so every split gets a mix of both
    regimes - without this, a split could end up with zero Charge windows, making
    per-segment "charge" reporting impossible for it. With only ~15 sessions total, exact
    60/20/20 proportions by window count are not achievable - this is an honest, documented
    tradeoff of session-level splitting on a small number of long sessions, not a bug.
    """
    rng = np.random.RandomState(seed)
    assignment: Dict[str, str] = {}

    folder_types = sorted(set(ftype for ftype, _ in session_sizes.values()))
    for folder_type in folder_types:
        group = [(sid, n) for sid, (ftype, n) in session_sizes.items() if ftype == folder_type]
        rng.shuffle(group)  # avoid alphabetical-order bias before greedy packing
        group.sort(key=lambda x: -x[1])  # largest sessions first

        total = sum(n for _, n in group)
        target_val = total * val_frac
        target_test = total * test_frac
        target_train = total - target_val - target_test
        cur = {'train': 0.0, 'val': 0.0, 'test': 0.0}

        for sid, n in group:
            deficits = {
                'train': target_train - cur['train'],
                'val': target_val - cur['val'],
                'test': target_test - cur['test'],
            }
            split = max(deficits, key=deficits.get)
            assignment[sid] = split
            cur[split] += n

    return assignment


def tag_segment_types(X: np.ndarray, session_ids: np.ndarray) -> np.ndarray:
    """B5: per-window segment label - 'charge' for windows from a Charge/ session, else
    'regen' if the window's mean current is negative (braking, per this dataset's sign
    convention: driving current is positive for accel/cruise, negative for braking - see
    docs/soc_pipeline.md), else 'drive'. X columns are [volt, curr, temp]
    (synchronize_data's column order), so current is X[:, :, 1]."""
    mean_current = X[:, :, 1].mean(axis=1)
    is_charge = np.array([sid.startswith("Charge/") for sid in session_ids])
    segment = np.where(is_charge, "charge", np.where(mean_current < 0, "regen", "drive"))
    return segment.astype("<U16")


def assert_no_session_leakage(session_id_train: np.ndarray, session_id_val: np.ndarray,
                               session_id_test: np.ndarray) -> None:
    """B5: the leak-free split's core guarantee - no session's windows may appear in more
    than one split. Raises if violated; called unconditionally so every future dataset
    regeneration is guarded, not just this one."""
    train_set, val_set, test_set = set(session_id_train), set(session_id_val), set(session_id_test)
    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    overlap_vt = val_set & test_set
    if overlap_tv or overlap_tt or overlap_vt:
        raise AssertionError(
            f"Session leakage detected! train&val={overlap_tv}, train&test={overlap_tt}, "
            f"val&test={overlap_vt}"
        )
    print(f"No session leakage: {len(train_set)} train / {len(val_set)} val / "
          f"{len(test_set)} test sessions, fully disjoint.")


def process_all_data():
    """Process all EV driving and charging data."""
    print("Processing real-world EV driving and charging data...")

    all_X, all_y, all_session_ids = [], [], []
    session_sizes: Dict[str, Tuple[str, int]] = {}

    # B5: sorted() for deterministic iteration order (was unsorted os.listdir() - matches
    # battery_rls_identification.py's list_cycle_folders() convention) and session_id
    # tracking per folder, so windows can later be assigned to splits by whole session
    # rather than randomly (the leak EV-1/B5 exists to close). process_single_trip() and
    # create_sliding_windows() are unchanged - session tracking is purely additive here.
    for folder_type in ("Drive", "Charge"):
        type_dir = os.path.join(DATA_DIR, folder_type)
        if not os.path.exists(type_dir):
            continue
        for folder in sorted(os.listdir(type_dir)):
            folder_path = os.path.join(type_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            session_id = f"{folder_type}/{folder}"
            print(f"Processing {folder_type.lower()} data: {folder}")
            X, y = process_single_trip(folder_path)
            if X is not None:
                all_X.append(X)
                all_y.append(y)
                all_session_ids.append(np.full(len(X), session_id, dtype="<U32"))
                session_sizes[session_id] = (folder_type, len(X))

    if not all_X:
        raise ValueError("No valid data found in any folder")

    # Concatenate all data
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    session_id_combined = np.concatenate(all_session_ids, axis=0)

    print(f"Total samples: {X_combined.shape[0]}")
    print(f"Feature shape: {X_combined.shape[1:]}")
    print(f"SoC range (raw %): [{y_combined.min():.3f}, {y_combined.max():.3f}]")

    # B1 fix: targets were previously saved on the raw 0-100 % scale while the model head
    # is a Sigmoid producing [0,1], which silently produced RMSE~73 / MAPE~98% downstream.
    # Scale once, here, so every consumer sees [0,1] and evaluate_soc.py inverse-transforms
    # back to % for reporting.
    y_combined = scale_soc(y_combined)
    print(f"SoC range (scaled): [{y_combined.min():.3f}, {y_combined.max():.3f}]")

    # B5: segment type (charge/drive/regen) is a per-window label, independent of which
    # split a session lands in - a Drive session can (and does) contain both drive and
    # regen windows.
    segment_type_combined = tag_segment_types(X_combined, session_id_combined)

    # B5: split by whole session (not window) - this is the actual leak fix. Random
    # train_test_split() on overlapping windows let near-duplicate windows land on both
    # sides of the split (EV-1); assigning whole sessions makes that structurally
    # impossible.
    split_assignment = assign_sessions_to_splits(session_sizes, val_frac=0.2, test_frac=0.2, seed=42)
    split_of_window = np.array([split_assignment[sid] for sid in session_id_combined])

    train_mask = split_of_window == 'train'
    val_mask = split_of_window == 'val'
    test_mask = split_of_window == 'test'

    X_train, y_train = X_combined[train_mask], y_combined[train_mask]
    X_val, y_val = X_combined[val_mask], y_combined[val_mask]
    X_test, y_test = X_combined[test_mask], y_combined[test_mask]
    session_id_train = session_id_combined[train_mask]
    session_id_val = session_id_combined[val_mask]
    session_id_test = session_id_combined[test_mask]
    segment_type_train = segment_type_combined[train_mask]
    segment_type_val = segment_type_combined[val_mask]
    segment_type_test = segment_type_combined[test_mask]

    assert_no_session_leakage(session_id_train, session_id_val, session_id_test)

    # Save datasets - same 6 files, same names/shapes/dtypes as before B5. This is what
    # keeps shared/dataset_loader.py::load_soc_dataset()'s contract completely unchanged;
    # only *which* windows populate them changes.
    np.save(os.path.join(OUTPUT_DIR, 'X_train_real.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_val_real.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'X_test_real.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_train_real.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_val_real.npy'), y_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_test_real.npy'), y_test)

    # B5: new, additive metadata files (session_id/segment_type), row-aligned with the six
    # files above. shared/dataset_loader.py::load_soc_dataset()'s own signature/return
    # value is untouched - these are loaded through a separate, new, opt-in method
    # (load_soc_split_metadata()) so every existing consumer needs zero changes.
    np.save(os.path.join(OUTPUT_DIR, 'session_id_train_real.npy'), session_id_train)
    np.save(os.path.join(OUTPUT_DIR, 'session_id_val_real.npy'), session_id_val)
    np.save(os.path.join(OUTPUT_DIR, 'session_id_test_real.npy'), session_id_test)
    np.save(os.path.join(OUTPUT_DIR, 'segment_type_train_real.npy'), segment_type_train)
    np.save(os.path.join(OUTPUT_DIR, 'segment_type_val_real.npy'), segment_type_val)
    np.save(os.path.join(OUTPUT_DIR, 'segment_type_test_real.npy'), segment_type_test)

    # Persist the scale so every downstream consumer (evaluate_soc.py, ensemble, physics
    # model) can invert predictions/targets back to % SoC through one shared definition.
    soc_scale_meta = {
        "soc_min_percent": SOC_MIN_PERCENT,
        "soc_max_percent": SOC_MAX_PERCENT,
        "scaled_range": "[0, 1]",
        "transform": "scaled = (raw_percent - soc_min_percent) / (soc_max_percent - soc_min_percent)",
        "inverse_transform": "raw_percent = scaled * (soc_max_percent - soc_min_percent) + soc_min_percent",
    }
    with open(os.path.join(OUTPUT_DIR, 'soc_scale.json'), 'w') as f:
        json.dump(soc_scale_meta, f, indent=2)

    print(f"Saved real-world EV datasets (B5 leak-free session split):")
    print(f"  Train: {X_train.shape[0]} samples, {len(set(session_id_train.tolist()))} sessions")
    print(f"  Val:   {X_val.shape[0]} samples, {len(set(session_id_val.tolist()))} sessions")
    print(f"  Test:  {X_test.shape[0]} samples, {len(set(session_id_test.tolist()))} sessions")
    for split_name, seg in [('Train', segment_type_train), ('Val', segment_type_val), ('Test', segment_type_test)]:
        unique, counts = np.unique(seg, return_counts=True)
        print(f"  {split_name} segment mix: {dict(zip(unique.tolist(), counts.tolist()))}")

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    try:
        process_all_data()
        print("Real-world EV data preprocessing completed successfully!")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
