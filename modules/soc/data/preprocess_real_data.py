import os
import numpy as np
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# project root is 3 levels up from modules/soc/data/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "Real-world electric vehicle data driving and charging")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
WINDOW_SIZE = 50
STEP_SIZE = 25  # larger step size for faster processing
CAPACITY_AH = 2.0

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

def process_all_data():
    """Process all EV driving and charging data."""
    print("Processing real-world EV driving and charging data...")
    
    all_X, all_y = [], []
    
    # Process driving data
    drive_dir = os.path.join(DATA_DIR, "Drive")
    if os.path.exists(drive_dir):
        for folder in os.listdir(drive_dir):
            folder_path = os.path.join(drive_dir, folder)
            if os.path.isdir(folder_path):
                print(f"Processing driving data: {folder}")
                X, y = process_single_trip(folder_path)
                if X is not None:
                    all_X.append(X)
                    all_y.append(y)
    
    # Process charging data
    charge_dir = os.path.join(DATA_DIR, "Charge")
    if os.path.exists(charge_dir):
        for folder in os.listdir(charge_dir):
            folder_path = os.path.join(charge_dir, folder)
            if os.path.isdir(folder_path):
                print(f"Processing charging data: {folder}")
                X, y = process_single_trip(folder_path)
                if X is not None:
                    all_X.append(X)
                    all_y.append(y)
    
    if not all_X:
        raise ValueError("No valid data found in any folder")
    
    # Concatenate all data
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    
    print(f"Total samples: {X_combined.shape[0]}")
    print(f"Feature shape: {X_combined.shape[1:]}")
    print(f"SoC range: [{y_combined.min():.3f}, {y_combined.max():.3f}]")
    
    # Split into train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_combined, y_combined, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
    )
    
    # Save datasets
    np.save(os.path.join(OUTPUT_DIR, 'X_train_real.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_val_real.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'X_test_real.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_train_real.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_val_real.npy'), y_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_test_real.npy'), y_test)
    
    print(f"Saved real-world EV datasets:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    try:
        process_all_data()
        print("Real-world EV data preprocessing completed successfully!")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
