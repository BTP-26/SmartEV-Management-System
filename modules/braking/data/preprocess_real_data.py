import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, "UAH-DRIVESET-v1")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__))
WINDOW_SIZE = 75
STEP_SIZE = 25

class DrivingBehavior:
    NORMAL = 0
    AGGRESSIVE = 1
    DROWSY = 2

def parse_trip_name(trip_folder: str) -> Dict[str, str]:
    """get trip info from folder name"""
    parts = os.path.basename(trip_folder).split('-')
    if len(parts) >= 5:
        return {
            'datetime': parts[0],
            'distance': parts[1],
            'driver': parts[2],
            'behavior': parts[3],
            'road': parts[4]
        }
    return {}

def load_accelerometer_data(trip_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    acc_file = os.path.join(trip_path, 'RAW_ACCELEROMETERS.txt')
    if not os.path.exists(acc_file):
        return None, None, None
    
    data = []
    with open(acc_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) >= 10:
                # format: time, status, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, ?, ?, ?
                time = float(values[0])
                acc_x = float(values[2])
                acc_y = float(values[3])
                acc_z = float(values[4])
                gyro_x = float(values[5])
                gyro_y = float(values[6])
                gyro_z = float(values[7])
                data.append([time, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
    
    if not data:
        return None, None, None
    
    data = np.array(data)
    return data[:, 0], data[:, 1:4], data[:, 4:7]  # time, accelerometer, gyroscope

def load_gps_data(trip_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """load gps data for speed information"""
    gps_file = os.path.join(trip_path, 'RAW_GPS.txt')
    if not os.path.exists(gps_file):
        return None, None
    
    data = []
    with open(gps_file, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) >= 11:
                # format: time, speed, latitude, longitude, altitude, ?, ?, ?, ?, ?, ?, ?
                time = float(values[0])
                speed = float(values[1])
                data.append([time, speed])
    
    if not data:
        return None, None
    
    data = np.array(data)
    return data[:, 0], data[:, 1]  # time, speed

def create_braking_labels(gps_speed: np.ndarray, time_gps: np.ndarray, 
                          time_acc: np.ndarray) -> np.ndarray:
    """create braking intention labels based on gps speed changes"""
    if len(gps_speed) < 2:
        return np.zeros(len(time_acc))
    
    # interpolate gps speed to accelerometer timestamps
    try:
        # Ensure time arrays are sorted and unique
        sort_idx = np.argsort(time_gps)
        time_gps_sorted = time_gps[sort_idx]
        speed_sorted = gps_speed[sort_idx]
        
        # Remove duplicates
        unique_mask = np.diff(time_gps_sorted) > 1e-6
        time_gps_unique = np.concatenate([time_gps_sorted[:1], time_gps_sorted[1:][unique_mask]])
        speed_unique = np.concatenate([speed_sorted[:1], speed_sorted[1:][unique_mask]])
        
        if len(time_gps_unique) >= 2:
            speed_interp = interpolate.interp1d(time_gps_unique, speed_unique, kind='linear', 
                                              bounds_error=False, fill_value='extrapolate')
            speed_sync = speed_interp(time_acc)
        else:
            speed_sync = np.full_like(time_acc, 0.0)
    except:
        speed_sync = np.full_like(time_acc, 0.0)
    
    # calculate acceleration from speed change
    labels = np.zeros(len(time_acc))
    for i in range(1, len(speed_sync)):
        dt = time_acc[i] - time_acc[i-1]
        if dt > 0:
            dv = speed_sync[i] - speed_sync[i-1]
            acceleration = dv / dt
            
            # braking detected if significant negative acceleration
            if acceleration < -1.0:  # Threshold for braking detection
                labels[i] = 1
    
    return labels

def create_behavior_labels(behavior: str, data_length: int) -> np.ndarray:
    """create behavior classification labels"""
    behavior_map = {
        'NORMAL': 0,
        'AGGRESSIVE': 1,
        'DROWSY': 2
    }
    return np.full(data_length, behavior_map.get(behavior.upper(), 0))

def synchronize_sensor_data(time_acc: np.ndarray, acc_data: np.ndarray, gyro_data: np.ndarray,
                         time_gps: np.ndarray, gps_speed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """synchronize all sensor data to accelerometer timestamps"""
    
    # use accelerometer timestamps as reference (highest frequency)
    ref_time = time_acc
    
    # synchronize gps data
    if len(gps_speed) > 1 and len(time_gps) > 1:
        try:
            # make sure time arrays are sorted and unique
            sort_idx = np.argsort(time_gps)
            time_gps_sorted = time_gps[sort_idx]
            gps_speed_sorted = gps_speed[sort_idx]
            
            # get rid of duplicates
            unique_mask = np.diff(time_gps_sorted) > 1e-6
            time_gps_unique = np.concatenate([time_gps_sorted[:1], time_gps_sorted[1:][unique_mask]])
            gps_speed_unique = np.concatenate([gps_speed_sorted[:1], gps_speed_sorted[1:][unique_mask]])
            
            if len(time_gps_unique) >= 2:
                gps_interp = interpolate.interp1d(time_gps_unique, gps_speed_unique, kind='linear', 
                                                  bounds_error=False, fill_value='extrapolate')
                gps_sync = gps_interp(ref_time)
            else:
                gps_sync = np.full_like(ref_time, 0.0)
        except:
            gps_sync = np.full_like(ref_time, 0.0)
    else:
        gps_sync = np.full_like(ref_time, 0.0)
    
    # combine features: accelerometer + gyroscope + speed
    features = np.column_stack([acc_data, gyro_data, gps_sync])
    
    # remove rows with nan values
    valid_mask = ~(np.isnan(features).any(axis=1))
    features = features[valid_mask]
    time_clean = ref_time[valid_mask]
    
    return features, time_clean

def create_sliding_windows(features: np.ndarray, labels: np.ndarray, 
                          intention_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """create sliding windows for time series data"""
    X, y_class, y_intention = [], [], []
    
    for i in range(0, len(features) - WINDOW_SIZE + 1, STEP_SIZE):
        window = features[i:i + WINDOW_SIZE]
        class_label = labels[i + WINDOW_SIZE - 1]  # label at end of window
        intention_label = intention_labels[i + WINDOW_SIZE - 1]
        
        X.append(window)
        y_class.append(class_label)
        y_intention.append(intention_label)
    
    return np.array(X), np.array(y_class), np.array(y_intention)

def process_single_trip(trip_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """process a single trip folder with comprehensive sensor data"""
    try:
        # parse trip information
        trip_info = parse_trip_name(trip_path)
        behavior = trip_info.get('behavior', 'NORMAL')
        
        # load all sensor data
        time_acc, acc_data, gyro_data = load_accelerometer_data(trip_path)
        time_gps, gps_speed = load_gps_data(trip_path)
        
        if time_acc is None or len(time_acc) < WINDOW_SIZE + 10:
            print(f"skipping {trip_path}: not enough accelerometer data ({len(time_acc) if time_acc is not None else 0} samples)")
            return None, None, None
        
        # synchronize sensor data
        features, time_clean = synchronize_sensor_data(time_acc, acc_data, gyro_data, time_gps, gps_speed)
        
        if len(features) < WINDOW_SIZE + 10:
            print(f"skipping {trip_path}: not enough synchronized data ({len(features)} samples)")
            return None, None, None
        
        # create labels
        braking_labels = create_braking_labels(gps_speed, time_gps, time_clean)
        behavior_labels = create_behavior_labels(behavior, len(features))
        
        # create sliding windows
        X, y_class, y_intention = create_sliding_windows(features, braking_labels, behavior_labels)
        
        if len(X) == 0:
            print(f"skipping {trip_path}: no valid windows created")
            return None, None, None
        
        print(f"  processed {len(X)} windows from {os.path.basename(trip_path)}")
        return X, y_class, y_intention
        
    except Exception as e:
        print(f"error processing {trip_path}: {e}")
        return None, None, None

def process_all_data():
    """process all uah-driveset v1 data comprehensively"""
    print("processing uah-driveset v1 data comprehensively...")
    
    all_X, all_y_class, all_y_intention = [], [], []
    driver_stats = {}
    
    # process all driver folders (only d1-d6 are valid)
    for driver_folder in sorted(os.listdir(DATA_DIR)):
        driver_path = os.path.join(DATA_DIR, driver_folder)
        if not os.path.isdir(driver_path) or driver_folder.startswith('.') or not driver_folder.startswith('D'):
            continue
        
        print(f"\nprocessing driver: {driver_folder}")
        driver_X, driver_y_class, driver_y_intention = [], [], []
        
        # process all trips for this driver
        for trip_folder in sorted(os.listdir(driver_path)):
            trip_path = os.path.join(driver_path, trip_folder)
            if not os.path.isdir(trip_path):
                continue
            
            print(f"  processing trip: {trip_folder}")
            X, y_class, y_intention = process_single_trip(trip_path)
            
            if X is not None:
                driver_X.append(X)
                driver_y_class.append(y_class)
                driver_y_intention.append(y_intention)
        
        if driver_X:
            # combine all trips for this driver
            driver_X_combined = np.concatenate(driver_X, axis=0)
            driver_y_class_combined = np.concatenate(driver_y_class, axis=0)
            driver_y_intention_combined = np.concatenate(driver_y_intention, axis=0)
            
            all_X.append(driver_X_combined)
            all_y_class.append(driver_y_class_combined)
            all_y_intention.append(driver_y_intention_combined)
            
            driver_stats[driver_folder] = {
                'windows': len(driver_X_combined),
                'braking_ratio': np.mean(driver_y_class_combined),
                'behaviors': np.unique(driver_y_intention_combined, return_counts=True)
            }
    
    if not all_X:
        raise ValueError("No valid data found in any folder")
    
    # combine all data
    X_combined = np.concatenate(all_X, axis=0)
    y_class_combined = np.concatenate(all_y_class, axis=0)
    y_intention_combined = np.concatenate(all_y_intention, axis=0)
    
    print(f"\ndataset summary:")
    print(f"total windows: {len(X_combined)}")
    print(f"features per window: {X_combined.shape[1]} x {X_combined.shape[2]}")
    print(f"braking events: {np.sum(y_class_combined)} ({np.mean(y_class_combined)*100:.1f}%)")
    print(f"behavior distribution: {np.unique(y_intention_combined, return_counts=True)}")
    
    # split into train/val/test (70/15/15)
    X_train, X_temp, y_class_train, y_class_temp, y_intention_train, y_intention_temp = train_test_split(
        X_combined, y_class_combined, y_intention_combined, test_size=0.3, random_state=42, stratify=y_intention_combined
    )
    
    X_val, X_test, y_class_val, y_class_test, y_intention_val, y_intention_test = train_test_split(
        X_temp, y_class_temp, y_intention_temp, test_size=0.5, random_state=42, stratify=y_intention_temp
    )
    
    # normalize features
    print("normalizing features...")
    scaler = StandardScaler()
    
    # fit on training data and transform all sets
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(n_samples, n_timesteps, n_features)
    
    n_samples = X_val.shape[0]
    X_val_reshaped = X_val.reshape(-1, n_features)
    X_val_scaled = scaler.transform(X_val_reshaped).reshape(n_samples, n_timesteps, n_features)
    
    n_samples = X_test.shape[0]
    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(n_samples, n_timesteps, n_features)
    
    # save datasets
    print("saving datasets...")
    np.save(os.path.join(OUTPUT_DIR, 'X_train_real.npy'), X_train_scaled)
    np.save(os.path.join(OUTPUT_DIR, 'y_class_train_real.npy'), y_class_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_intention_train_real.npy'), y_intention_train)
    
    np.save(os.path.join(OUTPUT_DIR, 'X_val_real.npy'), X_val_scaled)
    np.save(os.path.join(OUTPUT_DIR, 'y_class_val_real.npy'), y_class_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_intention_val_real.npy'), y_intention_val)
    
    np.save(os.path.join(OUTPUT_DIR, 'X_test_real.npy'), X_test_scaled)
    np.save(os.path.join(OUTPUT_DIR, 'y_class_test_real.npy'), y_class_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_intention_test_real.npy'), y_intention_test)
    
    # save scaler for future use
    import pickle
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nfinal dataset sizes:")
    print(f"training: {X_train_scaled.shape} samples")
    print(f"validation: {X_val_scaled.shape} samples")
    print(f"test: {X_test_scaled.shape} samples")
    
    print(f"\nbraking label distribution:")
    print(f"train: {np.mean(y_class_train)*100:.1f}% braking")
    print(f"val: {np.mean(y_class_val)*100:.1f}% braking")
    print(f"test: {np.mean(y_class_test)*100:.1f}% braking")
    
    print(f"\ncomprehensive preprocessing completed successfully!")
    return True

if __name__ == "__main__":
    process_all_data()
