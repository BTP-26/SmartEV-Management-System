import os
import numpy as np
import yaml
from typing import Tuple, Dict, Any
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

class DatasetLoader:
    """Utility class to load datasets based on configuration."""
    
    def __init__(self, config_path: str = "config/dataset_config.yaml"):
        """Initialize dataset loader with configuration."""
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load dataset configuration."""
        config_file = os.path.join(project_root, self.config_path)
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def get_dataset_paths(self, module: str) -> Dict[str, str]:
        """Get dataset paths for a specific module based on configuration."""
        if module not in ['soc', 'braking']:
            raise ValueError(f"Unknown module: {module}")
        
        module_config = self.config[module]
        source = module_config['source']
        
        if source == 'real':
            return module_config['real_data']
        elif source == 'simulated':
            return module_config['simulated_data']
        else:
            raise ValueError(f"Unknown dataset source: {source}")
    
    def load_soc_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load SoC dataset based on configuration."""
        paths = self.get_dataset_paths('soc')
        
        X_train = np.load(os.path.join(project_root, paths['train_file']))
        X_val = np.load(os.path.join(project_root, paths['val_file']))
        X_test = np.load(os.path.join(project_root, paths['test_file']))
        y_train = np.load(os.path.join(project_root, paths['y_train_file']))
        y_val = np.load(os.path.join(project_root, paths['y_val_file']))
        y_test = np.load(os.path.join(project_root, paths['y_test_file']))
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def load_braking_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                          np.ndarray, np.ndarray, np.ndarray,
                                          np.ndarray, np.ndarray, np.ndarray]:
        """Load braking dataset based on configuration."""
        paths = self.get_dataset_paths('braking')
        
        X_train = np.load(os.path.join(project_root, paths['train_file']))
        X_val = np.load(os.path.join(project_root, paths['val_file']))
        X_test = np.load(os.path.join(project_root, paths['test_file']))
        y_int_train = np.load(os.path.join(project_root, paths['y_int_train_file']))
        y_int_val = np.load(os.path.join(project_root, paths['y_int_val_file']))
        y_int_test = np.load(os.path.join(project_root, paths['y_int_test_file']))
        y_class_train = np.load(os.path.join(project_root, paths['y_class_train_file']))
        y_class_val = np.load(os.path.join(project_root, paths['y_class_val_file']))
        y_class_test = np.load(os.path.join(project_root, paths['y_class_test_file']))
        
        return (X_train, X_val, X_test, 
                y_int_train, y_int_val, y_int_test,
                y_class_train, y_class_val, y_class_test)
    
    def get_dataset_info(self, module: str) -> Dict[str, Any]:
        """Get information about the current dataset for a module."""
        if module == 'soc':
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_soc_dataset()
            
            return {
                'module': 'soc',
                'source': self.config['soc']['source'],
                'train_samples': X_train.shape[0],
                'val_samples': X_val.shape[0],
                'test_samples': X_test.shape[0],
                'sequence_length': X_train.shape[1],
                'features': X_train.shape[2],
                'target_range': (float(y_train.min()), float(y_train.max())),
                'target_mean': float(y_train.mean()),
                'target_std': float(y_train.std())
            }
        
        elif module == 'braking':
            (X_train, X_val, X_test, 
             y_int_train, y_int_val, y_int_test,
             y_class_train, y_class_val, y_class_test) = self.load_braking_dataset()
            
            return {
                'module': 'braking',
                'source': self.config['braking']['source'],
                'train_samples': X_train.shape[0],
                'val_samples': X_val.shape[0],
                'test_samples': X_test.shape[0],
                'sequence_length': X_train.shape[1],
                'features': X_train.shape[2],
                'intention_classes': len(np.unique(y_int_train)),
                'behavior_classes': len(np.unique(y_class_train)),
                'intention_distribution': dict(zip(*np.unique(y_int_train, return_counts=True))),
                'behavior_distribution': dict(zip(*np.unique(y_class_train, return_counts=True)))
            }
        
        else:
            raise ValueError(f"Unknown module: {module}")

def get_dataset_loader() -> DatasetLoader:
    """Get a dataset loader instance."""
    return DatasetLoader()

if __name__ == "__main__":
    # Test the dataset loader
    loader = get_dataset_loader()
    
    print("Dataset Loader Test")
    print("=" * 50)
    
    # Test SoC dataset
    soc_info = loader.get_dataset_info('soc')
    print(f"SoC Dataset ({soc_info['source']}):")
    print(f"  Train: {soc_info['train_samples']} samples")
    print(f"  Val:   {soc_info['val_samples']} samples")
    print(f"  Test:  {soc_info['test_samples']} samples")
    print(f"  Shape: {soc_info['sequence_length']} x {soc_info['features']}")
    print(f"  Target range: {soc_info['target_range']}")
    print()
    
    # Test Braking dataset
    braking_info = loader.get_dataset_info('braking')
    print(f"Braking Dataset ({braking_info['source']}):")
    print(f"  Train: {braking_info['train_samples']} samples")
    print(f"  Val:   {braking_info['val_samples']} samples")
    print(f"  Test:  {braking_info['test_samples']} samples")
    print(f"  Shape: {braking_info['sequence_length']} x {braking_info['features']}")
    print(f"  Intention classes: {braking_info['intention_classes']}")
    print(f"  Behavior classes: {braking_info['behavior_classes']}")
    print(f"  Intention distribution: {braking_info['intention_distribution']}")
    print(f"  Behavior distribution: {braking_info['behavior_distribution']}")
