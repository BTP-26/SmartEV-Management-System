
import os
import yaml
import torch
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._setup_paths()
        
    def _get_default_config_path(self) -> str:
        project_root = Path(__file__).parent.parent
        return str(project_root / "config" / "default.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"invalid yaml configuration: {e}")
    
    def _setup_paths(self):
        project_root = Path(__file__).parent.parent
        
        if 'paths' in self.config and 'data' in self.config['paths']:
            for key, path in self.config['paths']['data'].items():
                self.config['paths']['data'][key] = str(project_root / path)
        
        if 'paths' in self.config and 'models' in self.config['paths']:
            for key, path in self.config['paths']['models'].items():
                self.config['paths']['models'][key] = str(project_root / path)
        
        if 'paths' in self.config:
            for key, path in self.config['paths'].items():
                if key not in ['data', 'models']:
                    self.config['paths'][key] = str(project_root / path)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def get_device(self) -> torch.device:
        device_config = self.get('system.device', 'auto')
        
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cuda':
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                return torch.device('cpu')
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
    def get_braking_model_config(self) -> Dict[str, Any]:
        return self.get('models.braking', {})
    
    def get_soc_model_config(self) -> Dict[str, Any]:
        return self.get('models.soc', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        return self.get('training', {})
    
    def get_data_config(self, module: str) -> Dict[str, Any]:
        return self.get(f'data.{module}', {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        return self.get('inference', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        return self.get('performance', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        return self.get('paths', {})
    
    def save(self, config_path: Optional[str] = None):
        save_path = config_path or self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def __str__(self) -> str:
        return yaml.dump(self.config, default_flow_style=False, indent=2)

config = Config()

def get_config() -> Config:
    return config

def reload_config(config_path: Optional[str] = None):
    global config
    config = Config(config_path)
    return config
