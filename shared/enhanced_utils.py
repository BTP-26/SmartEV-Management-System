import os
import sys
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from shared.config import get_config
from modules.braking.models.multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention
from modules.soc.models.lstm_cnn_attention_soc import LSTMCNNAttentionSoC
from shared.cognitive_manager import CognitiveEnergyManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InputValidator:
    """validates input data for the unified pipeline"""
    
    def __init__(self, config):
        self.config = config
        self.braking_config = config.get_braking_model_config()
        self.soc_config = config.get_soc_model_config()
    
    def validate_braking_input(self, driving_window: np.ndarray) -> bool:
        try:
            if not isinstance(driving_window, np.ndarray):
                raise ValueError("driving window must be numpy array")
            
            expected_shape = (self.braking_config.get('window_size', 75), 
                           self.braking_config.get('input_dim', 7))
            if driving_window.shape != expected_shape:
                raise ValueError(f"driving window shape must be {expected_shape}, got {driving_window.shape}")
            
            if driving_window.dtype != np.float32:
                logger.warning(f"driving window dtype is {driving_window.dtype}, converting to float32")
                driving_window = driving_window.astype(np.float32)
            
            if np.any(np.isnan(driving_window)) or np.any(np.isinf(driving_window)):
                raise ValueError("driving window contains nan or inf values")
            
            if np.abs(driving_window).max() > 1000:
                logger.warning("driving window contains very large values, consider normalization")
            
            return True
            
        except Exception as e:
            logger.error(f"braking input validation failed: {e}")
            return False
    
    def validate_soc_input(self, battery_window: np.ndarray) -> bool:
        try:
            if not isinstance(battery_window, np.ndarray):
                raise ValueError("battery window must be numpy array")
            
            expected_shape = (self.soc_config.get('window_size', 50), 
                           self.soc_config.get('input_dim', 3))
            if battery_window.shape != expected_shape:
                raise ValueError(f"battery window shape must be {expected_shape}, got {battery_window.shape}")
            
            if battery_window.dtype != np.float32:
                logger.warning(f"battery window dtype is {battery_window.dtype}, converting to float32")
                battery_window = battery_window.astype(np.float32)
            
            if np.any(np.isnan(battery_window)) or np.any(np.isinf(battery_window)):
                raise ValueError("battery window contains nan or inf values")
            
            if battery_window.shape[1] >= 1:
                voltage = battery_window[:, 0]
                if np.any(np.abs(voltage) > 5):
                    logger.warning("normalized voltage values seem unusual (abs > 5)")
            
            return True
            
        except Exception as e:
            logger.error(f"soc input validation failed: {e}")
            return False
    
    def validate_soc_value(self, current_soc: float) -> bool:
        try:
            if not isinstance(current_soc, (int, float)):
                raise ValueError("current soc must be a number")
            
            if not 0 <= current_soc <= 1:
                raise ValueError("current soc must be between 0 and 1")
            
            return True
            
        except Exception as e:
            logger.error(f"soc value validation failed: {e}")
            return False

class ModelQuantizer:
    """handles model quantization for faster inference"""
    
    @staticmethod
    def quantize_model(model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear: "dynamic", nn.LSTM: "dynamic", nn.Conv1d: "dynamic"}
            )
            return quantized_model
            
        except Exception as e:
            logger.warning(f"model quantization failed: {e}, using original model")
            return model

class EnhancedEVPipeline:
    """enhanced ev smart management system with all improvements"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config() if config_path is None else get_config()
        self.device = self.config.get_device()
        
        # setup paths and configs first
        self.paths = self.config.get_paths_config()
        self.inference_config = self.config.get_inference_config()
        self.performance_config = self.config.get_performance_config()
        
        # initialize components
        self.validator = InputValidator(self.config)
        self.quantizer = ModelQuantizer()
        self.cognitive_manager = CognitiveEnergyManager()
        
        # initialize models
        self.braking_model = None
        self.soc_model = None
        self.braking_model_quantized = None
        self.soc_model_quantized = None
        
        # load models
        self._load_models()
        
        logger.info("enhanced ev pipeline initialized successfully")
    
    def _load_models(self):
        try:
            # load braking model
            self._load_braking_model()
        except Exception as e:
            logger.error(f"failed to load braking model: {e}")
            raise RuntimeError(f"braking model loading failed: {e}")
        
        try:
            # load soc model
            self._load_soc_model()
        except Exception as e:
            logger.error(f"failed to load soc model: {e}")
            raise RuntimeError(f"soc model loading failed: {e}")
    
    def _load_braking_model(self):
        model_path = os.path.join(self.paths['models']['braking'], 'final_multitask_model.pth')
        hp_path = os.path.join(self.paths['models']['braking'], 'best_ga_hyperparams.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"braking model not found: {model_path}")
        
        try:
            if os.path.exists(hp_path):
                with open(hp_path) as f:
                    hp = json.load(f)["hyperparams"]
                self.braking_model = MultitaskLSTMCNNAttention(
                    input_dim=7,
                    cnn_channels=32,  
                    lstm_hidden=hp.get("lstm_hidden_size", 64),
                    num_lstm_layers=1, 
                    dropout_rate=0.0, 
                )
            else:
                braking_config = self.config.get_braking_model_config()
                self.braking_model = MultitaskLSTMCNNAttention(
                    input_dim=braking_config.get('input_dim', 7),
                    cnn_channels=braking_config.get('cnn_channels', 32),
                    lstm_hidden=braking_config.get('lstm_hidden', 64),
                    num_lstm_layers=braking_config.get('num_lstm_layers', 1),
                    dropout_rate=braking_config.get('dropout_rate', 0.0), 
                )
            
            # load state dict
            state_dict = torch.load(model_path, map_location=self.device)
            self.braking_model.load_state_dict(state_dict)
            self.braking_model.to(self.device)
            self.braking_model.eval()
            
            if self.inference_config.get('quantization', True):
                example_input = torch.randn(1, 75, 7).to(self.device)
                self.braking_model_quantized = self.quantizer.quantize_model(
                    self.braking_model, example_input
                )
            
            logger.info("braking model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"error loading braking model: {e}")
    
    def _load_soc_model(self):
        model_path = os.path.join(self.paths['models']['soc'], 'lstm_cnn_attention_soc.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"soc model not found: {model_path}")
        
        try:
            self.soc_model = LSTMCNNAttentionSoC(
                input_dim=3,
                cnn_channels=64,  
                lstm_hidden=128,  
                num_lstm_layers=2,  
                dropout=0.2,  
            )
            
            state_dict = torch.load(model_path, map_location=self.device)
            self.soc_model.load_state_dict(state_dict)
            self.soc_model.to(self.device)
            self.soc_model.eval()
            
            if self.inference_config.get('quantization', True):
                example_input = torch.randn(1, 50, 3).to(self.device)
                self.soc_model_quantized = self.quantizer.quantize_model(
                    self.soc_model, example_input
                )
            
            logger.info("soc model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"error loading soc model: {e}")
    
    def run_single(self, driving_window: np.ndarray, battery_window: np.ndarray, 
                  current_soc: float) -> Dict[str, Any]:
        """run single inference with validation"""
        if self.inference_config.get('validate_inputs', True):
            if not self.validator.validate_braking_input(driving_window):
                raise ValueError("invalid braking input")
            if not self.validator.validate_soc_input(battery_window):
                raise ValueError("invalid soc input")
            if not self.validator.validate_soc_value(current_soc):
                raise ValueError("invalid current soc value")
        
        driving_tensor = torch.tensor(driving_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        battery_tensor = torch.tensor(battery_window, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            braking_model = self.braking_model_quantized or self.braking_model
            soc_model = self.soc_model_quantized or self.soc_model
            
            class_logits, intensity = braking_model(driving_tensor)
            class_pred = torch.argmax(class_logits, dim=1).item()
            intensity_val = torch.sigmoid(intensity).item()
            
            soc_pred = soc_model(battery_tensor).item()
            soc_pred = np.clip(soc_pred, 0, 1)
        
        return self._process_results(class_pred, intensity_val, soc_pred, current_soc)
    
    def run_batch(self, driving_windows: List[np.ndarray], battery_windows: List[np.ndarray], 
                 current_socs: List[float]) -> List[Dict[str, Any]]:
        """run batch inference"""
        if len(driving_windows) != len(battery_windows) or len(driving_windows) != len(current_socs):
            raise ValueError("all input lists must have the same length")
        
        # validate all inputs
        if self.inference_config.get('validate_inputs', True):
            for i, (dw, bw, soc) in enumerate(zip(driving_windows, battery_windows, current_socs)):
                if not self.validator.validate_braking_input(dw):
                    raise ValueError(f"Invalid braking input at index {i}")
                if not self.validator.validate_soc_input(bw):
                    raise ValueError(f"Invalid SoC input at index {i}")
                if not self.validator.validate_soc_value(soc):
                    raise ValueError(f"Invalid current SoC at index {i}")
        
        # convert to tensors
        driving_tensor = torch.tensor(np.stack(driving_windows), dtype=torch.float32).to(self.device)
        battery_tensor = torch.tensor(np.stack(battery_windows), dtype=torch.float32).to(self.device)
        
        # run inference
        with torch.no_grad():
            # use quantized models if available
            braking_model = self.braking_model_quantized or self.braking_model
            soc_model = self.soc_model_quantized or self.soc_model
            
            # batch predictions
            class_logits, intensities = braking_model(driving_tensor)
            class_preds = torch.argmax(class_logits, dim=1).cpu().numpy()
            intensity_vals = torch.sigmoid(intensities).cpu().numpy()
            
            soc_preds = soc_model(battery_tensor).cpu().numpy()
            soc_preds = np.clip(soc_preds, 0, 1)  # Ensure valid range
        
        # Process results
        results = []
        for i in range(len(driving_windows)):
            result = self._process_results(
                class_preds[i], intensity_vals[i], soc_preds[i], current_socs[i]
            )
            results.append(result)
        
        return results
    
    def run_with_cognitive(self, driving_window: np.ndarray, battery_window: np.ndarray, 
                           current_soc: float, driver_id: str = "default_driver") -> Dict[str, Any]:
        """run inference with cognitive driver profiling"""
        if self.inference_config.get('validate_inputs', True):
            if not self.validator.validate_braking_input(driving_window):
                raise ValueError("invalid braking input")
            if not self.validator.validate_soc_input(battery_window):
                raise ValueError("invalid soc input")
            if not self.validator.validate_soc_value(current_soc):
                raise ValueError("invalid current soc value")
        
        basic_result = self.run_single(driving_window, battery_window, current_soc)
        
        braking_class = basic_result['braking']['class_id']
        braking_intensity = basic_result['braking']['intensity']
        
        vehicle_state = {
            'battery_temp': 25.0,
            'motor_temp': 30.0,
            'avg_speed': np.mean(driving_window[:, 0]),
            'base_regen_efficiency': self.performance_config.get('regen_efficiency', 0.65)
        }
        
        try:
            cognitive_result = self.cognitive_manager.process_driving_event(
                driver_id=driver_id,
                driving_window=driving_window,
                braking_class=braking_class,
                intensity=braking_intensity,
                current_soc=current_soc,
                vehicle_state=vehicle_state
            )
            
            enhanced_result = basic_result.copy()
            enhanced_result['cognitive'] = {
                'driver_profile': cognitive_result['driver_profile'],
                'soc_adjustment': cognitive_result['soc_adjustment'],
                'prediction_confidence': cognitive_result['prediction_confidence'],
                'energy_recovery': cognitive_result['energy_recovery'],
                'recommendations': cognitive_result['recommendations'],
                'cognitive_insights': cognitive_result['cognitive_insights']
            }
            if cognitive_result['prediction_confidence'] > 0.7:
                enhanced_result['soc']['updated'] = np.clip(
                    current_soc + cognitive_result['soc_adjustment'], 0, 1
                )
                enhanced_result['soc']['delta'] = enhanced_result['soc']['updated'] - current_soc
            
            driver_style = cognitive_result['driver_profile']['driving_style']
            if driver_style == 'eco':
                enhanced_result['system_action'] = "COGNITIVE_ECO: Optimized regenerative braking for efficiency"
            elif driver_style == 'aggressive':
                enhanced_result['system_action'] = "COGNITIVE_SPORT: Performance-oriented regenerative braking"
            elif driver_style == 'conservative':
                enhanced_result['system_action'] = "COGNITIVE_SAFE: Battery-protective regenerative braking"
            else:
                enhanced_result['system_action'] = enhanced_result['system_action']
            
            logger.info(f"Cognitive processing completed for driver {driver_id}")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Cognitive processing failed: {e}, using basic results")
            return basic_result
    
    def get_cognitive_summary(self) -> Dict[str, Any]:
        try:
            return self.cognitive_manager.get_cognitive_summary()
        except Exception as e:
            logger.error(f"Failed to get cognitive summary: {e}")
            return {"error": str(e), "cognitive_available": False}
    
    def run(self, driving_window: Union[np.ndarray, List[np.ndarray]], 
            battery_window: Union[np.ndarray, List[np.ndarray]], 
            current_soc: Union[float, List[float]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Run inference with automatic batch/single detection."""
        # Determine if batch or single
        if isinstance(driving_window, list):
            return self.run_batch(driving_window, battery_window, current_soc)
        else:
            return self.run_single(driving_window, battery_window, current_soc)
    
    def _process_results(self, class_pred: int, intensity_val: float, 
                       soc_pred: float, current_soc: float) -> Dict[str, Any]:
        """Process inference results into unified output."""
        class_labels = ["Light Braking", "Normal Braking", "Emergency Braking"]
        class_label = class_labels[class_pred]
        
        regen_efficiency = self.performance_config.get('regen_efficiency', 0.65)
        energy_recovered = intensity_val * regen_efficiency
        
        soc_update_rate = self.performance_config.get('soc_update_rate', 1.0)
        updated_soc = np.clip(current_soc + energy_recovered * soc_update_rate, 0, 1)
        soc_delta = updated_soc - current_soc
        
        # Determine system action
        system_action = self._determine_system_action(class_label, intensity_val, updated_soc)
        
        return {
            "braking": {
                "class": class_label,
                "class_id": class_pred,
                "intensity": intensity_val,
                "confidence": max(0.5, 1.0 - intensity_val)  # Simple confidence estimate
            },
            "energy": {
                "recovered_normalised": energy_recovered,
                "regen_efficiency": regen_efficiency,
                "intensity_raw": intensity_val
            },
            "soc": {
                "estimated": soc_pred,
                "current": current_soc,
                "updated": updated_soc,
                "delta": soc_delta
            },
            "system_action": system_action,
            "metadata": {
                "timestamp": time.time(),
                "device": str(self.device),
                "quantized": self.braking_model_quantized is not None
            }
        }
    
    def _determine_system_action(self, class_label: str, intensity_val: float, 
                               updated_soc: float) -> str:
        if class_label == "Emergency Braking":
            return "EMERGENCY: Maximum regenerative braking - safety mode"
        elif class_label == "Normal Braking":
            if updated_soc < 0.2:
                return "REGEN: Normal regenerative braking - battery protection mode"
            else:
                return "REGEN: Normal regenerative braking - efficiency mode"
        else:  # Light Braking
            if updated_soc > 0.9:
                return "REGEN: Light regenerative braking - battery full mode"
            else:
                return "REGEN: Light regenerative braking — comfort mode"
    
    def generate_sample_inputs(self, num_samples: int = 1) -> Tuple[Union[np.ndarray, List[np.ndarray]], 
                                                              Union[np.ndarray, List[np.ndarray]]]:
        if num_samples == 1:
            return self._generate_single_sample()
        else:
            return self._generate_batch_samples(num_samples)
    
    def _generate_single_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        # Driving window: (75, 7) - acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, speed
        t = np.linspace(0, 1, 75)
        acc_x = np.random.normal(0, 0.5, 75)
        acc_y = np.random.normal(0, 0.5, 75)
        acc_z = np.random.normal(0, 0.5, 75)
        gyro_x = np.random.normal(0, 0.1, 75)
        gyro_y = np.random.normal(0, 0.1, 75)
        gyro_z = np.random.normal(0, 0.1, 75)
        speed = 60 - 30 * t + np.random.normal(0, 0.5, 75)
        driving_window = np.stack([acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, speed], axis=1).astype(np.float32)
        
        # Normalize
        mean = driving_window.mean(axis=0)
        std = driving_window.std(axis=0) + 1e-8
        driving_window = (driving_window - mean) / std
        
        # Battery window: (50, 3) - voltage, current, temperature
        voltage = 3.8 - 0.5 * np.linspace(0, 1, 50) + np.random.normal(0, 0.01, 50)
        current = -1.0 * np.ones(50) + np.random.normal(0, 0.05, 50)
        temperature = 25 + np.random.normal(0, 0.5, 50)
        battery_window = np.stack([voltage, current, temperature], axis=1).astype(np.float32)
        
        mean = battery_window.mean(axis=0)
        std = battery_window.std(axis=0) + 1e-8
        battery_window = (battery_window - mean) / std
        
        return driving_window, battery_window
    
    def _generate_batch_samples(self, num_samples: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        driving_windows = []
        battery_windows = []
        
        for _ in range(num_samples):
            dw, bw = self._generate_single_sample()
            driving_windows.append(dw)
            battery_windows.append(bw)
        
        return driving_windows, battery_windows
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "braking_model_loaded": self.braking_model is not None,
            "soc_model_loaded": self.soc_model is not None,
            "braking_model_quantized": self.braking_model_quantized is not None,
            "soc_model_quantized": self.soc_model_quantized is not None,
            "device": str(self.device),
            "config": {
                "quantization_enabled": self.inference_config.get('quantization', True),
                "input_validation": self.inference_config.get('validate_inputs', True),
                "batch_size": self.inference_config.get('batch_size', 1)
            }
        }
