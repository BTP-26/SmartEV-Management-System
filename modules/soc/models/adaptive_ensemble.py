
# Combines multiple model types with GA-optimized ensemble weights


import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.soc.models.lstm_cnn_attention_soc import (
    LSTMCNNAttentionSoC, train_soc_model, evaluate_soc_model
)
from shared.dataset_loader import get_dataset_loader
from shared.train_utils import calculate_regression_metrics
from modules.soc.soc_scale import load_soc_scale, inverse_scale_soc
from modules.soc.data.preprocess_real_data import CAPACITY_AH
from modules.soc.models.coulomb_counting import coulomb_counting_soc
from modules.soc.models.battery_rls_identification import list_cycle_folders, load_cycle


class TransformerSoCModel(nn.Module):
    """Transformer-based SoC estimation model."""
    
    def __init__(self, input_dim=3, d_model=128, nhead=8, num_layers=3, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # Input: (batch, seq_len, input_dim)
        x = self.input_projection(x) * np.sqrt(self.d_model)
        x = self.transformer(x)
        # Global average pooling
        x = torch.mean(x, dim=1)
        output = self.regressor(x)
        return output.squeeze(-1)


class PhysicsInformedSoCModel(nn.Module):
    """Physics-informed neural network for SoC estimation."""
    
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Feature extraction layers
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        
        # Physics constraints layer
        self.physics_layer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # SoC estimation layer
        self.soc_regressor = nn.Sequential(
            nn.Linear(32 + 3, 16),  # +3 for physics features
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Ensure SoC in [0, 1]
        )
    
    def _compute_physics_features(self, x):
        """Compute physics-based features from input."""
        # x: (batch, seq_len, input_dim) -> [voltage, current, temperature]
        voltage = x[:, :, 0]  # (batch, seq_len)
        current = x[:, :, 1]   # (batch, seq_len)
        temp = x[:, :, 2]      # (batch, seq_len)
        
        # Physics-based features
        avg_voltage = torch.mean(voltage, dim=1)
        avg_current = torch.mean(current, dim=1)
        avg_temp = torch.mean(temp, dim=1)
        
        # Power estimation (simplified)
        avg_power = avg_voltage * avg_current
        
        # Temperature effect on capacity (simplified model)
        temp_effect = torch.sigmoid((avg_temp - 25) / 10)  # Normalized around 25°C
        
        # Current integration for SoC change (Coulomb counting approximation)
        current_integral = torch.trapz(current, dim=1)  # Approximate integral
        
        return torch.stack([avg_voltage, avg_power, temp_effect], dim=1)
    
    def forward(self, x):
        # Extract features from sequence
        batch_size, seq_len, _ = x.shape
        
        # Process each timestep and aggregate
        features = []
        for t in range(seq_len):
            timestep_features = self.feature_extractor(x[:, t, :])
            features.append(timestep_features)
        
        # Aggregate features (mean pooling)
        aggregated_features = torch.mean(torch.stack(features, dim=1), dim=1)
        
        # Physics features
        physics_features = self._compute_physics_features(x)
        
        # Combine neural and physics features
        combined_features = torch.cat([
            self.physics_layer(aggregated_features),
            physics_features
        ], dim=1)
        
        # Final SoC estimation
        soc_output = self.soc_regressor(combined_features)
        return soc_output.squeeze(-1)


@dataclass
class EnsembleWeights:
    """Weights for ensemble models."""
    lstm_cnn_weight: float
    transformer_weight: float
    physics_weight: float
    
    def as_array(self) -> np.ndarray:
        return np.array([self.lstm_cnn_weight, self.transformer_weight, self.physics_weight])
    
    def normalize(self):
        """Normalize weights to sum to 1."""
        total = self.lstm_cnn_weight + self.transformer_weight + self.physics_weight
        if total > 0:
            self.lstm_cnn_weight /= total
            self.transformer_weight /= total
            self.physics_weight /= total


class AdaptationReferenceSignal(ABC):
    """B4: a deployable (no continuous ground truth) reference signal used to drive
    ensemble weight adaptation. CoulombCountingReferenceSignal below is the initial
    implementation; a future reference signal (e.g. an OCV-based estimate, or the literal
    alpha-fusion blend in roadmap L-B2) is a new class implementing this same interface -
    no changes needed to AdaptiveEnsembleSoC or run_adaptive_trajectory()."""

    @abstractmethod
    def predict(self, cycle: Dict[str, np.ndarray], window_end_indices: np.ndarray) -> np.ndarray:
        """Return a [0,1]-scale SoC reference at each given window-end index into `cycle`."""
        ...


class CoulombCountingReferenceSignal(AdaptationReferenceSignal):
    """Wraps coulomb_counting_soc() (modules/soc/models/coulomb_counting.py, B3). Needs
    exactly one true-SoC anchor per cycle (cycle["soc"][0]) - a single per-trip reference
    reading (e.g. a full-charge calibration), not continuous ground truth. This is
    exactly what "deployable" means in the CC literature (roadmap L-B2), but it must be
    stated plainly here, not glossed over."""

    def __init__(self, capacity_ah: float = CAPACITY_AH):
        self.capacity_ah = capacity_ah

    def predict(self, cycle: Dict[str, np.ndarray], window_end_indices: np.ndarray) -> np.ndarray:
        soc_init = cycle["soc"][0] / 100.0  # the one true-SoC anchor this signal uses
        full_cc = coulomb_counting_soc(cycle["curr"], cycle["t"], self.capacity_ah, soc_init=soc_init)
        return full_cc[window_end_indices]


class AdaptiveEnsembleSoC:
    """Adaptive ensemble of SoC models with GA-optimized weights."""
    
    def __init__(self, device=None):

        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Mac GPU
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")   
                    
        # Initialize models
        self.lstm_cnn_model = None
        self.transformer_model = None
        self.physics_model = None
        
        # Ensemble weights
        self.weights = EnsembleWeights(0.4, 0.3, 0.3)
        
        # Performance tracking for adaptive weight adjustment
        self.performance_history = {
            'lstm_cnn': [],
            'transformer': [],
            'physics': []
        }
        
        self.adaptation_rate = 0.1  # Learning rate for weight adaptation
    
    def _create_lstm_cnn_model(self, **kwargs):
        """Create LSTM-CNN-Attention model."""
        return LSTMCNNAttentionSoC(**kwargs)
    
    def _create_transformer_model(self, **kwargs):
        """Create Transformer model."""
        return TransformerSoCModel(**kwargs)
    
    def _create_physics_model(self, **kwargs):
        """Create Physics-informed model."""
        return PhysicsInformedSoCModel(**kwargs)
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, 
                     lstm_params=None, transformer_params=None, physics_params=None):
        """Train all models in the ensemble."""
        print("Training Adaptive Ensemble SoC Models...")
        
        # Default parameters
        lstm_params = lstm_params or {
            'cnn_channels': 64, 'lstm_hidden': 128, 'num_lstm_layers': 2, 'dropout': 0.2
        }
        transformer_params = transformer_params or {
            'input_dim': 3, 'd_model': 128, 'nhead': 8, 'num_layers': 3, 'dropout': 0.2
        }
        physics_params = physics_params or {
            'input_dim': 3, 'hidden_dim': 128, 'num_layers': 3, 'dropout': 0.2
        }
        
        # Train LSTM-CNN model
        print("  Training LSTM-CNN-Attention model...")
        self.lstm_cnn_model = self._create_lstm_cnn_model(**lstm_params)
        self.lstm_cnn_model.to(self.device)
        
        _, lstm_history = train_soc_model(
            self.lstm_cnn_model, X_train, y_train, X_val, y_val,
            lr=0.001, batch_size=64, epochs=5, patience=2,
            device=self.device, save_path="modules/soc/models/ensemble_lstm_cnn.pth"
        )
        
        # Train Transformer model
        print("  Training Transformer model...")
        self.transformer_model = self._create_transformer_model(**transformer_params)
        self.transformer_model.to(self.device)
        
        _, transformer_history = train_soc_model(
            self.transformer_model, X_train, y_train, X_val, y_val,
            lr=0.001, batch_size=64, epochs=5, patience=2,
            device=self.device, save_path="modules/soc/models/ensemble_transformer.pth"
        )
        
        # Train Physics-informed model
        print("  Training Physics-informed model...")
        self.physics_model = self._create_physics_model(**physics_params)
        self.physics_model.to(self.device)
        
        _, physics_history = train_soc_model(
            self.physics_model, X_train, y_train, X_val, y_val,
            lr=0.001, batch_size=64, epochs=5, patience=2,
            device=self.device, save_path="modules/soc/models/ensemble_physics.pth"
        )
        
        # Store final validation RMSE for initial weight optimization
        lstm_rmse = min(lstm_history['val_rmse'])
        transformer_rmse = min(transformer_history['val_rmse'])
        physics_rmse = min(physics_history['val_rmse'])
        
        # Initialize weights based on inverse RMSE (better model gets higher weight)
        inv_rmse_sum = (1/lstm_rmse + 1/transformer_rmse + 1/physics_rmse)
        self.weights.lstm_cnn_weight = (1/lstm_rmse) / inv_rmse_sum
        self.weights.transformer_weight = (1/transformer_rmse) / inv_rmse_sum
        self.weights.physics_weight = (1/physics_rmse) / inv_rmse_sum
        
        print(f"  Initial ensemble weights: LSTM-CNN={self.weights.lstm_cnn_weight:.3f}, "
              f"Transformer={self.weights.transformer_weight:.3f}, "
              f"Physics={self.weights.physics_weight:.3f}")
        
        return {
            'lstm_rmse': lstm_rmse,
            'transformer_rmse': transformer_rmse,
            'physics_rmse': physics_rmse
        }
    
    def forward(self, x):
        """Forward pass through ensemble."""
        if not all([self.lstm_cnn_model, self.transformer_model, self.physics_model]):
            raise RuntimeError("Models not trained. Call train_ensemble() first.")
        
        self.lstm_cnn_model.eval()
        self.transformer_model.eval()
        self.physics_model.eval()
        
        with torch.no_grad():
            # Get predictions from all models
            lstm_pred = self.lstm_cnn_model(x)
            transformer_pred = self.transformer_model(x)
            physics_pred = self.physics_model(x)
            
            # Weighted ensemble
            weights_tensor = torch.tensor(self.weights.as_array(), dtype=torch.float32, device=self.device)
            predictions = torch.stack([lstm_pred, transformer_pred, physics_pred], dim=1)
            ensemble_pred = torch.sum(predictions * weights_tensor.unsqueeze(0), dim=1)
            
            return ensemble_pred
    
    def evaluate_ensemble(self, X_test, y_test):
        """Evaluate ensemble performance."""
        self.lstm_cnn_model.eval()
        self.transformer_model.eval()
        self.physics_model.eval()
        
        with torch.no_grad():
            # Individual model evaluations
            lstm_pred = []
            transformer_pred = []
            physics_pred = []
            
            batch_size = 64
            for i in range(0, len(X_test), batch_size):
                batch_x = torch.tensor(X_test[i:i+batch_size], dtype=torch.float32).to(self.device)
                
                lstm_pred.append(self.lstm_cnn_model(batch_x).cpu().numpy())
                transformer_pred.append(self.transformer_model(batch_x).cpu().numpy())
                physics_pred.append(self.physics_model(batch_x).cpu().numpy())
            
            lstm_pred = np.concatenate(lstm_pred)
            transformer_pred = np.concatenate(transformer_pred)
            physics_pred = np.concatenate(physics_pred)

            # Ensemble prediction
            weights = self.weights.as_array()
            ensemble_pred = (weights[0] * lstm_pred +
                          weights[1] * transformer_pred +
                          weights[2] * physics_pred)

            # Report in % SoC via the one inverse transform (roadmap B1 step 2), not the
            # raw [0,1] training scale, so this matches evaluate_soc.py's Table 6 numbers.
            # soc_min_percent=0 makes this a pure scale factor, so weight adaptation (which
            # only depends on the *ratio* of these RMSEs, see adaptive_weight_update()) is
            # unaffected by reporting in % instead of [0,1].
            scale = load_soc_scale()
            y_test_pct = inverse_scale_soc(y_test, scale)
            lstm_pred_pct = inverse_scale_soc(lstm_pred, scale)
            transformer_pred_pct = inverse_scale_soc(transformer_pred, scale)
            physics_pred_pct = inverse_scale_soc(physics_pred, scale)
            ensemble_pred_pct = inverse_scale_soc(ensemble_pred, scale)

            # Calculate RMSE for each model and ensemble
            def rmse(y_true, y_pred):
                return np.sqrt(np.mean((y_true - y_pred) ** 2))

            lstm_rmse = rmse(y_test_pct, lstm_pred_pct)
            transformer_rmse = rmse(y_test_pct, transformer_pred_pct)
            physics_rmse = rmse(y_test_pct, physics_pred_pct)
            ensemble_rmse = rmse(y_test_pct, ensemble_pred_pct)
            
            # Update performance history
            self.performance_history['lstm_cnn'].append(lstm_rmse)
            self.performance_history['transformer'].append(transformer_rmse)
            self.performance_history['physics'].append(physics_rmse)
            
            return {
                'lstm_rmse': lstm_rmse,
                'transformer_rmse': transformer_rmse,
                'physics_rmse': physics_rmse,
                'ensemble_rmse': ensemble_rmse,
                'weights': self.weights.as_array().tolist()
            }
    
    def update_performance_from_reference(self, model_preds: Dict[str, np.ndarray],
                                           reference_pred: np.ndarray) -> None:
        """B4: the deployable counterpart to evaluate_ensemble()'s ground-truth-based
        performance_history population. Computes each sub-model's mean absolute residual
        against `reference_pred` (from an AdaptationReferenceSignal - e.g.
        CoulombCountingReferenceSignal) instead of RMSE against true SoC, so this path
        never needs continuous ground truth. Feeds the same performance_history that
        adaptive_weight_update() reads - this is what actually replaces the ground-truth
        error signal (roadmap B4 step 1)."""
        for key, preds in (('lstm_cnn', model_preds['lstm_cnn']),
                            ('transformer', model_preds['transformer']),
                            ('physics', model_preds['physics'])):
            residual = float(np.mean(np.abs(np.asarray(preds) - reference_pred)))
            self.performance_history[key].append(residual)

    def adaptive_weight_update(self, recent_performance_window=5):
        """Adaptively update ensemble weights based on recent performance."""
        if len(self.performance_history['lstm_cnn']) < recent_performance_window:
            return
        
        # Get recent performance
        recent_lstm = np.mean(self.performance_history['lstm_cnn'][-recent_performance_window:])
        recent_transformer = np.mean(self.performance_history['transformer'][-recent_performance_window:])
        recent_physics = np.mean(self.performance_history['physics'][-recent_performance_window:])
        
        # Calculate new weights based on inverse recent performance
        inv_perf_sum = (1/recent_lstm + 1/recent_transformer + 1/recent_physics)
        
        new_lstm_weight = (1/recent_lstm) / inv_perf_sum
        new_transformer_weight = (1/recent_transformer) / inv_perf_sum
        new_physics_weight = (1/recent_physics) / inv_perf_sum
        
        # Smooth weight update
        self.weights.lstm_cnn_weight = (1 - self.adaptation_rate) * self.weights.lstm_cnn_weight + \
                                    self.adaptation_rate * new_lstm_weight
        self.weights.transformer_weight = (1 - self.adaptation_rate) * self.weights.transformer_weight + \
                                     self.adaptation_rate * new_transformer_weight
        self.weights.physics_weight = (1 - self.adaptation_rate) * self.weights.physics_weight + \
                                   self.adaptation_rate * new_physics_weight
        
        self.weights.normalize()
    
    def save_ensemble(self, path="modules/soc/models/adaptive_ensemble.json"):
        """Save ensemble configuration."""
        # Convert numpy types to regular Python types
        def convert_types(obj):
            if hasattr(obj, 'tolist'):  # numpy arrays
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, 'item'):  # numpy scalar types
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        ensemble_config = {
            'weights': convert_types(self.weights.as_array()),
            'adaptation_rate': self.adaptation_rate,
            'performance_history': convert_types(self.performance_history)
        }
        
        with open(path, 'w') as f:
            json.dump(ensemble_config, f, indent=4)
        print(f"Ensemble configuration saved: {path}")
    
    def load_ensemble(self, path="modules/soc/models/adaptive_ensemble.json"):
        """Load ensemble configuration."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                config = json.load(f)
            
            weights_array = config['weights']
            self.weights = EnsembleWeights(weights_array[0], weights_array[1], weights_array[2])
            self.adaptation_rate = config.get('adaptation_rate', 0.1)
            self.performance_history = config.get('performance_history', {
                'lstm_cnn': [], 'transformer': [], 'physics': []
            })
            
            print(f"Ensemble configuration loaded: {path}")
            return True
        return False


class GAEnsembleOptimizer:
    """Genetic Algorithm optimizer for ensemble weights."""
    
    def __init__(self, population_size=20, generations=10, mutation_rate=0.1):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
    
    def _random_weights(self):
        """Generate random ensemble weights."""
        weights = np.random.rand(3)
        return weights / np.sum(weights)  # Normalize
    
    def _crossover(self, parent1, parent2):
        """Crossover operation for weights."""
        alpha = np.random.rand()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child / np.sum(child)  # Normalize
    
    def _mutate(self, weights):
        """Mutation operation for weights."""
        mutated = weights + np.random.normal(0, 0.1, 3)
        mutated = np.abs(mutated)  # Ensure non-negative
        return mutated / np.sum(mutated)  # Normalize
    
    def optimize_weights(self, ensemble, X_val, y_val):
        """Optimize ensemble weights using GA."""
        print("Optimizing ensemble weights with GA...")
        
        # Initialize population
        population = [self._random_weights() for _ in range(self.population_size)]
        
        # Evaluate fitness (negative RMSE for maximization)
        def evaluate_weights(weights):
            ensemble.weights.lstm_cnn_weight = weights[0]
            ensemble.weights.transformer_weight = weights[1]
            ensemble.weights.physics_weight = weights[2]
            
            results = ensemble.evaluate_ensemble(X_val, y_val)
            return -results['ensemble_rmse']  # Negative for maximization
        
        fitness_scores = [evaluate_weights(weights) for weights in population]
        
        best_idx = np.argmax(fitness_scores)
        best_weights = population[best_idx]
        best_fitness = fitness_scores[best_idx]
        
        for gen in range(self.generations):
            new_population = [best_weights]  # Elitism
            
            while len(new_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                parent1_idx = max(np.random.choice(len(population), tournament_size, replace=False), 
                                key=lambda i: fitness_scores[i])
                parent2_idx = max(np.random.choice(len(population), tournament_size, replace=False), 
                                key=lambda i: fitness_scores[i])
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Crossover and mutation
                child = self._crossover(parent1, parent2)
                if np.random.rand() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
            fitness_scores = [evaluate_weights(weights) for weights in population]
            
            # Update best
            gen_best_idx = np.argmax(fitness_scores)
            if fitness_scores[gen_best_idx] > best_fitness:
                best_fitness = fitness_scores[gen_best_idx]
                best_weights = population[gen_best_idx]
            
            print(f"  Generation {gen+1}/{self.generations}: Best RMSE = {-best_fitness:.4f}")
        
        # Set best weights
        ensemble.weights.lstm_cnn_weight = best_weights[0]
        ensemble.weights.transformer_weight = best_weights[1]
        ensemble.weights.physics_weight = best_weights[2]
        
        return best_weights, -best_fitness


def _windows_from_cycle(cycle: Dict[str, np.ndarray], window_size: int = 50,
                         step: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Slice one temporally-ordered real cycle into overlapping [window_size, 3] windows
    (features [volt, curr, temp], matching preprocess_real_data.py's convention) plus the
    index into `cycle` each window ends at. Unlike X_test_real.npy's shuffled windows,
    these preserve real temporal order - no session identity survives the windowed
    dataset (same gap as EV-1/BL-4), so this reload is required, same as B3's
    coulomb_counting.py."""
    features = np.column_stack([cycle["volt"], cycle["curr"], cycle["temp"]])
    n = len(features)
    end_indices = np.arange(window_size - 1, n, step)
    X_windows = np.stack([features[e - window_size + 1: e + 1] for e in end_indices])
    return X_windows.astype(np.float32), end_indices


def run_adaptive_trajectory(ensemble: "AdaptiveEnsembleSoC", cycle: Dict[str, np.ndarray],
                             reference_signal: Optional[AdaptationReferenceSignal] = None,
                             window_size: int = 50, step: int = 10,
                             recent_performance_window: int = 5) -> pd.DataFrame:
    """B4 deliverable: walk one real, temporally-ordered cycle and log ensemble weights at
    every step, for both a fixed-weight arm (frozen at the offline-tuned starting weights
    already on `ensemble`) and an adaptive arm whose weights evolve via
    `reference_signal`-driven residuals - never the cycle's true SoC. True SoC
    (cycle["soc"]) is used ONLY afterward, by compute_fixed_vs_adaptive_ablation(), to
    score both arms - never to drive the adaptation itself (verify: grep this function for
    `cycle["soc"]` - the only other read is via `reference_signal.predict()`'s single
    per-cycle anchor, documented on CoulombCountingReferenceSignal)."""
    reference_signal = reference_signal or CoulombCountingReferenceSignal()

    X_windows, end_indices = _windows_from_cycle(cycle, window_size, step)
    reference_pred = reference_signal.predict(cycle, end_indices)
    true_soc_unit = cycle["soc"][end_indices] / 100.0

    fixed_weights = ensemble.weights.as_array().copy()
    ensemble.performance_history = {'lstm_cnn': [], 'transformer': [], 'physics': []}

    ensemble.lstm_cnn_model.eval()
    ensemble.transformer_model.eval()
    ensemble.physics_model.eval()

    log = []
    with torch.no_grad():
        for step_idx, (x_window, ref, true_val) in enumerate(zip(X_windows, reference_pred, true_soc_unit)):
            x = torch.tensor(x_window, dtype=torch.float32).unsqueeze(0).to(ensemble.device)
            lstm_pred = ensemble.lstm_cnn_model(x).cpu().numpy()
            transformer_pred = ensemble.transformer_model(x).cpu().numpy()
            physics_pred = ensemble.physics_model(x).cpu().numpy()

            model_preds = {'lstm_cnn': lstm_pred, 'transformer': transformer_pred, 'physics': physics_pred}
            ensemble.update_performance_from_reference(model_preds, np.array([ref]))
            ensemble.adaptive_weight_update(recent_performance_window=recent_performance_window)

            adaptive_weights = ensemble.weights.as_array()
            preds_vec = [float(lstm_pred[0]), float(transformer_pred[0]), float(physics_pred[0])]
            adaptive_pred = float(np.dot(adaptive_weights, preds_vec))
            fixed_pred = float(np.dot(fixed_weights, preds_vec))

            log.append({
                'step': step_idx,
                'lstm_cnn_weight': adaptive_weights[0],
                'transformer_weight': adaptive_weights[1],
                'physics_weight': adaptive_weights[2],
                'reference_pred_unit': float(ref),
                'true_soc_unit': float(true_val),
                'adaptive_pred_unit': adaptive_pred,
                'fixed_pred_unit': fixed_pred,
            })

    return pd.DataFrame(log)


def plot_weight_trajectory(log_df: pd.DataFrame, save_path: str) -> None:
    """Single source (log_df) for both the CSV and the PNG, so the figure is always
    reproducible from the raw logged values, not a separate computation."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(log_df['step'], log_df['lstm_cnn_weight'], label='LSTM-CNN', marker='.')
    plt.plot(log_df['step'], log_df['transformer_weight'], label='Transformer', marker='.')
    plt.plot(log_df['step'], log_df['physics_weight'], label='Physics', marker='.')
    plt.xlabel('Timestep (window index across cycle)')
    plt.ylabel('Ensemble weight')
    plt.title('Adaptive ensemble weight trajectory (Coulomb-Counting-residual driven)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def compute_fixed_vs_adaptive_ablation(log_df: pd.DataFrame) -> pd.DataFrame:
    """Scoring only, run after the trajectory - compares both arms against the cycle's
    true SoC (ground truth is fine and necessary here, it's what makes the ablation
    meaningful; neither arm's adaptation mechanism itself ever consumed it during
    run_adaptive_trajectory())."""
    scale = load_soc_scale()
    true_pct = inverse_scale_soc(log_df['true_soc_unit'].values, scale)
    rows = []
    for arm, col in [('Fixed Weights', 'fixed_pred_unit'), ('Adaptive (CC-residual)', 'adaptive_pred_unit')]:
        pred_pct = inverse_scale_soc(log_df[col].values, scale)
        metrics = calculate_regression_metrics(true_pct, pred_pct)
        rows.append({
            'arm': arm, 'rmse_pct_soc': metrics['rmse'], 'mae_pct_soc': metrics['mae'],
            'n_cycles': 1, 'n_steps': len(log_df),
        })
    return pd.DataFrame(rows)


def train_adaptive_ensemble():
    """Main function to train adaptive ensemble."""
    print("=== Training Adaptive Ensemble SoC Model ===")

    # Load the same real, [0,1]-scaled dataset the deployed model uses (see
    # shared/dataset_loader.py + config/dataset_config.yaml). Previously this loaded a
    # separate, orphaned *_soc.npy convention that no script in the repo produced.
    dataset_loader = get_dataset_loader()
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_loader.load_soc_dataset()

    # Use a smaller subset for faster training, but never truncate the sequence length -
    # doing so silently desynced this ensemble from the deployed model's window size (50)
    # and broke EnhancedPhysicsInformedSoC's hardcoded input_dim*25 assumption.
    sample_size = min(30000, len(X_train))
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]

    print(f"Training data shape: {X_train.shape}")
    
    # Create and train ensemble
    ensemble = AdaptiveEnsembleSoC()
    training_results = ensemble.train_ensemble(X_train, y_train, X_val, y_val)
    
    # Optimize ensemble weights with GA
    ga_optimizer = GAEnsembleOptimizer(population_size=10, generations=5)
    best_weights, best_rmse = ga_optimizer.optimize_weights(ensemble, X_val, y_val)
    
    # Final evaluation
    final_results = ensemble.evaluate_ensemble(X_val, y_val)
    
    print(f"\nADAPTIVE ENSEMBLE RESULTS:")
    print(f"Final Ensemble RMSE: {final_results['ensemble_rmse']:.4f}")
    print(f"Individual Model RMSEs:")
    print(f"  LSTM-CNN: {final_results['lstm_rmse']:.4f}")
    print(f"  Transformer: {final_results['transformer_rmse']:.4f}")
    print(f"  Physics: {final_results['physics_rmse']:.4f}")
    print(f"Optimized Weights: {final_results['weights']}")
    
    # Save ensemble
    ensemble.save_ensemble()
    
    # Convert numpy types to regular Python types for JSON serialization
    def convert_numpy_types(obj):
        if hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):  # numpy scalar types
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Save results
    results = {
        'training_results': convert_numpy_types(training_results),
        'final_evaluation': convert_numpy_types(final_results),
        'ga_optimized_weights': convert_numpy_types(best_weights.tolist()),
        'ga_best_rmse': float(best_rmse)
    }
    
    with open("modules/soc/models/adaptive_ensemble_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("Adaptive ensemble training completed!")
    return ensemble, results


if __name__ == "__main__":
    train_adaptive_ensemble()
