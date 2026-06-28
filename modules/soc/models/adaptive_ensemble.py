
# Combines multiple model types with GA-optimized ensemble weights


import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
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
            
            # Calculate RMSE for each model and ensemble
            def rmse(y_true, y_pred):
                return np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            lstm_rmse = rmse(y_test, lstm_pred)
            transformer_rmse = rmse(y_test, transformer_pred)
            physics_rmse = rmse(y_test, physics_pred)
            ensemble_rmse = rmse(y_test, ensemble_pred)
            
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


def train_adaptive_ensemble():
    """Main function to train adaptive ensemble."""
    print("=== Training Adaptive Ensemble SoC Model ===")
    
    # Get project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    
    # Load data
    DATA = os.path.join(project_root, "modules/soc/data")
    X_train = np.load(os.path.join(DATA, "X_train_soc.npy"))
    y_train = np.load(os.path.join(DATA, "y_train_soc.npy"))
    X_val = np.load(os.path.join(DATA, "X_val_soc.npy"))
    y_val = np.load(os.path.join(DATA, "y_val_soc.npy"))
    
    # Use smaller dataset for faster training
    sample_size = 30000
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]
    X_train = X_train[:, :25, :]  # Reduce sequence length
    X_val = X_val[:, :25, :]
    
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
