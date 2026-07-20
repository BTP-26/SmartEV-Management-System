
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
from shared.train_utils import calculate_regression_metrics, set_seed
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
    """Constraint-regularised neural network for SoC estimation (ensemble sub-model). "Physics-
    informed" here means loss-level physical constraints, not a physically-calibrated model -
    see modules/soc/models/physics_informed_soc.py's identical honesty note (B2)."""
    
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
    stated plainly here, not glossed over. Reused below by the standalone L-B2 alpha-fusion
    baseline as well as by B4's adaptive-ensemble adaptation signal - this class is the
    shared CC-wrapping utility, not itself "the" L-B2 implementation (see below)."""

    def __init__(self, capacity_ah: float = CAPACITY_AH):
        self.capacity_ah = capacity_ah

    def predict(self, cycle: Dict[str, np.ndarray], window_end_indices: np.ndarray) -> np.ndarray:
        soc_init = cycle["soc"][0] / 100.0  # the one true-SoC anchor this signal uses
        full_cc = coulomb_counting_soc(cycle["curr"], cycle["t"], self.capacity_ah, soc_init=soc_init)
        return full_cc[window_end_indices]


# ---------------------------------------------------------------------------------------
# L-B2: alpha-fusion literature baseline (Carrera, Quiroz, Guevara, Acosta-Vargas, "SoC
# Estimation ... LSTM Optimized with Genetic Algorithms", Sensors 2025).
#
# Deliberately standalone - NOT an AdaptationReferenceSignal subclass. That ABC's
# predict(cycle, window_end_indices) contract only receives raw cycle data, but this
# fusion needs both the Coulomb Counting estimate AND the primary LSTM model's own
# prediction (SOC_hybrid = alpha*SOC_CC + (1-alpha)*SOC_LSTM), which the ABC has no way to
# supply. This is base paper 2's own standalone deployable model, evaluated as an
# independent literature-comparison baseline, not a reference signal for adapting the
# 3-way ensemble above. Needs no new model training: it's a deterministic post-hoc blend
# of two already-trained/already-deterministic predictors (the deployed LSTMCNNAttentionSoC
# checkpoint, and coulomb_counting_soc()).
#
# The alpha VALUES (0.3/0.5/0.7) are the paper's own dimensionless regime weights, kept as
# cited defaults - they express "how much to trust CC vs the LSTM at low/medium/high
# current," which is a modeling choice independent of any particular pack's scale. The
# CURRENT THRESHOLDS separating those regimes are NOT copied from the paper: their cutoffs
# (2A/8A) were empirically tuned on a 48-72V pack, and our pack is a full 240Ah/~424V EV
# pack - a completely different current scale, so reusing their absolute amperes would be
# meaningless. compute_alpha_fusion_thresholds() below re-derives them from our own data.
# ---------------------------------------------------------------------------------------

def compute_alpha_fusion_thresholds(X_train: np.ndarray, segment_type: np.ndarray) -> Tuple[float, float]:
    """Derive our own pack's low/high |current| thresholds for alpha_fusion_weight(), from
    the real training set's window-end current (X_train[:, -1, 1] - the same granularity
    alpha_fusion_weight() is actually evaluated at via evaluate_alpha_fusion(), i.e. one
    instantaneous current reading per window, not every timestep in the window pooled
    together).

    Method: the 33rd/66.7th percentile of |current|, restricted to `drive`/`regen`
    windows only (segment_type from B5's metadata, shared.dataset_loader.
    load_soc_split_metadata()) - deliberately excluding `charge` windows. This is a
    finding, not an assumption: charge segments are 74.4% of the training set, and
    checking their current distribution directly showed 55.8% of ALL training samples
    (pooled across every segment) sit at a single ~0.305A value from the 20th through
    74th percentile - a regulated/near-constant charge-current setpoint the BMS holds for
    most of a charge session. Pooling all segments together (an earlier version of this
    function did) makes the 33rd/67th percentile collapse onto that one plateau value,
    producing degenerate (low == high) thresholds - verified directly, not assumed. Drive
    and regen current genuinely varies across a meaningful dynamic range (drive p10=0.11A
    to p99=3.29A; regen p10=0.11A to p99=2.27A) and is the operating regime the paper's
    regime-discrimination logic is actually about - charging current is already externally
    regulated by the BMS, not something that needs CC-vs-LSTM trust to adapt across.

    Why percentiles and not a validation-set RMSE search (which is what the paper itself
    did on its own pack): a search needs held-out cycles with true SoC to optimize against,
    and this dataset only has 13-15 usable cycles total (see battery_rls_identification.py) -
    fitting two extra threshold parameters against that few cycles risks overfitting the
    thresholds themselves. A plain quantile of the (much larger) windowed current data is a
    deterministic, reproducible, non-overfit statistic of our own pack's actual current
    distribution, which is what "appropriate for our battery pack" requires at minimum, even
    if it is a simpler estimate than the paper's own tuning procedure. alpha_fusion_weight()
    is still applied to ALL segments (including charge) at inference time - only the
    threshold *derivation* excludes charge, for the reason above.
    """
    window_end_current = X_train[:, -1, 1]
    dynamic_mask = np.isin(segment_type, ["drive", "regen"])
    abs_current = np.abs(window_end_current[dynamic_mask])
    low_threshold = float(np.percentile(abs_current, 33.33))
    high_threshold = float(np.percentile(abs_current, 66.67))
    return low_threshold, high_threshold


def alpha_fusion_weight(current: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    """Piecewise dynamic fusion weight alpha(|I|), per L-B2 (see module-level note above):
    favors Coulomb Counting (alpha=0.3) at low current (minimal integration drift), equal
    trust (0.5) at moderate current, and favors the LSTM (0.7) at high current (CC less
    reliable under transients) - same directional logic as the source paper, applied at our
    own pack's thresholds (see compute_alpha_fusion_thresholds)."""
    abs_current = np.abs(current)
    alpha = np.full_like(abs_current, 0.5, dtype=np.float64)
    alpha[abs_current < low_threshold] = 0.3
    alpha[abs_current >= high_threshold] = 0.7
    return alpha


def alpha_fusion_soc(cc_pred: np.ndarray, lstm_pred: np.ndarray, current: np.ndarray,
                      low_threshold: float, high_threshold: float) -> np.ndarray:
    """SOC_hybrid(t) = alpha(t)*SOC_CC(t) + (1-alpha(t))*SOC_LSTM(t) - L-B2's literal 2-way
    fusion, blended continuously at every timestep (not just as an initial condition)."""
    alpha = alpha_fusion_weight(current, low_threshold, high_threshold)
    return alpha * cc_pred + (1 - alpha) * lstm_pred


def evaluate_alpha_fusion(lstm_cnn_model: nn.Module, cycle: Dict[str, np.ndarray],
                           low_threshold: float, high_threshold: float,
                           capacity_ah: float = CAPACITY_AH, window_size: int = 50,
                           step: int = 10, device: Optional[torch.device] = None) -> pd.DataFrame:
    """Evaluate the L-B2 alpha-fusion baseline against one real cycle's ground-truth SoC,
    alongside its two ingredients (Coulomb Counting alone, the LSTM alone) for comparison -
    reuses _windows_from_cycle() and CoulombCountingReferenceSignal (both already
    established for B4), and the already-trained lstm_cnn_model passed in (no retraining).
    Device auto-detection matches AdaptiveEnsembleSoC.__init__'s pattern (MPS/CUDA/CPU) -
    previously defaulted to plain CPU regardless of what device lstm_cnn_model actually
    lived on, which could mismatch a caller's GPU/MPS-resident model."""
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    X_windows, end_indices = _windows_from_cycle(cycle, window_size, step)

    cc_pred = CoulombCountingReferenceSignal(capacity_ah=capacity_ah).predict(cycle, end_indices)

    lstm_cnn_model = lstm_cnn_model.to(device)
    lstm_cnn_model.eval()
    with torch.no_grad():
        x = torch.tensor(X_windows, dtype=torch.float32).to(device)
        lstm_pred = lstm_cnn_model(x).cpu().numpy()

    window_end_current = cycle["curr"][end_indices]
    fused_pred = alpha_fusion_soc(cc_pred, lstm_pred, window_end_current, low_threshold, high_threshold)

    scale = load_soc_scale()
    true_pct = inverse_scale_soc(cycle["soc"][end_indices] / 100.0, scale)

    rows = []
    for name, pred_unit in [("Coulomb Counting", cc_pred),
                             ("LSTM-CNN-Attention", lstm_pred),
                             ("Alpha-Fusion (L-B2)", fused_pred)]:
        pred_pct = inverse_scale_soc(pred_unit, scale)
        metrics = calculate_regression_metrics(true_pct, pred_pct)
        rows.append({"method": name, "rmse_pct_soc": metrics["rmse"], "mae_pct_soc": metrics["mae"],
                     "n_steps": len(true_pct)})
    return pd.DataFrame(rows)


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


def train_adaptive_ensemble(seed: int = 42):
    """Main function to train adaptive ensemble. `seed` is a reproducibility override for
    multi-seed experiments (SV-1); it does not change any model/training logic."""
    print("=== Training Adaptive Ensemble SoC Model ===")
    set_seed(seed)

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
    import argparse
    parser = argparse.ArgumentParser(description="Train the adaptive SoC ensemble")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed override, for multi-seed experiments (SV-1).")
    args = parser.parse_args()
    train_adaptive_ensemble(seed=args.seed)
