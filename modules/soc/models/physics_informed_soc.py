
# Constraint-Regularised SoC Estimation with Battery Constraints
# Integrates State of Health (SoH), thermal dynamics, and electrochemical constraints.
#
# This was previously described as "Physics-Informed" (PINN-style). That framing implied
# the constraints below were physically calibrated; in fact only the electrochemical/
# voltage parameters are grounded in real data (see _load_identified_battery_params()
# and modules/soc/models/battery_rls_identification.py, roadmap B2). The SoH and thermal
# parameters are not identifiable from this dataset (no capacity-fade/aging cycles or
# direct thermal measurements are recorded) and remain literature-typical placeholders -
# labeled as such below rather than presented as calibrated.


import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.soc.models.lstm_cnn_attention_soc import LSTMCNNAttentionSoC
from shared.dataset_loader import get_dataset_loader
from shared.train_utils import set_seed

IDENTIFIED_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "battery_params_identified.json")


def _load_identified_battery_params() -> Dict[str, float]:
    """Load the RLS-identified pack parameters (see battery_rls_identification.py / B2).

    Falls back to the last values that script produced (checked into this repo as the
    defaults below) if battery_params_identified.json isn't present in a fresh checkout,
    so BatteryPhysicsParams() never silently reverts to the old 18650-cell numbers.
    """
    defaults = {
        "internal_resistance": 0.0246,  # R0, Ohm - pack-level, RLS mean across cycles
        "r1_ohm": 0.0513,
        "c1_farad": 847.8,
        "sample_interval_s": 0.099998,  # Ts, pack current-channel rate after 10x downsampling (L-B4)
        "nominal_capacity": 240.0,  # Ah - see CAPACITY_AH in preprocess_real_data.py
        "min_voltage": 363.75,
        "max_voltage": 456.25,
        "nominal_voltage": 424.22,
        "max_charge_rate": 1.0,  # C-rate, p99.9 of observed charge current / capacity
        "max_discharge_rate": 3.45,  # C-rate, p99.9 of observed discharge current / capacity
    }
    if not os.path.exists(IDENTIFIED_PARAMS_PATH):
        print(f"warning: {IDENTIFIED_PARAMS_PATH} not found; using last-known RLS "
              "results checked into this file. Run battery_rls_identification.py to regenerate.")
        return defaults

    with open(IDENTIFIED_PARAMS_PATH) as f:
        data = json.load(f)
    defaults["internal_resistance"] = data["ecm_r0_ohm"]["mean"]
    defaults["r1_ohm"] = data["ecm_r1_ohm"]["mean"]
    defaults["c1_farad"] = data["ecm_c1_farad"]["mean"]
    if "ecm_ts_s" in data:
        defaults["sample_interval_s"] = data["ecm_ts_s"]["mean"]
    defaults["nominal_capacity"] = data["provenance"]["capacity_ah"]
    defaults["min_voltage"] = data["voltage"]["min_v"]
    defaults["max_voltage"] = data["voltage"]["max_v"]
    defaults["nominal_voltage"] = data["voltage"]["mean_v"]
    if data["current"].get("max_charge_c_rate"):
        defaults["max_charge_rate"] = data["current"]["max_charge_c_rate"]
    if data["current"].get("max_discharge_c_rate"):
        defaults["max_discharge_rate"] = data["current"]["max_discharge_c_rate"]
    return defaults


_IDENTIFIED = _load_identified_battery_params()


@dataclass
class BatteryPhysicsParams:
    """Battery constraint parameters.

    Electrochemical/voltage fields are identified from real pack-level data (Mendeley
    "Real-world electric vehicle data driving and charging" - see
    battery_rls_identification.py). SoH/thermal fields are NOT identifiable from this
    dataset (no aging cycles or direct thermal measurements) and are literature-typical
    placeholders, not calibrated values - flagged individually below.
    """
    # State of Health parameters (NOT identifiable from this dataset - placeholders)
    soh_nominal: float = 1.0  # Nominal state of health
    soh_degradation_rate: float = 0.0001  # Per cycle degradation
    soh_temp_factor: float = 0.02  # Temperature effect on degradation

    # Thermal parameters (NOT identifiable from this dataset - placeholders)
    nominal_temp: float = 25.0  # Nominal temperature (°C)
    temp_coefficient: float = -0.003  # Voltage temperature coefficient
    thermal_resistance: float = 0.1  # Thermal resistance (°C/W)
    heat_capacity: float = 1000  # Heat capacity (J/K)

    # Electrochemical parameters - identified from real pack data (see module docstring)
    nominal_capacity: float = _IDENTIFIED["nominal_capacity"]  # Ah (pack-level)
    internal_resistance: float = _IDENTIFIED["internal_resistance"]  # R0, Ohm (pack-level)
    max_charge_rate: float = _IDENTIFIED["max_charge_rate"]  # C-rate
    max_discharge_rate: float = _IDENTIFIED["max_discharge_rate"]  # C-rate

    # First-order Thevenin RC branch (R1, C1), identified via RLS - see
    # battery_rls_identification.py for the ARX<->ECM derivation and per-cycle fits.
    r1_ohm: float = _IDENTIFIED["r1_ohm"]
    c1_farad: float = _IDENTIFIED["c1_farad"]
    # Sample interval for the RC branch's ZOH discretization (L-B4) - the pack
    # current-channel rate after preprocessing's 10x downsampling, not a per-window
    # value (the windowed dataset doesn't carry timestamps), see battery_rls_identification.py.
    sample_interval_s: float = _IDENTIFIED["sample_interval_s"]

    # Voltage limits - pack-level, observed extrema/mean (NOT single-cell values)
    min_voltage: float = _IDENTIFIED["min_voltage"]  # V
    max_voltage: float = _IDENTIFIED["max_voltage"]  # V
    nominal_voltage: float = _IDENTIFIED["nominal_voltage"]  # V

    def to_dict(self) -> Dict:
        return {
            'soh_nominal': self.soh_nominal,
            'soh_degradation_rate': self.soh_degradation_rate,
            'soh_temp_factor': self.soh_temp_factor,
            'nominal_temp': self.nominal_temp,
            'temp_coefficient': self.temp_coefficient,
            'thermal_resistance': self.thermal_resistance,
            'heat_capacity': self.heat_capacity,
            'nominal_capacity': self.nominal_capacity,
            'internal_resistance': self.internal_resistance,
            'max_charge_rate': self.max_charge_rate,
            'max_discharge_rate': self.max_discharge_rate,
            'r1_ohm': self.r1_ohm,
            'c1_farad': self.c1_farad,
            'sample_interval_s': self.sample_interval_s,
            'min_voltage': self.min_voltage,
            'max_voltage': self.max_voltage,
            'nominal_voltage': self.nominal_voltage
        }


class BatteryPhysicsConstraints:
    """Implements battery physics constraints for SoC estimation."""
    
    def __init__(self, params: BatteryPhysicsParams):
        self.params = params
        self.current_soh = params.soh_nominal
        self.temp_history = []
        self.cycle_count = 0
    
    def update_soh(self, temperature: float, cycle_increment: float = 1.0):
        """Update State of Health based on temperature and cycles."""
        # Temperature effect on degradation
        temp_factor = 1.0 + self.params.soh_temp_factor * abs(temperature - self.params.nominal_temp)
        
        # Update SoH
        degradation = self.params.soh_degradation_rate * cycle_increment * temp_factor
        self.current_soh = max(0.0, self.current_soh - degradation)
        self.cycle_count += cycle_increment
        
        return self.current_soh
    
    def calculate_capacity_adjustment(self, temperature: float) -> float:
        """Calculate capacity adjustment based on temperature and SoH."""
        # Temperature effect on capacity
        temp_effect = 1.0 - 0.01 * abs(temperature - self.params.nominal_temp)
        
        # SoH effect on capacity
        soh_effect = self.current_soh
        
        return temp_effect * soh_effect
    
    def calculate_voltage_adjustment(self, soc: float, temperature: float, current: float) -> float:
        """Calculate voltage adjustment based on physics."""
        # Base OCV (Open Circuit Voltage) curve approximation
        ocv = self.params.min_voltage + (self.params.max_voltage - self.params.min_voltage) * (
            0.5 * (1 + np.tanh(6 * (soc - 0.5)))
        )
        
        # Temperature effect
        temp_adjustment = self.params.temp_coefficient * (temperature - self.params.nominal_temp)
        
        # IR drop (Internal Resistance)
        ir_drop = self.params.internal_resistance * current
        
        # Adjusted voltage
        adjusted_voltage = ocv + temp_adjustment - ir_drop
        
        return np.clip(adjusted_voltage, self.params.min_voltage, self.params.max_voltage)
    
    def calculate_power_limits(self, soc: float, temperature: float) -> Tuple[float, float]:
        """Calculate charge/discharge power limits."""
        # Capacity adjustment
        capacity_factor = self.calculate_capacity_adjustment(temperature)
        
        # SoC-based limits
        if soc < 0.1:  # Low SoC
            charge_limit = self.params.max_charge_rate * capacity_factor * 0.5
            discharge_limit = self.params.max_discharge_rate * capacity_factor
        elif soc > 0.9:  # High SoC
            charge_limit = self.params.max_charge_rate * capacity_factor * 0.3
            discharge_limit = self.params.max_discharge_rate * capacity_factor * 0.8
        else:  # Normal SoC range
            charge_limit = self.params.max_charge_rate * capacity_factor
            discharge_limit = self.params.max_discharge_rate * capacity_factor
        
        # Temperature derating
        temp_derating = 1.0 - 0.02 * abs(temperature - self.params.nominal_temp)
        temp_derating = max(0.5, temp_derating)  # Minimum 50% capacity
        
        return charge_limit * temp_derating, discharge_limit * temp_derating
    
    def validate_soc_range(self, soc: float) -> float:
        """Validate and clamp SoC to physically valid range."""
        return np.clip(soc, 0.0, 1.0)


class PhysicsInformedSoCModel(nn.Module):
    """Constraint-regularised neural network for SoC estimation."""
    
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=3, dropout=0.2, 
                 physics_params: Optional[BatteryPhysicsParams] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Physics constraints
        self.physics = physics_params or BatteryPhysicsParams()

        # L-B4: RC-branch (R1, C1) ZOH decay constant, fixed at construction time since
        # R1/C1/Ts are RLS-identified dataset-level constants, not learned - see
        # battery_rls_identification.py's header for the continuous ECM and this
        # discretization (alpha = exp(-Ts/(R1*C1))). Not a nn.Parameter: this must not
        # be gradient-updated, it is a physical constant, unlike soh_embedding/temp_embedding.
        self._rc_alpha = float(
            np.exp(-self.physics.sample_interval_s / (self.physics.r1_ohm * self.physics.c1_farad))
        )

        # Neural network layers
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
        
        # Constraint-regularised layers
        self.physics_processor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Final SoC estimator with constraints
        self.soc_estimator = nn.Sequential(
            nn.Linear(32 + 12, 16),  # +12 for physics features (L-B4 added ir_drop_mean, vc_relaxation)
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Ensure SoC in [0, 1]
        )
        
        # Learnable physics parameters (fine-tuning)
        self.soh_embedding = nn.Parameter(torch.tensor(1.0))  # Current SoH
        self.temp_embedding = nn.Parameter(torch.tensor(25.0))  # Current temperature bias
    
    def _extract_physics_features(self, x):
        """Extract physics-based features from input."""
        # x: (batch, seq_len, input_dim) -> [voltage, current, temperature]
        voltage = x[:, :, 0]
        current = x[:, :, 1]
        temp = x[:, :, 2]
        
        # Statistical features
        voltage_mean = torch.mean(voltage, dim=1)
        voltage_std = torch.std(voltage, dim=1)
        current_mean = torch.mean(current, dim=1)
        current_std = torch.std(current, dim=1)
        temp_mean = torch.mean(temp, dim=1)
        temp_std = torch.std(temp, dim=1)
        
        # Physics-based features
        # Power estimation
        power = voltage_mean * current_mean
        
        # Energy estimation (simplified)
        energy = voltage_mean * current_mean  # Simplified energy estimation
        
        # Temperature deviation
        temp_deviation = temp_mean - self.temp_embedding

        # L-B4: R0 IR-drop and R1/C1 RC-branch relaxation voltage, both computed from
        # `current` alone (a genuine input, not a predicted quantity) so there's no
        # circularity with the SoC this model is estimating. Vc is reset to 0 at the
        # start of each window (no continuous state across independent, shuffled
        # training windows) - since tau (~27s mean) is longer than a 50-step/5s window
        # at the identified Ts, this under-estimates any relaxation carried in from
        # before the window started; treat vc_relaxation as an approximation, not a
        # full simulation of the true RC state.
        ir_drop_mean = self.physics.internal_resistance * current_mean
        vc = torch.zeros(current.shape[0], device=x.device, dtype=current.dtype)
        for t in range(current.shape[1]):
            vc = self._rc_alpha * vc + self.physics.r1_ohm * (1 - self._rc_alpha) * current[:, t]
        vc_relaxation = vc

        # SoH-adjusted capacity factor
        temp_factor = 1.0 - 0.01 * torch.abs(temp_mean - self.physics.nominal_temp)
        capacity_factor = self.soh_embedding * temp_factor

        return torch.stack([
            voltage_mean, voltage_std, current_mean, current_std,
            temp_mean, temp_std, power, energy, temp_deviation, capacity_factor,
            ir_drop_mean, vc_relaxation
        ], dim=1)
    
    def _apply_physics_constraints(self, soc_pred, physics_features):
        """Apply physics constraints to SoC prediction."""
        # Extract relevant physics features
        temp = physics_features[:, 4]  # Temperature mean
        voltage = physics_features[:, 0]  # Voltage mean
        current = physics_features[:, 2]  # Current mean
        
        # Voltage-based SoC validation
        # Approximate SoC-voltage relationship
        voltage_soc_lower = self.physics.min_voltage + 0.1 * (self.physics.max_voltage - self.physics.min_voltage)
        voltage_soc_upper = self.physics.max_voltage - 0.1 * (self.physics.max_voltage - self.physics.min_voltage)
        
        # Adjust SoC based on voltage constraints
        voltage_factor = torch.sigmoid((voltage - voltage_soc_lower) / (voltage_soc_upper - voltage_soc_lower))
        
        # Temperature-based adjustment
        temp_factor = torch.sigmoid(-0.1 * torch.abs(temp - self.physics.nominal_temp))
        
        # Apply constraints
        constrained_soc = soc_pred * voltage_factor * temp_factor
        
        # Ensure physical bounds
        constrained_soc = torch.clamp(constrained_soc, 0.0, 1.0)
        
        return constrained_soc
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Extract neural features
        neural_features = []
        for t in range(seq_len):
            timestep_features = self.feature_extractor(x[:, t, :])
            neural_features.append(timestep_features)
        
        # Aggregate neural features
        aggregated_neural = torch.mean(torch.stack(neural_features, dim=1), dim=1)
        
        # Process through constraint-regularised layers
        physics_neural = self.physics_processor(aggregated_neural)
        
        # Extract physics features
        physics_features = self._extract_physics_features(x)
        
        # Combine neural and physics features
        combined_features = torch.cat([physics_neural, physics_features], dim=1)
        
        # Initial SoC prediction
        soc_pred = self.soc_estimator(combined_features).squeeze(-1)
        
        # Apply physics constraints
        soc_constrained = self._apply_physics_constraints(soc_pred, physics_features)
        
        return soc_constrained
    
    def update_physics_state(self, temperature: float, cycle_increment: float = 1.0):
        """Update physics state (SoH, etc.)."""
        new_soh = self.physics.update_soh(temperature, cycle_increment)
        self.soh_embedding.data = torch.tensor(new_soh, dtype=torch.float32)
        return new_soh


class EnhancedPhysicsInformedSoC(nn.Module):
    """Enhanced constraint-regularised model with adaptive constraints."""

    def __init__(self, input_dim=3, hidden_dim=128, num_layers=3, dropout=0.2,
                 seq_len=50, physics_params: Optional[BatteryPhysicsParams] = None):
        super().__init__()

        self.base_model = PhysicsInformedSoCModel(
            input_dim, hidden_dim, num_layers, dropout, physics_params
        )

        # Adaptive constraint learning
        # `seq_len` must match the window size of whatever data this model consumes
        # (config/dataset_config.yaml: soc_window_size=50). This used to be hardcoded to
        # 25 to match an orphaned, truncated dataset convention that no longer exists.
        self.constraint_learner = nn.Sequential(
            nn.Linear(input_dim * seq_len, 64),  # Flatten sequence
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Learn constraint weights
            nn.Softmax(dim=1)
        )
        
        # Residual connection for constraint refinement
        self.constraint_refiner = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Base constraint-regularised prediction
        base_pred = self.base_model(x)
        
        # Learn constraint weights from data
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        constraint_weights = self.constraint_learner(x_flat)
        
        # Apply learned constraints
        constraint_factor = constraint_weights[:, 0]  # Use first weight as constraint factor
        
        # Refine prediction with learned constraints
        refined_pred = base_pred * constraint_factor + self.constraint_refiner(base_pred.unsqueeze(-1)).squeeze(-1)
        
        # Final bounds check
        final_pred = torch.clamp(refined_pred, 0.0, 1.0)
        
        return final_pred


def train_physics_informed_model(X_train, y_train, X_val, y_val, 
                               physics_params: Optional[BatteryPhysicsParams] = None,
                               epochs=10, batch_size=64, lr=0.001, device='cpu'):
    """Train constraint-regularised SoC model."""
    print("Training Constraint-Regularised SoC Model...")
    
    # Create model
    model = EnhancedPhysicsInformedSoC(
        input_dim=3, hidden_dim=128, num_layers=3, dropout=0.2,
        physics_params=physics_params
    )
    model.to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training loop
    best_val_rmse = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                
                val_predictions.extend(pred.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # Calculate RMSE
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        val_rmse = np.sqrt(np.mean((val_targets - val_predictions) ** 2))
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val RMSE: {val_rmse:.4f}")
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), "modules/soc/models/physics_informed_soc.pth")
    
    print(f"Best Validation RMSE: {best_val_rmse:.4f}")
    return model, best_val_rmse


def test_physics_constraints():
    """Test physics constraints implementation."""
    print("=== Testing Physics Constraints ===")
    
    # Create physics parameters
    params = BatteryPhysicsParams()
    constraints = BatteryPhysicsConstraints(params)
    
    # Test SoH update
    initial_soh = constraints.current_soh
    new_soh = constraints.update_soh(35.0, 10)  # High temperature, 10 cycles
    print(f"SoH degradation: {initial_soh:.4f} -> {new_soh:.4f}")
    
    # Test capacity adjustment
    capacity_factor = constraints.calculate_capacity_adjustment(35.0)
    print(f"Capacity factor at 35°C: {capacity_factor:.4f}")
    
    # Test voltage adjustment
    voltage = constraints.calculate_voltage_adjustment(0.5, 35.0, 10.0)
    print(f"Adjusted voltage (SoC=0.5, 35°C, 10A): {voltage:.3f}V")
    
    # Test power limits
    charge_limit, discharge_limit = constraints.calculate_power_limits(0.5, 35.0)
    print(f"Power limits at 35°C, SoC=0.5: Charge={charge_limit:.2f}C, Discharge={discharge_limit:.2f}C")
    
    print("Physics constraints test completed!")


def create_physics_informed_model(seed: int = 42):
    """Main function to create and test the constraint-regularised SoC model. `seed` is a
    reproducibility override for multi-seed experiments (SV-1); it does not change any
    model/training logic."""
    print("=== Creating Constraint-Regularised SoC Model ===")
    set_seed(seed)

    # Test physics constraints
    test_physics_constraints()
    
    # Load the same real, [0,1]-scaled dataset the deployed model uses, instead of the
    # orphaned *_soc.npy convention that no preprocessing script in the repo produced.
    dataset_loader = get_dataset_loader()
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_loader.load_soc_dataset()

    # Use a smaller subset for faster training; keep the full window length (50) so the
    # model matches the deployed dataset shape instead of a truncated 25-step convention.
    sample_size = min(20000, len(X_train))
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]

    print(f"Training data shape: {X_train.shape}")
    
    # Create physics parameters
    physics_params = BatteryPhysicsParams()
    
    # Train model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, best_rmse = train_physics_informed_model(
        X_train, y_train, X_val, y_val,
        physics_params=physics_params,
        epochs=5, batch_size=128, lr=0.001, device=device
    )
    
    # Save physics parameters
    with open("modules/soc/models/physics_params.json", "w") as f:
        json.dump(physics_params.to_dict(), f, indent=4)
    
    print(f"Constraint-regularised model trained with RMSE: {best_rmse:.4f}")
    print("Physics parameters saved!")

    return model, physics_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train the constraint-regularised SoC model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed override, for multi-seed experiments (SV-1).")
    args = parser.parse_args()
    model, params = create_physics_informed_model(seed=args.seed)
    print("Constraint-regularised SoC model created successfully!")
