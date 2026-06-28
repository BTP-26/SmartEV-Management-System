
# Physics-Informed SoC Estimation with Battery Constraints
# Integrates State of Health (SoH), thermal dynamics, and electrochemical constraints


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


@dataclass
class BatteryPhysicsParams:
    """Battery physics parameters for constraints."""
    # State of Health parameters
    soh_nominal: float = 1.0  # Nominal state of health
    soh_degradation_rate: float = 0.0001  # Per cycle degradation
    soh_temp_factor: float = 0.02  # Temperature effect on degradation
    
    # Thermal parameters
    nominal_temp: float = 25.0  # Nominal temperature (°C)
    temp_coefficient: float = -0.003  # Voltage temperature coefficient
    thermal_resistance: float = 0.1  # Thermal resistance (°C/W)
    heat_capacity: float = 1000  # Heat capacity (J/K)
    
    # Electrochemical parameters
    nominal_capacity: float = 100.0  # Ah
    internal_resistance: float = 0.05  # Ohms
    max_charge_rate: float = 2.0  # C-rate
    max_discharge_rate: float = 3.0  # C-rate
    
    # Voltage limits
    min_voltage: float = 2.5  # V
    max_voltage: float = 4.2  # V
    nominal_voltage: float = 3.7  # V
    
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
    """Physics-informed neural network for SoC estimation with constraints."""
    
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=3, dropout=0.2, 
                 physics_params: Optional[BatteryPhysicsParams] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Physics constraints
        self.physics = physics_params or BatteryPhysicsParams()
        
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
        
        # Physics-informed layers
        self.physics_processor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Final SoC estimator with constraints
        self.soc_estimator = nn.Sequential(
            nn.Linear(32 + 10, 16),  # +10 for physics features
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
        
        # Voltage deviation from nominal
        voltage_deviation = voltage_mean - self.physics.nominal_voltage
        
        # Current rate (simplified)
        current_rate = torch.std(current, dim=1) if current.dim() > 1 else torch.zeros_like(current_mean)
        
        # SoH-adjusted capacity factor
        temp_factor = 1.0 - 0.01 * torch.abs(temp_mean - self.physics.nominal_temp)
        capacity_factor = self.soh_embedding * temp_factor
        
        return torch.stack([
            voltage_mean, voltage_std, current_mean, current_std,
            temp_mean, temp_std, power, energy, temp_deviation, capacity_factor
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
        
        # Process through physics-informed layers
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
    """Enhanced physics-informed model with adaptive constraints."""
    
    def __init__(self, input_dim=3, hidden_dim=128, num_layers=3, dropout=0.2,
                 physics_params: Optional[BatteryPhysicsParams] = None):
        super().__init__()
        
        self.base_model = PhysicsInformedSoCModel(
            input_dim, hidden_dim, num_layers, dropout, physics_params
        )
        
        # Adaptive constraint learning
        self.constraint_learner = nn.Sequential(
            nn.Linear(input_dim * 25, 64),  # Flatten sequence (assuming 25 timesteps)
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
        # Base physics-informed prediction
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
    """Train physics-informed SoC model."""
    print("Training Physics-Informed SoC Model...")
    
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


def create_physics_informed_model():
    """Main function to create and test physics-informed model."""
    print("=== Creating Physics-Informed SoC Model ===")
    
    # Test physics constraints
    test_physics_constraints()
    
    # Load data
    DATA = "modules/soc/data"
    X_train = np.load(f"{DATA}/X_train_soc.npy")
    y_train = np.load(f"{DATA}/y_train_soc.npy")
    X_val = np.load(f"{DATA}/X_val_soc.npy")
    y_val = np.load(f"{DATA}/y_val_soc.npy")
    
    # Use smaller dataset for faster training
    sample_size = 20000
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]
    X_train = X_train[:, :25, :]
    X_val = X_val[:, :25, :]
    
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
    
    print(f"Physics-informed model trained with RMSE: {best_rmse:.4f}")
    print("Physics parameters saved!")
    
    return model, physics_params


if __name__ == "__main__":
    model, params = create_physics_informed_model()
    print("Physics-informed SoC model created successfully!")
