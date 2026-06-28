import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DrivingStyle(Enum):
    """Driver behavior classification."""
    ECO = "eco"
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"


@dataclass
class DriverProfile:
    driver_id: str
    driving_style: DrivingStyle
    avg_braking_intensity: float
    braking_frequency: float
    speed_variance: float
    acceleration_patterns: List[float]
    preferred_regen_level: float
    battery_usage_efficiency: float
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['driving_style'] = self.driving_style.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict):
        data['driving_style'] = DrivingStyle(data['driving_style'])
        return cls(**data)


@dataclass
class CognitiveState:
    """Cognitive state of the energy management system."""
    current_driver_profile: Optional[DriverProfile]
    recent_braking_events: List[Dict]
    energy_recovery_history: List[float]
    soc_prediction_confidence: float
    adaptation_level: float
    last_update_time: float


class DriverBehaviorAnalyzer:
    """Analyzes driver behavior patterns."""
    
    def __init__(self):
        self.braking_history = []
        self.speed_history = []
        self.acceleration_history = []
        
    def analyze_driving_pattern(self, driving_window: np.ndarray, 
                               braking_class: int, intensity: float) -> Dict:
        """Analyze driving pattern from window."""
        # Extract features
        speed = driving_window[:, 0]
        acceleration = driving_window[:, 1]
        brake_pedal = driving_window[:, 2]
        
        # Calculate metrics
        avg_speed = np.mean(speed)
        speed_variance = np.var(speed)
        avg_acceleration = np.mean(acceleration)
        accel_variance = np.var(acceleration)
        
        # Braking patterns
        braking_frequency = np.sum(brake_pedal > 0.1) / len(brake_pedal)
        max_brake_force = np.max(brake_pedal)
        brake_smoothness = 1.0 - np.std(np.diff(brake_pedal[brake_pedal > 0.1])) if np.any(brake_pedal > 0.1) else 1.0
        
        return {
            'avg_speed': avg_speed,
            'speed_variance': speed_variance,
            'avg_acceleration': avg_acceleration,
            'accel_variance': accel_variance,
            'braking_frequency': braking_frequency,
            'max_brake_force': max_brake_force,
            'brake_smoothness': brake_smoothness,
            'intensity': intensity,
            'class': braking_class
        }
    
    def classify_driving_style(self, patterns: List[Dict]) -> DrivingStyle:
        """Classify driving style based on patterns."""
        if not patterns:
            return DrivingStyle.NORMAL
        
        # Aggregate metrics
        avg_intensity = np.mean([p['intensity'] for p in patterns])
        avg_frequency = np.mean([p['braking_frequency'] for p in patterns])
        avg_speed_var = np.mean([p['speed_variance'] for p in patterns])
        avg_smoothness = np.mean([p['brake_smoothness'] for p in patterns])
        
        # Classification logic
        if avg_intensity < 0.3 and avg_frequency < 0.2 and avg_smoothness > 0.8:
            return DrivingStyle.ECO
        elif avg_intensity > 0.7 and avg_frequency > 0.5:
            return DrivingStyle.AGGRESSIVE
        elif avg_intensity < 0.2 and avg_frequency < 0.1:
            return DrivingStyle.CONSERVATIVE
        else:
            return DrivingStyle.NORMAL
    
    def create_driver_profile(self, driver_id: str, patterns: List[Dict]) -> DriverProfile:
        """Create driver profile from analyzed patterns."""
        driving_style = self.classify_driving_style(patterns)
        
        avg_braking_intensity = np.mean([p['intensity'] for p in patterns])
        braking_frequency = np.mean([p['braking_frequency'] for p in patterns])
        speed_variance = np.mean([p['speed_variance'] for p in patterns])
        acceleration_patterns = [p['avg_acceleration'] for p in patterns]
        
        # Estimate preferred regen level based on driving style
        regen_preferences = {
            DrivingStyle.ECO: 0.8,
            DrivingStyle.NORMAL: 0.6,
            DrivingStyle.AGGRESSIVE: 0.4,
            DrivingStyle.CONSERVATIVE: 0.3
        }
        preferred_regen_level = regen_preferences[driving_style]
        
        # Estimate battery usage efficiency
        efficiency_scores = {
            DrivingStyle.ECO: 0.9,
            DrivingStyle.NORMAL: 0.7,
            DrivingStyle.AGGRESSIVE: 0.5,
            DrivingStyle.CONSERVATIVE: 0.8
        }
        battery_usage_efficiency = efficiency_scores[driving_style]
        
        return DriverProfile(
            driver_id=driver_id,
            driving_style=driving_style,
            avg_braking_intensity=avg_braking_intensity,
            braking_frequency=braking_frequency,
            speed_variance=speed_variance,
            acceleration_patterns=acceleration_patterns,
            preferred_regen_level=preferred_regen_level,
            battery_usage_efficiency=battery_usage_efficiency
        )


class PersonalizedSoCPredictor:
    """Personalized SoC prediction based on driver profile."""
    
    def __init__(self):
        self.profile_models = {}  # Store models for different driver profiles
        self.prediction_history = []
        
    def predict_soc_adjustment(self, base_soc: float, driver_profile: DriverProfile,
                             current_conditions: Dict) -> Tuple[float, float]:
        """Predict SoC adjustment based on driver profile and conditions."""
        # Base adjustment factors
        style_factor = self._get_style_factor(driver_profile.driving_style)
        efficiency_factor = driver_profile.battery_usage_efficiency
        
        # Environmental factors
        temperature_factor = current_conditions.get('temperature_factor', 1.0)
        traffic_factor = current_conditions.get('traffic_factor', 1.0)
        
        # Calculate personalized adjustment
        adjustment = base_soc * style_factor * efficiency_factor * temperature_factor * traffic_factor
        
        # Calculate confidence based on profile maturity
        confidence = min(1.0, len(driver_profile.acceleration_patterns) / 50)  # More data = higher confidence
        
        return adjustment, confidence
    
    def _get_style_factor(self, style: DrivingStyle) -> float:
        """Get SoC adjustment factor for driving style."""
        factors = {
            DrivingStyle.ECO: 0.95,      # More conservative SoC usage
            DrivingStyle.NORMAL: 1.0,    # Standard usage
            DrivingStyle.AGGRESSIVE: 1.1, # Higher consumption
            DrivingStyle.CONSERVATIVE: 0.9  # Very conservative
        }
        return factors[style]


class AdaptiveEnergyRecovery:
    """Adaptive energy recovery strategies."""
    
    def __init__(self):
        self.recovery_strategies = {
            DrivingStyle.ECO: self._eco_recovery,
            DrivingStyle.NORMAL: self._normal_recovery,
            DrivingStyle.AGGRESSIVE: self._aggressive_recovery,
            DrivingStyle.CONSERVATIVE: self._conservative_recovery
        }
        self.recovery_performance = []
        
    def calculate_adaptive_recovery(self, braking_intensity: float, current_soc: float,
                                  driver_profile: DriverProfile, vehicle_state: Dict) -> Dict:
        """Calculate adaptive energy recovery based on driver profile."""
        strategy = self.recovery_strategies[driver_profile.driving_style]
        
        # Get base recovery parameters
        base_efficiency = vehicle_state.get('base_regen_efficiency', 0.65)
        battery_temperature = vehicle_state.get('battery_temp', 25.0)
        motor_temperature = vehicle_state.get('motor_temp', 30.0)
        
        # Apply strategy-specific recovery
        recovery_params = strategy(braking_intensity, current_soc, driver_profile)
        
        # Temperature compensation
        temp_factor = self._calculate_temperature_factor(battery_temperature, motor_temperature)
        
        # Calculate final recovery
        adjusted_efficiency = base_efficiency * recovery_params['efficiency_modifier'] * temp_factor
        energy_recovered = braking_intensity * adjusted_efficiency * recovery_params['intensity_factor']
        
        # Apply safety limits
        max_recovery = self._calculate_max_recovery(current_soc, driver_profile)
        energy_recovered = min(energy_recovered, max_recovery)
        
        return {
            'energy_recovered': energy_recovered,
            'efficiency': adjusted_efficiency,
            'strategy': recovery_params['strategy_name'],
            'temperature_factor': temp_factor,
            'safety_applied': energy_recovered >= max_recovery
        }
    
    def _eco_recovery(self, intensity: float, soc: float, profile: DriverProfile) -> Dict:
        """Eco driving recovery strategy - maximizes regeneration."""
        return {
            'strategy_name': 'eco_max_regen',
            'efficiency_modifier': 1.1,  # Boost efficiency for eco drivers
            'intensity_factor': 1.05      # Slightly more aggressive recovery
        }
    
    def _normal_recovery(self, intensity: float, soc: float, profile: DriverProfile) -> Dict:
        """Normal driving recovery strategy."""
        return {
            'strategy_name': 'normal_balanced',
            'efficiency_modifier': 1.0,
            'intensity_factor': 1.0
        }
    
    def _aggressive_recovery(self, intensity: float, soc: float, profile: DriverProfile) -> Dict:
        """Aggressive driving recovery strategy - focused on performance."""
        return {
            'strategy_name': 'aggressive_performance',
            'efficiency_modifier': 0.9,   # Slightly reduced efficiency
            'intensity_factor': 0.85     # Less aggressive recovery to maintain performance
        }
    
    def _conservative_recovery(self, intensity: float, soc: float, profile: DriverProfile) -> Dict:
        """Conservative driving recovery strategy - battery protection focused."""
        return {
            'strategy_name': 'conservative_protection',
            'efficiency_modifier': 0.95,  # Gentle on battery
            'intensity_factor': 0.9       # Moderate recovery
        }
    
    def _calculate_temperature_factor(self, battery_temp: float, motor_temp: float) -> float:
        """Calculate temperature-based efficiency factor."""
        # Optimal temperature range: 20-35°C
        if 20 <= battery_temp <= 35 and 20 <= motor_temp <= 45:
            return 1.0
        elif battery_temp < 10 or battery_temp > 45:
            return 0.7  # Significant derating
        else:
            return 0.85  # Moderate derating
    
    def _calculate_max_recovery(self, soc: float, profile: DriverProfile) -> float:
        """Calculate maximum safe recovery based on SoC and profile."""
        # Base limits
        if soc > 0.95:
            max_recovery = 0.1  # Very limited when almost full
        elif soc > 0.85:
            max_recovery = 0.3  # Limited when high
        elif soc < 0.1:
            max_recovery = 0.8  # Maximum when low
        else:
            max_recovery = 0.6  # Normal range
        
        # Profile adjustment
        profile_adjustments = {
            DrivingStyle.ECO: 1.1,        # Eco drivers can handle more
            DrivingStyle.NORMAL: 1.0,      # Standard
            DrivingStyle.AGGRESSIVE: 0.9,  # Aggressive drivers need more headroom
            DrivingStyle.CONSERVATIVE: 0.85 # Conservative drivers prefer safety
        }
        
        return max_recovery * profile_adjustments[profile.driving_style]


class CognitiveEnergyManager:
    """Main cognitive energy management system."""
    
    def __init__(self, save_path: str = "shared/cognitive_state.json"):
        self.save_path = save_path
        self.behavior_analyzer = DriverBehaviorAnalyzer()
        self.soc_predictor = PersonalizedSoCPredictor()
        self.recovery_manager = AdaptiveEnergyRecovery()
        
        # Load or initialize cognitive state
        self.cognitive_state = self._load_cognitive_state()
        
        # Driver profiles database
        self.driver_profiles = self._load_driver_profiles()
    
    def process_driving_event(self, driver_id: str, driving_window: np.ndarray,
                            braking_class: int, intensity: float, current_soc: float,
                            vehicle_state: Dict) -> Dict:
        """Process a driving event and update cognitive state."""
        # Analyze driving pattern
        pattern = self.behavior_analyzer.analyze_driving_pattern(
            driving_window, braking_class, intensity
        )
        
        # Update or create driver profile
        driver_profile = self._update_driver_profile(driver_id, pattern)
        
        # Calculate personalized SoC prediction
        current_conditions = {
            'temperature_factor': self._get_temperature_factor(vehicle_state),
            'traffic_factor': self._get_traffic_factor(vehicle_state)
        }
        
        soc_adjustment, prediction_confidence = self.soc_predictor.predict_soc_adjustment(
            current_soc, driver_profile, current_conditions
        )
        
        # Calculate adaptive energy recovery
        recovery_result = self.recovery_manager.calculate_adaptive_recovery(
            intensity, current_soc, driver_profile, vehicle_state
        )
        
        # Update cognitive state
        self._update_cognitive_state(driver_profile, pattern, recovery_result, prediction_confidence)
        
        # Generate cognitive recommendations
        recommendations = self._generate_recommendations(driver_profile, recovery_result, current_soc)
        
        return {
            'driver_profile': driver_profile.to_dict(),
            'soc_adjustment': soc_adjustment,
            'prediction_confidence': prediction_confidence,
            'energy_recovery': recovery_result,
            'recommendations': recommendations,
            'cognitive_insights': self._generate_insights(driver_profile, pattern)
        }
    
    def _update_driver_profile(self, driver_id: str, pattern: Dict) -> DriverProfile:
        """Update or create driver profile."""
        if driver_id not in self.driver_profiles:
            # Create new profile
            self.driver_profiles[driver_id] = self.behavior_analyzer.create_driver_profile(
                driver_id, [pattern]
            )
        else:
            # Update existing profile
            existing_profile = self.driver_profiles[driver_id]
            
            # Add new pattern to history (keep last 100 patterns)
            if not hasattr(existing_profile, 'pattern_history'):
                existing_profile.pattern_history = []
            
            existing_profile.pattern_history.append(pattern)
            if len(existing_profile.pattern_history) > 100:
                existing_profile.pattern_history.pop(0)
            
            # Recreate profile with updated patterns
            self.driver_profiles[driver_id] = self.behavior_analyzer.create_driver_profile(
                driver_id, existing_profile.pattern_history
            )
        
        return self.driver_profiles[driver_id]
    
    def _update_cognitive_state(self, driver_profile: DriverProfile, pattern: Dict,
                              recovery_result: Dict, confidence: float):
        """Update cognitive state."""
        self.cognitive_state.current_driver_profile = driver_profile
        self.cognitive_state.recent_braking_events.append(pattern)
        self.cognitive_state.energy_recovery_history.append(recovery_result['energy_recovered'])
        self.cognitive_state.soc_prediction_confidence = confidence
        self.cognitive_state.last_update_time = time.time()
        
        # Keep history manageable
        if len(self.cognitive_state.recent_braking_events) > 50:
            self.cognitive_state.recent_braking_events.pop(0)
        if len(self.cognitive_state.energy_recovery_history) > 100:
            self.cognitive_state.energy_recovery_history.pop(0)
        
        # Calculate adaptation level
        self.cognitive_state.adaptation_level = min(1.0, len(self.cognitive_state.recent_braking_events) / 20)
        
        # Save state
        self._save_cognitive_state()
    
    def _generate_recommendations(self, profile: DriverProfile, recovery: Dict, soc: float) -> List[str]:
        """Generate personalized recommendations."""
        recommendations = []
        
        # Driving style recommendations
        if profile.driving_style == DrivingStyle.AGGRESSIVE:
            recommendations.append("Consider smoother braking to improve energy recovery by 15-20%")
        elif profile.driving_style == DrivingStyle.ECO:
            recommendations.append("Excellent eco-driving! Your regenerative efficiency is optimized")
        
        # SoC-based recommendations
        if soc > 0.9:
            recommendations.append("Battery nearly full. Consider reducing regenerative braking to protect battery health")
        elif soc < 0.2:
            recommendations.append("Low battery level. Maximize regenerative braking when possible")
        
        # Recovery strategy feedback
        if recovery['safety_applied']:
            recommendations.append("Safety limits applied to protect battery health")
        
        # Temperature-based recommendations
        if recovery.get('temperature_factor', 1.0) < 0.8:
            recommendations.append("Battery temperature affecting efficiency. Consider thermal management")
        
        return recommendations
    
    def _generate_insights(self, profile: DriverProfile, pattern: Dict) -> Dict:
        """Generate cognitive insights."""
        return {
            'driving_style_consistency': self._calculate_style_consistency(profile),
            'efficiency_trend': self._calculate_efficiency_trend(),
            'adaptation_progress': self.cognitive_state.adaptation_level,
            'predicted_range_impact': self._estimate_range_impact(profile, pattern)
        }
    
    def _calculate_style_consistency(self, profile: DriverProfile) -> float:
        """Calculate how consistent the driving style is."""
        if not hasattr(profile, 'pattern_history') or len(profile.pattern_history) < 5:
            return 0.5  # Neutral for insufficient data
        
        intensities = [p['intensity'] for p in profile.pattern_history[-10:]]
        return 1.0 - (np.std(intensities) / np.mean(intensities)) if np.mean(intensities) > 0 else 0.5
    
    def _calculate_efficiency_trend(self) -> str:
        """Calculate energy efficiency trend."""
        if len(self.cognitive_state.energy_recovery_history) < 10:
            return "insufficient_data"
        
        recent = np.mean(self.cognitive_state.energy_recovery_history[-5:])
        previous = np.mean(self.cognitive_state.energy_recovery_history[-10:-5])
        
        if recent > previous * 1.05:
            return "improving"
        elif recent < previous * 0.95:
            return "declining"
        else:
            return "stable"
    
    def _estimate_range_impact(self, profile: DriverProfile, pattern: Dict) -> str:
        """Estimate impact on driving range."""
        base_efficiency = profile.battery_usage_efficiency
        pattern_efficiency = 1.0 - (pattern['intensity'] * 0.1)  # Rough estimate
        
        combined_efficiency = base_efficiency * pattern_efficiency
        
        if combined_efficiency > 0.85:
            return "positive"
        elif combined_efficiency > 0.7:
            return "neutral"
        else:
            return "negative"
    
    def _get_temperature_factor(self, vehicle_state: Dict) -> float:
        """Get temperature factor from vehicle state."""
        battery_temp = vehicle_state.get('battery_temp', 25.0)
        if 20 <= battery_temp <= 35:
            return 1.0
        elif battery_temp < 10 or battery_temp > 45:
            return 0.8
        else:
            return 0.9
    
    def _get_traffic_factor(self, vehicle_state: Dict) -> float:
        """Get traffic factor from vehicle state."""
        # Simplified - could be enhanced with real traffic data
        avg_speed = vehicle_state.get('avg_speed', 50)
        if avg_speed < 20:
            return 0.8  # Heavy traffic
        elif avg_speed > 80:
            return 1.1  # Highway
        else:
            return 1.0  # Normal
    
    def _load_cognitive_state(self) -> CognitiveState:
        """Load cognitive state from file."""
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path, 'r') as f:
                    data = json.load(f)
                
                # Convert driver profile if present
                if data.get('current_driver_profile'):
                    data['current_driver_profile'] = DriverProfile.from_dict(data['current_driver_profile'])
                
                return CognitiveState(**data)
            except Exception as e:
                print(f"Error loading cognitive state: {e}")
        
        # Return default state
        return CognitiveState(
            current_driver_profile=None,
            recent_braking_events=[],
            energy_recovery_history=[],
            soc_prediction_confidence=0.5,
            adaptation_level=0.0,
            last_update_time=time.time()
        )
    
    def _save_cognitive_state(self):
        """Save cognitive state to file."""
        try:
            def convert_numpy_types(obj):
                """Convert numpy types to native Python types for JSON serialization."""
                if hasattr(obj, 'dtype'):
                    if obj.dtype == np.float32 or obj.dtype == np.float64:
                        return float(obj)
                    elif obj.dtype == np.int32 or obj.dtype == np.int64:
                        return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            state_data = {
                'current_driver_profile': convert_numpy_types(self.cognitive_state.current_driver_profile.to_dict()) if self.cognitive_state.current_driver_profile else None,
                'recent_braking_events': convert_numpy_types(self.cognitive_state.recent_braking_events),
                'energy_recovery_history': convert_numpy_types(self.cognitive_state.energy_recovery_history),
                'soc_prediction_confidence': convert_numpy_types(self.cognitive_state.soc_prediction_confidence),
                'adaptation_level': convert_numpy_types(self.cognitive_state.adaptation_level),
                'last_update_time': convert_numpy_types(self.cognitive_state.last_update_time)
            }
            
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            with open(self.save_path, 'w') as f:
                json.dump(state_data, f, indent=4)
        except Exception as e:
            print(f"Error saving cognitive state: {e}")
    
    def _load_driver_profiles(self) -> Dict[str, DriverProfile]:
        """Load driver profiles from file."""
        profiles_path = "shared/driver_profiles.json"
        if os.path.exists(profiles_path):
            try:
                with open(profiles_path, 'r') as f:
                    data = json.load(f)
                
                profiles = {}
                for driver_id, profile_data in data.items():
                    profiles[driver_id] = DriverProfile.from_dict(profile_data)
                
                return profiles
            except Exception as e:
                print(f"Error loading driver profiles: {e}")
        
        return {}
    
    def _save_driver_profiles(self):
        """Save driver profiles to file."""
        try:
            def convert_numpy_types(obj):
                """Convert numpy types to native Python types for JSON serialization."""
                if hasattr(obj, 'dtype'):
                    if obj.dtype == np.float32 or obj.dtype == np.float64:
                        return float(obj)
                    elif obj.dtype == np.int32 or obj.dtype == np.int64:
                        return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            profiles_data = {}
            for driver_id, profile in self.driver_profiles.items():
                profiles_data[driver_id] = convert_numpy_types(profile.to_dict())
            
            os.makedirs(os.path.dirname("shared/driver_profiles.json"), exist_ok=True)
            with open("shared/driver_profiles.json", 'w') as f:
                json.dump(profiles_data, f, indent=4)
        except Exception as e:
            print(f"Error saving driver profiles: {e}")
    
    def get_cognitive_summary(self) -> Dict:
        """Get summary of cognitive system state."""
        return {
            'active_drivers': len(self.driver_profiles),
            'adaptation_level': self.cognitive_state.adaptation_level,
            'prediction_confidence': self.cognitive_state.soc_prediction_confidence,
            'total_events_processed': len(self.cognitive_state.recent_braking_events),
            'avg_recovery_efficiency': np.mean(self.cognitive_state.energy_recovery_history) if self.cognitive_state.energy_recovery_history else 0.0,
            'last_update': self.cognitive_state.last_update_time
        }


def test_cognitive_system():
    """Test the cognitive energy management system."""
    print("=== Testing Cognitive Energy Management System ===")
    
    # Create cognitive manager
    manager = CognitiveEnergyManager()
    
    # Simulate driving events
    driver_id = "test_driver_001"
    
    for i in range(5):
        # Generate sample driving window with 7 features (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, speed)
        driving_window = np.random.randn(75, 7)
        driving_window[:, 0] = np.abs(driving_window[:, 0]) * 2 + 0.5  # acc_x
        driving_window[:, 1] = np.abs(driving_window[:, 1]) * 2 + 0.5  # acc_y
        driving_window[:, 2] = np.abs(driving_window[:, 2]) * 2 + 0.5  # acc_z
        driving_window[:, 3] = np.random.randn(75) * 0.5  # gyro_x
        driving_window[:, 4] = np.random.randn(75) * 0.5  # gyro_y
        driving_window[:, 5] = np.random.randn(75) * 0.5  # gyro_z
        driving_window[:, 6] = np.abs(driving_window[:, 6]) * 50 + 30  # Speed
        
        braking_class = np.random.randint(0, 3)
        intensity = np.random.rand()
        current_soc = 0.3 + i * 0.1
        
        vehicle_state = {
            'battery_temp': 25.0 + np.random.randn() * 5,
            'motor_temp': 30.0 + np.random.randn() * 10,
            'avg_speed': 40 + np.random.randn() * 20
        }
        
        # Process event
        result = manager.process_driving_event(
            driver_id, driving_window, braking_class, intensity, current_soc, vehicle_state
        )
        
        print(f"Event {i+1}: {result['driver_profile']['driving_style']} style, "
              f"Recovery: {result['energy_recovery']['energy_recovered']:.3f}")
    
    # Get summary
    summary = manager.get_cognitive_summary()
    print(f"\n=== Cognitive System Summary ===")
    print(f"Active drivers: {summary['active_drivers']}")
    print(f"Adaptation level: {summary['adaptation_level']:.2f}")
    print(f"Prediction confidence: {summary['prediction_confidence']:.2f}")
    print(f"Avg recovery efficiency: {summary['avg_recovery_efficiency']:.3f}")
    
    print("Cognitive system test completed!")


if __name__ == "__main__":
    test_cognitive_system()
