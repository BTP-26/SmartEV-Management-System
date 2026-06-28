"""
enhanced ev smart management system streamlit ui
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import json
import time

from shared.config import get_config
from shared.enhanced_utils import EnhancedEVPipeline

def generate_driving_sequence(seq_len=75, init_speed=60, aggressiveness=0.5, noise_level=0.05):
    """generate driving sequence for braking prediction with 7 features"""
    speed = init_speed
    brake = 0.0
    data = []

    for _ in range(seq_len):
        brake += (aggressiveness - brake) * np.random.uniform(0.03, 0.08)
        brake += np.random.normal(0, noise_level)
        brake = np.clip(brake, 0, 1)

        # generate 7 features: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, speed
        accel_x = -brake * np.random.uniform(2.0, 3.5) + np.random.normal(0, noise_level)
        accel_y = np.random.normal(0, noise_level)
        accel_z = np.random.normal(0, noise_level)
        gyro_x = np.random.normal(0, noise_level * 0.5)
        gyro_y = np.random.normal(0, noise_level * 0.5)
        gyro_z = np.random.normal(0, noise_level * 0.5)
        
        speed += accel_x * 0.1
        speed = max(speed, 0)

        data.append([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, speed])

    return np.array(data, dtype=np.float32)

def generate_battery_sequence(seq_len=50, initial_soc=0.7, discharge_rate=0.01, noise_level=0.02):
    """generate battery sequence for soc prediction"""
    voltage = 3.8 - 0.5 * np.linspace(0, 1, seq_len)
    current = -1.0 * np.ones(seq_len) + np.random.normal(0, 0.05, seq_len)
    temperature = 25 + np.random.normal(0, 0.5, seq_len)
    
    # add noise
    voltage += np.random.normal(0, noise_level, seq_len)
    current += np.random.normal(0, noise_level * 0.5, seq_len)
    
    return np.stack([voltage, current, temperature], axis=1).astype(np.float32)

# scenario presets
SCENARIOS = {
    "Manual (use sliders)": None,
    "Highway Gentle Slowdown": (90, 0.18, 0.05),
    "City Traffic Braking": (50, 0.45, 0.08),
    "Sudden Emergency Stop": (80, 0.9, 0.04),
    "Stop-and-Go Traffic": (40, 0.6, 0.12),
}

battery_scenarios = {
    "Normal Driving": (0.7, 0.01, 0.02),
    "Aggressive Driving": (0.8, 0.02, 0.03),
    "Eco Mode": (0.6, 0.005, 0.01),
    "Battery Low": (0.2, 0.015, 0.02),
}

# load enhanced pipeline
@st.cache_resource
def load_enhanced_pipeline():
    """load the enhanced ev pipeline"""
    try:
        return EnhancedEVPipeline()
    except Exception as e:
        st.error(f"failed to load pipeline: {e}")
        return None

# page config
st.set_page_config(
    page_title="ev smart management system", 
    layout="wide",
    page_icon="EV-Smart-Management-System/assets/img/1.jpg"
)

# title
st.markdown(
    """
    <h1 style="text-align:center;">EV Smart Management System</h1>
    <p style="text-align:center; color:#b0b0b0;">
    Unified Braking Intention + Battery SoC Prediction for Electric Vehicles
    </p>
    """,
    unsafe_allow_html=True
)

# top explainers
c1, c2, c3 = st.columns(3)

with c1:
    with st.expander("What does this system do?"):
        st.write(
            "Predicts **driver braking intention** (Light / Normal / Emergency) "
            "and **battery State-of-Charge** using deep learning models. "
            "Integrates both for intelligent regenerative braking control."
        )

with c2:
    with st.expander("What data is used?"):
        st.write(
            "**Braking Module:**\n"
            "- Vehicle speed\n"
            "- Acceleration (deceleration)\n"
            "- Brake pedal input\n\n"
            "**SoC Module:**\n"
            "- Battery voltage\n"
            "- Current draw\n"
            "- Temperature"
        )

with c3:
    with st.expander("How to interpret results?"):
        st.write(
            "- **Braking:** Light/Normal/Emergency with intensity\n"
            "- **Energy Recovery:** Calculated regenerative energy\n"
            "- **SoC Update:** Battery state after regen\n"
            "- **System Action:** EV controller recommendation"
        )

st.divider()

st.subheader("Scenario & Controls")

sc_col, sliders_col, run_col = st.columns([1.1, 2.1, 0.8])

with sc_col:
    st.markdown("### **Scenario Preset**")
    scenario = st.selectbox(
        "Choose a predefined driving scenario:",
        list(SCENARIOS.keys())
    )

    battery_scenario = st.selectbox(
        "Choose battery scenario:",
        list(BATTERY_SCENARIOS.keys())
    )

with sliders_col:
    st.markdown("### **Input Controls**")
    
    st.markdown("**Driving Parameters**")
    if SCENARIOS[scenario]:
        init_speed, aggressiveness, noise_level = SCENARIOS[scenario]
    else:
        init_speed = aggressiveness = noise_level = None

    init_speed = st.slider(
        "Initial Speed (km/h)", 20, 120,
        init_speed if init_speed is not None else 60
    )

    aggressiveness = st.slider(
        "Braking Aggressiveness", 0.1, 1.0,
        aggressiveness if aggressiveness is not None else 0.5
    )

    noise_level = st.slider(
        "Noise Level", 0.0, 0.2,
        noise_level if noise_level is not None else 0.05
    )
    
    # Battery controls
    st.markdown("**Battery Parameters**")
    if BATTERY_SCENARIOS[battery_scenario]:
        initial_soc, discharge_rate, battery_noise = BATTERY_SCENARIOS[battery_scenario]
    else:
        initial_soc = discharge_rate = battery_noise = None

    initial_soc = st.slider(
        "Initial SoC", 0.1, 1.0,
        initial_soc if initial_soc is not None else 0.7
    )

with run_col:
    st.markdown("<br><br>", unsafe_allow_html=True)
    run = st.button(
        "**Run Prediction**",
        use_container_width=True
    )

st.divider()

# Main layout
left, right = st.columns([1.6, 1])

if run:
    # Generate sequences
    driving_sequence = generate_driving_sequence(
        init_speed=init_speed,
        aggressiveness=aggressiveness,
        noise_level=noise_level
    )

    battery_sequence = generate_battery_sequence(
        initial_soc=initial_soc,
        discharge_rate=discharge_rate if discharge_rate else 0.01,
        noise_level=battery_noise if battery_noise else 0.02
    )

    # LEFT: Time-series plots
    with left:
        st.subheader("Input Time-Series Signals")

        fig, axs = plt.subplots(2, 3, figsize=(10, 6))
        
        # Driving signals
        axs[0, 0].plot(driving_sequence[:, 0], color="#1f77b4")
        axs[0, 0].set_ylabel("Speed")
        axs[0, 0].set_title("Driving Signals")
        axs[0, 0].grid(alpha=0.3)

        axs[0, 1].plot(driving_sequence[:, 1], color="#ff7f0e")
        axs[0, 1].set_ylabel("Acceleration")
        axs[0, 1].grid(alpha=0.3)

        axs[0, 2].plot(driving_sequence[:, 2], color="#d62728")
        axs[0, 2].set_ylabel("Brake Pedal")
        axs[0, 2].grid(alpha=0.3)
        
        # Battery signals
        axs[1, 0].plot(battery_sequence[:, 0], color="#2ca02c")
        axs[1, 0].set_ylabel("Voltage")
        axs[1, 0].set_xlabel("Time Step")
        axs[1, 0].set_title("Battery Signals")
        axs[1, 0].grid(alpha=0.3)

        axs[1, 1].plot(battery_sequence[:, 1], color="#d62728")
        axs[1, 1].set_ylabel("Current")
        axs[1, 1].set_xlabel("Time Step")
        axs[1, 1].grid(alpha=0.3)

        axs[1, 2].plot(battery_sequence[:, 2], color="#ff7f0e")
        axs[1, 2].set_ylabel("Temperature")
        axs[1, 2].set_xlabel("Time Step")
        axs[1, 2].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

    # RIGHT: Predictions
    with right:
        pipeline = load_enhanced_pipeline()
        
        if pipeline is None:
            st.error("Failed to load prediction pipeline")
        else:
            # Run inference
            start_time = time.time()
            try:
                result = pipeline.run(driving_sequence, battery_sequence, initial_soc)
                inference_time = (time.time() - start_time) * 1000
                
                st.subheader("Prediction Results")

                # Braking prediction
                braking_class = result['braking']['class']
                intensity = result['braking']['intensity']
                confidence = result['braking']['confidence']
                
                # SoC prediction
                soc_current = result['soc']['current']
                soc_updated = result['soc']['updated']
                soc_delta = result['soc']['delta']
                
                # Energy recovery
                energy_recovered = result['energy']['recovered_normalised']
                
                # System action
                system_action = result['system_action']
                
                # Color coding for braking class
                colors = {
                    "Light Braking": "#2ecc71",
                    "Normal Braking": "#f1c40f", 
                    "Emergency Braking": "#e74c3c"
                }
                emojis = {
                    "Light Braking": "Light",
                    "Normal Braking": "Normal", 
                    "Emergency Braking": "Emergency"
                }
                
                color = colors.get(braking_class, "#95a5a6")
                
                # Main result display
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, {color} 0%, {color} 100%);
                        padding: 20px;
                        border-radius: 14px;
                        text-align: center;
                        color: white;
                        font-size: 22px;
                        font-weight: bold;
                        margin-bottom: 20px;">
                        {emojis.get(braking_class, braking_class)} {braking_class}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Brake Intensity", f"{intensity:.3f}")
                    st.metric("Confidence", f"{confidence:.2f}")
                    st.metric("Energy Recovered", f"{energy_recovered:.3f}")
                
                with col2:
                    st.metric("Current SoC", f"{soc_current:.2%}")
                    st.metric("Updated SoC", f"{soc_updated:.2%}")
                    st.metric("SoC Change", f"{soc_delta:+.3f}")
                
                # System action
                st.markdown("**System Recommendation:**")
                st.info(system_action)
                
                # Performance info
                st.markdown("**Performance:**")
                st.caption(f"Inference time: {inference_time:.2f}ms")
                
                # Detailed results expander
                with st.expander("**Detailed Results**"):
                    st.json(result)
                
                # Download report
                report = {
                    "timestamp": time.time(),
                    "scenario": scenario,
                    "battery_scenario": battery_scenario,
                    "initial_speed": init_speed,
                    "aggressiveness": aggressiveness,
                    "initial_soc": initial_soc,
                    "prediction": {
                        "braking_class": braking_class,
                        "intensity": round(intensity, 3),
                        "confidence": round(confidence, 3),
                        "energy_recovered": round(energy_recovered, 3),
                        "soc_current": round(soc_current, 3),
                        "soc_updated": round(soc_updated, 3),
                        "soc_delta": round(soc_delta, 3),
                        "system_action": system_action
                    },
                    "inference_time_ms": round(inference_time, 2)
                }

                buffer = io.StringIO()
                json.dump(report, buffer, indent=2)

                st.download_button(
                    "Download Prediction Report",
                    buffer.getvalue(),
                    file_name="ev_smart_prediction_report.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 12px;'>
    EV Smart Management System | Enhanced with Input Validation | Error Handling | Batch Inference | Quantization
    </div>
    """,
    unsafe_allow_html=True
)
