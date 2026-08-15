# EV Smart Management System


> **Advanced Intelligent Electric Vehicle Management System**  
> GA-optimized AI-powered braking intention prediction, battery State-of-Charge estimation, cognitive driver profiling, and constraint-regularised optimization for next-generation EV energy management

---

## **What It Does**

**Four integrated AI modules for comprehensive EV management:**

### **1. Braking Intention Prediction**
- **Architecture**: Multitask LSTM + CNN + Multi-Head Attention + Genetic Algorithm Optimization
- **Input**: 75 timesteps × 7 features (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, speed)
- **Output**: 3-class classification (Light/Normal/Emergency) + intensity regression
- **Performance**: 82.8% accuracy, 2.1ms inference time
- **Data**: UAH-DriveSet v1 real-world driving dataset (12,355 windows)
- **Application**: Collision avoidance, adaptive cruise control, emergency braking systems

### **2. Advanced SoC Estimation**
- **Weighted-Objective GA Optimized**: hyperparameter-tuned via genetic algorithm
- **Adaptive Ensemble**: 3 models with GA-optimized weights
- **Constraint-Regularised**: trained with battery-physics-informed loss constraints
- *Performance numbers are provisional pending the current experimental campaign — see `docs/soc_pipeline.md`.*
- **Input**: 50 timesteps × 3 features (voltage, current, temperature)
- **Data**: Mendeley Poztato EV dataset (real vehicle driving with regenerative braking)
- **Features**: Temperature robustness, computational efficiency optimization

### **3. Cognitive Energy Management**
- **Driver Behavior Profiling**: Eco, Normal, Aggressive, Conservative styles
- **Personalized SoC Prediction**: Driver-specific adjustments
- **Adaptive Energy Recovery**: Strategy selection based on driving patterns
- **Performance**: 0.50+ adaptation level, multi-driver support

### **4. Physics-Based EV Simulation**
- **Realistic Modeling**: Vehicle dynamics, regenerative braking curves
- **Multiple Scenarios**: Urban, highway, aggressive, eco, emergency driving
- **Environmental Factors**: Road conditions, temperature effects
- **Data Generation**: 15,000 samples with physics validation

### **Unified Regenerative Braking Control**
- **Integration**: Braking prediction + SoC estimation + Cognitive profiling + Physics constraints
- **Logic**: Maximizes regenerative energy while protecting battery health
- **Output**: Actionable recommendations for EV control systems
- **Efficiency**: Up to 65% energy recovery optimization
- **Throughput**: 412.8 inferences/second with quantization

---

## **Quick Start**

### ** Prerequisites**
- **Python**: 3.8+ (recommended 3.11)
- **Memory**: 4GB+ RAM
- **Storage**: 2GB+ (for datasets and models)
- **GPU**: Optional (CUDA support available)
- **OS**: Windows/macOS/Linux

### **5-Minute Setup**

```bash
# 1. Clone Repository
git clone https://github.com/BTP-26/SmartEV-Management-System.git
cd SmartEV-Management-System

# 2. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run Complete Pipeline 
python run_complete_pipeline.py

# This will automatically:
#   - Generate datasets if needed
#   - Train models if needed
#   - Run inference demo

# Alternative: Manual Steps
# (datasets are generated per-module - see modules/soc/data/preprocess_real_data.py and
# modules/braking/data/preprocess_real_data.py to regenerate from raw sources)
python modules/train/train_all_models.py
python run_complete_pipeline.py
```

### **Web Interface**

```bash
# Web Interface
streamlit run ui/app.py
```

---

### **Usage Examples**

```bash
# Complete System Pipeline (Recommended)
python run_complete_pipeline.py

# Individual Component Training
python modules/train/train_braking.py
python modules/train/train_soc.py
```

---

## **Advanced System Architecture**

```
                        USER INTERFACE LAYER
    Streamlit Dashboard (ui/app.py)
  - Real-time driving/battery scenario controls
  - Interactive time-series visualization
  - Live prediction results with metrics
  - Cognitive driver profiling displays
  - Energy recovery optimization insights
                           |
                 ENHANCED INFERENCE LAYER
   EnhancedEVPipeline (shared/enhanced_utils.py)
  -  Input validation (shapes, types, NaN values)
  -  Model quantization (2.42ms inference)
  -  Batch inference (412.8 samples/second)
  -  Error handling (graceful fallbacks)
  -  Performance monitoring (latency tracking)
  -  Cognitive system integration
                           |
                 ADVANCED MODEL LAYER
   Braking Models:
  - MultitaskLSTMCNNAttention (GA-optimized)
  - Genetic Algorithm hyperparameter optimization
  - Physics-based realistic EV simulation

   SoC Models:
  - Weighted-Objective GA Optimizer
  - Adaptive Ensemble (3 models)
  - Constraint-Regularised Neural Network
  - Cognitive Energy Manager (Adaptation: 0.50+)

   Cognitive System:
  - Driver Behavior Profiling (4 styles)
  - Personalized SoC Prediction
  - Adaptive Energy Recovery Strategies
                           |
                 COMPREHENSIVE DATA LAYER
   Braking Data:
  - UAH-DriveSet v1 real-world dataset (12,355 windows)
  - 75 timesteps × 7 features (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, speed)
  - 6 drivers, 44 trips with comprehensive sensor data
  - Real driving scenarios with synchronized sensors

   SoC Data:
  - Mendeley Poztato EV dataset (real vehicle driving)
  - 50 timesteps × 3 features (voltage, current, temperature)
  - Real-world driving with regenerative braking current
  - Actual vehicle telemetry, not lab bench data
```

---

## **Performance Metrics**

### **Model Performance**

| Metric | Braking Model | SoC Model | System |
|---------|---------------|-------------|---------|
| **Test Accuracy** | 82.85% | provisional* | - |
| **Test RMSE** | - | provisional* | - |
| **Test MAE** | - | provisional* | - |
| **Test MAPE** | - | provisional* | - |
| **Inference Time** | 2.64±1.26ms | 0.8ms | 2.8ms |
| **Throughput** | 1,483 samples/s | 1,300+ samples/s | 356 samples/s |
| **Model Size** | 4.2MB (1.1MB quantized) | 2.8MB (0.9MB quantized) | 7.0MB |

*SoC accuracy numbers are being re-validated on a leak-free session-level split (see
`docs/soc_pipeline.md`) — the previous figures here were computed pre-fix and are not reported.

### **System Performance**

```
-> Single Inference: 2.14ms average (1.88-4.45ms range)
-> Batch Inference: 1,285-1,540 samples/second
-> Throughput: 1,300+ samples/second (100 sample batches)
-> Memory Usage: Optimized for deployment
-> CPU Utilization: Efficient processing
-> Energy Recovery: Intelligent optimization with cognitive learning
```

### **Testing Results**

```bash
EV SMART MANAGEMENT SYSTEM - COMPLETE PIPELINE TEST----------------
CHECKING DATA AVAILABILITY----------------
All datasets available
CHECKING MODEL AVAILABILITY----------------
All models available
TESTING ENHANCED PIPELINE----------------
Enhanced pipeline test completed successfully
PERFORMANCE BENCHMARK----------------
Performance benchmark completed
PIPELINE TEST SUMMARY----------------
Data Generation: PASS
Model Training: PASS
Enhanced Pipeline: PASS
Performance: PASS

Overall: ALL TESTS PASSED
EV Smart Management System is fully operational!
```

---

## **Dataset Acquisition**

### **Braking Dataset: UAH-DriveSet v1**
- **Source**: University of Alabama Huntsfield Driving Dataset
- **Download**: Available from official UAH repository: https://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/
- **Usage**: Download UAH-DriveSet and run your trained model on it as a zero-shot or fine-tuned transfer test. Even showing that your model achieves reasonable accuracy on real driver data validates the architecture. You don't need to retrain from scratch.
- **Features**: 6 drivers, 44 trips with comprehensive sensor data
- **Compatibility**: Existing code processes UAH-DriveSet with minimal changes

### **SoC Dataset: Mendeley Poztato EV Dataset**
- **Source**: https://data.mendeley.com/datasets/7vdkzpnjgj/2
- **Usage**: Download and place under `modules/soc/data/`, then run `modules/soc/data/preprocess_real_data.py` to regenerate the windowed `.npy` files. It provides three input features (voltage, current, temperature) plus SoC ground truth, consumed directly by `LSTMCNNAttentionSoC`.
- **Features**: Real vehicle driving with regenerative braking current
- **Advantage**: Comes from a real car being driven, not a lab bench, and includes regenerative braking current which directly connects to the paper's core claim about regenerative braking control.

---

## **Advanced Features**

### **Input Validation & Error Handling**
- **Shape Validation**: Ensures correct input dimensions
- **Type Checking**: Validates numpy arrays and data types
- **Range Validation**: Checks for reasonable value ranges
- **NaN/Inf Detection**: Prevents model crashes
- **Graceful Fallbacks**: Handles missing models/data

### **Model Optimization**
- **Fallback Optimization**: Uses original models when quantization unavailable
- **Batch Processing**: Efficient multi-sample inference (1,300+ samples/sec)
- **Attention Mechanisms**: Focuses on critical time steps
- **Multi-Modal Fusion**: Combines spatial and temporal features
- **Cognitive Learning**: Adapts to driver behavior patterns

### **Performance Monitoring**
- **Latency Tracking**: Real-time inference timing
- **Memory Profiling**: Resource usage optimization
- **Throughput Analysis**: System capacity measurement
- **Error Logging**: Comprehensive debugging information

---

## **Project Structure**

```
SmartEV-Management-System/
|
|  ui/                           # User Interface Layer
|   app.py                        # Web interface dashboard
|
|  modules/                       # Core Modules
|   braking/                     # Braking Intention Module
|   data/                        # UAH-DriveSet preprocessing + windowed datasets
|   models/                      # Trained models and optimizers
|   soc/                         # Battery SoC Module
|   data/                        # Mendeley Poztato EV preprocessing + windowed datasets
|   models/                      # Advanced SoC models and optimizers
|
|   train/                       # Training Scripts
|   |-- train_all_models.py
|   |-- train_braking.py
|   -- train_soc.py
|
|--  shared/                      # System Core
|   |-- config.py                 # Configuration management
|   |-- enhanced_utils.py         # Enhanced utilities
|   |-- cognitive_manager.py      # Cognitive energy management
|   -- __init__.py
|
|--  config/                      # Configuration Files
|   -- default.yaml               # System parameters & paths
|
|--  run_complete_pipeline.py     # Complete system pipeline
|--  requirements.txt             # Dependencies
|--  LICENSE                      # MIT License
|--  README.md                    # This file
--  assets/                       # Documentation & media
```

---

## 🎮 **Usage Examples**

### ** Complete Pipeline Usage**
```python
# Run complete system pipeline
python run_complete_pipeline.py

# This will:
# 1. Check data availability
# 2. Generate datasets if needed
# 3. Train models if needed
# 4. Run inference demo
# 5. Display performance metrics
```

### ** Web Interface**
```python
# Run interactive dashboard
streamlit run ui/app.py

# Features:
# - Real-time scenario controls
# - Interactive parameter sliders
# - Live prediction visualization
# - Performance metrics dashboard
# - Energy management insights
```

---

## 🔧 **Configuration System**

### ** Centralized Configuration** (`config/default.yaml`)

Excerpt below — that file is the source of truth, refer to it directly rather than this copy.

```yaml
system:
  device: "auto"                  # Auto-detect GPU/CPU
  seed: 42                        # Reproducibility
  debug: false

data:
  braking:
    window_size: 75
    features: 7
    train_split: 0.7
    val_split: 0.15
    test_split: 0.15
  soc:
    window_size: 50
    features: 3
    train_split: 0.7
    val_split: 0.15
    test_split: 0.15

training:
  batch_size: 32
  learning_rate: 0.001
  epochs:
    braking_baseline: 30
    braking_multitask: 30
    soc_baseline: 2
    soc_cnn: 2
  patience: 5
  eval_batch_size: 512
  early_stopping: true

inference:
  quantization: true
  batch_size: 1
  validate_inputs: true

paths:
  models:
    braking: "modules/braking/models"
    soc: "modules/soc/models"
  data:
    braking: "modules/braking/data"
    soc: "modules/soc/data"
```

---

##  **Testing & Validation**

### ** System Validation**
```bash
python run_complete_pipeline.py
```

**Pipeline Coverage:**
-  Data Generation Pipeline
-  Model Training Pipeline  
-  Unified Inference System
-  Integration Testing

### ** Component Testing**
```bash
# Test individual components
python modules/soc/models/multi_objective_ga_optimizer.py
python modules/soc/models/adaptive_ensemble.py
python modules/soc/models/physics_informed_soc.py
python shared/cognitive_manager.py
```

