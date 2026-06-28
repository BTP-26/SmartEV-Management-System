import os
import sys
import torch
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from shared.train_utils import set_seed, create_data_loaders, EarlyStopper, MetricsTracker

def main():
    print("TRAINING ALL MODELS")
    if torch.backends.mps.is_available():
       device = torch.device("mps")  # Mac GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") 
    print(f"Device: {device}")
    
    set_seed(42)  # For reproducibility

    # 1. BRAKING BASELINE (LSTM-CNN-Attention)
    print("\n1. Training Braking Baseline...")
    try:
        from modules.train.train_braking import main as train_braking_main
        # Use subprocess to call train_braking.py with baseline flag
        import subprocess
        result = subprocess.run([
            sys.executable, "modules/train/train_braking.py", 
            "--baseline", "--device", str(device)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" Braking baseline saved")
        else:
            print(f" Braking baseline training failed: {result.stderr}")
            
    except Exception as e:
        print(f" Error training braking baseline: {e}")

    # 2. BRAKING HARD MULTITASK + GA
    print("\n2. Running GA + Multitask...")
    try:
        from modules.train.train_braking import main as train_braking_main
        # Use subprocess to call train_braking.py with multitask and GA flags
        import subprocess
        result = subprocess.run([
            sys.executable, "modules/train/train_braking.py", 
            "--multitask", "--ga", "--device", str(device)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" Braking GA complete (best_ga_hyperparams.json)")
        else:
            print(f" Braking GA training failed: {result.stderr}")
            
    except Exception as e:
        print(f" Error running GA optimization: {e}")

    # 3. SoC MODELS
    print("\n3. Training SoC models...")
    try:
        from modules.train.train_soc import main as train_soc_main
        # Use subprocess to call train_soc.py
        import subprocess
        result = subprocess.run([
            sys.executable, "modules/train/train_soc.py", 
            "--device", str(device)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(" SoC models trained")
        else:
            print(f" SoC training failed: {result.stderr}")
            
    except Exception as e:
        print(f" Error training SoC models: {e}")

    print("\n ALL MODELS TRAINED! .pth files ready.")
    print("Run: python run_unified.py")


if __name__ == "__main__":
    main()
