import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from shared.config import get_config
from shared.train_utils import set_seed, create_data_loaders, EarlyStopper, MetricsTracker, calculate_regression_metrics, save_model_checkpoint
from shared.dataset_loader import get_dataset_loader
from modules.soc.models.lstm_cnn_attention_soc import LSTMCNNAttentionSoC, train_soc_model, evaluate_soc_model


class LSTMSoCBaseline(nn.Module):
    """Plain single-stack LSTM baseline (the deployed model without its CNN front-end or
    attention). Previously this was an inline `SimpleLSTMSOC` created inside a
    `try: model = LSTMSOC() except:` block, where `LSTMSOC` was undefined and always
    raised NameError - the bare except silently masked the bug. Promoted to a real, named
    class so the baseline path is deterministic and reproducible."""

    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(-1)


def train_lstm_baseline(X_train, y_train, X_val, y_val, device, config=None, patience=5):
    if device:
        device = device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("training baseline soc model...")

    best_val_loss = float('inf')
    wait = 0  # was referenced in the early-stopping branch below before being defined

    model = LSTMSoCBaseline().to(device)
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val).float()
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(3):
        model.train()
        train_loss = 0
        for i in range(0, len(X_train), 32):
            batch_x = torch.from_numpy(X_train[i:i+32]).float().to(device)
            batch_y = torch.from_numpy(y_train[i:i+32]).float().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"epoch {epoch+1}, loss: {train_loss/len(X_train):.4f}")
        
        model.eval()
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for i in range(0, len(X_val), 32):
                batch_x = torch.from_numpy(X_val[i:i+32]).float().to(device)
                batch_y = torch.from_numpy(y_val[i:i+32]).float().to(device)
                outputs = model(batch_x)
                val_predictions.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        val_rmse = np.sqrt(mean_squared_error(val_targets, val_predictions))
        val_mae = mean_absolute_error(val_targets, val_predictions)
        val_loss = train_loss / len(X_train)
        
        if epoch % 10 == 0:
            print(f"epoch {epoch}: train loss: {train_loss:.6f}, val loss: {val_loss:.6f}, val rmse: {val_rmse:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            paths = config.get_paths_config()
            model_path = paths['models']['soc']
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_path, "lstm_soc_baseline.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print("baseline lstm soc model training complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train SoC Prediction Models")
    parser.add_argument("--baseline", action="store_true", help="Train baseline LSTM model only")
    parser.add_argument("--cnn", action="store_true", help="Train LSTM-CNN-Attention model only")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs for the deployed LSTM-CNN-Attention model (default 50)")
    parser.add_argument("--subset", type=int, default=0,
                        help="If >0, train on only this many windows (quick smoke run). Default 0 = "
                             "full dataset - the canonical run that regenerates the deployed checkpoint.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed override, for multi-seed experiments (SV-1). "
                             "Default: config system.seed (42).")

    args = parser.parse_args()

    config = get_config()

    if args.device == "auto":
        device = config.get_device()
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"using device: {device}")

    seed = args.seed if args.seed is not None else config.get('system.seed', 42)
    print(f"using seed: {seed}")
    set_seed(seed)
    
    try:
        print("loading soc datasets...")
        dataset_loader = get_dataset_loader()
        dataset_info = dataset_loader.get_dataset_info('soc')
        
        print(f"using {dataset_info['source']} dataset")
        print(f"dataset info: {dataset_info}")
        
        X_train, X_val, X_test, y_train, y_val, y_test = dataset_loader.load_soc_dataset()

        # Canonical run (--subset 0, the default) trains on the full dataset and regenerates
        # the deployed checkpoint reproducibly. --subset N is an opt-in quick smoke run only;
        # it does NOT reproduce the deployed checkpoint.
        if args.subset and args.subset > 0:
            subset_size = min(args.subset, len(X_train))
            X_train = X_train[:subset_size]
            y_train = y_train[:subset_size]
            X_val = X_val[:max(1, subset_size // 5)]
            y_val = y_val[:max(1, subset_size // 5)]
            X_test = X_test[:max(1, subset_size // 5)]
            y_test = y_test[:max(1, subset_size // 5)]
            print(f"[quick mode] training on a {subset_size}-window subset (NOT the canonical checkpoint)")
        else:
            print("[canonical mode] training on the full dataset")

        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
    except FileNotFoundError as e:
        print(f"Dataset files not found: {e}")
        print("Please run dataset generation first!")
        return
    
    # Train models based on arguments
    if args.baseline or (not args.cnn):
        baseline_model = train_lstm_baseline(X_train, y_train, X_val, y_val, device, config)
        
        baseline_model.eval()
        with torch.no_grad():
            test_predictions = baseline_model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
        
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_mae = mean_absolute_error(y_test, test_predictions)
        print(f"\nBaseline LSTM Test Results:")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"MAE: {test_mae:.4f}")
    
    if args.cnn or (not args.baseline):
        print("\nTraining LSTM-CNN-Attention SoC Model (canonical, via train_soc_model)...")

        # Route the deployed checkpoint through the one shared trainer used everywhere else
        # (lstm_cnn_attention_soc.train_soc_model), rather than a divergent inline copy. This
        # also fixes a real defect: the old inline path saved via save_model_checkpoint(),
        # which wraps the weights in {'model_state_dict': ...}, but the deployed pipeline
        # (shared/enhanced_utils.py) loads a *raw* state_dict - so that checkpoint could not
        # actually be loaded for deployment. train_soc_model() saves a raw state_dict, which
        # both enhanced_utils.py and evaluate_soc.py load correctly.
        paths = config.get_paths_config()
        model_path = paths['models']['soc']
        os.makedirs(model_path, exist_ok=True)
        # Default invocation (no --seed) writes the exact same canonical filename as before -
        # unchanged behavior. An explicit --seed suffixes the filename so multi-seed runs
        # (SV-1) never collide with the canonical checkpoint or with each other.
        ckpt_name = "lstm_cnn_attention_soc.pth" if args.seed is None else f"lstm_cnn_attention_soc_seed{args.seed}.pth"
        deployed_ckpt = os.path.join(model_path, ckpt_name)

        cnn_model = LSTMCNNAttentionSoC()
        cnn_model, history = train_soc_model(
            cnn_model, X_train, y_train, X_val, y_val,
            epochs=args.epochs, device=device, save_path=deployed_ckpt,
        )

        # Preserve the existing metrics side-output, sourced from the returned history's
        # best epoch (val RMSE/MAE are in raw [0,1] scale, same as before).
        best_idx = int(np.argmin(history["val_rmse"]))
        final_metrics = {
            'val_rmse': float(history["val_rmse"][best_idx]),
            'val_mae': float(history["val_mae"][best_idx]),
            'training_epochs': len(history["val_rmse"]),
            'model_type': 'lstm_cnn_attention_soc',
        }
        import json
        with open(os.path.join(model_path, "lstm_cnn_attention_soc_metrics.json"), 'w') as f:
            json.dump(final_metrics, f, indent=2)

        # Report test metrics in % SoC via the canonical evaluator.
        cnn_results = evaluate_soc_model(
            cnn_model, X_test.astype(np.float32), y_test.astype(np.float32), device=device
        )

    print("All SoC training completed!")


if __name__ == "__main__":
    main()
