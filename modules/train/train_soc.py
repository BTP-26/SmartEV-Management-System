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


def train_lstm_baseline(X_train, y_train, X_val, y_val, device, config=None):
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
    
    try:
        model = LSTMSOC()
    except:
        class SimpleLSTMSOC(nn.Module):
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
        
        model = SimpleLSTMSOC()
    
    model = model.to(device)
    
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
    parser = argparse.ArgumentParser(description="Train SOC Prediction Models")
    parser.add_argument("--baseline", action="store_true", help="Train baseline LSTM model only")
    parser.add_argument("--cnn", action="store_true", help="Train LSTM+CNN+Attention model only")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    config = get_config()
    
    if args.device == "auto":
        device = config.get_device()
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"using device: {device}")
    
    set_seed(config.get('system.seed', 42))
    
    try:
        print("loading soc datasets...")
        dataset_loader = get_dataset_loader()
        dataset_info = dataset_loader.get_dataset_info('soc')
        
        print(f"using {dataset_info['source']} dataset")
        print(f"dataset info: {dataset_info}")
        
        X_train, X_val, X_test, y_train, y_val, y_test = dataset_loader.load_soc_dataset()
        
        subset_size = min(5000, len(X_train))
        if len(X_train) > subset_size:
            X_train = X_train[:subset_size]
            y_train = y_train[:subset_size]
            X_val = X_val[:subset_size//5]
            y_val = y_val[:subset_size//5]
            X_test = X_test[:subset_size//5]
            y_test = y_test[:subset_size//5]
        
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
        print("\nTraining LSTM+CNN+Attention SOC Model...")
        
        best_val_loss = float('inf')
        
        num_classes = len(np.unique(y_train))
        
        cnn_model = LSTMCNNAttentionSoC()
        
        train_loader = DataLoader(TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        ), batch_size=32, shuffle=False)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
        
        patience = 2
        wait = 0
        
        for epoch in range(2):
            cnn_model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = cnn_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            cnn_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = cnn_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            train_loss /= len(train_loader)
            
            # Calculate regression metrics on validation set
            cnn_model.eval()
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = cnn_model(batch_x)
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_predictions = np.array(val_predictions)
            val_targets = np.array(val_targets)
            val_metrics = calculate_regression_metrics(val_targets, val_predictions)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                paths = config.get_paths_config()
                model_path = paths['models']['soc']
                os.makedirs(model_path, exist_ok=True)
                
                # Prepare metrics to save with model
                final_metrics = {
                    'val_rmse': val_metrics['rmse'],
                    'val_mae': val_metrics['mae'],
                    'val_mape': val_metrics['mape'],
                    'best_val_loss': best_val_loss,
                    'training_epochs': epoch + 1,
                    'model_type': 'lstm_cnn_attention_soc'
                }
                
                # Save model with metrics
                save_model_checkpoint(cnn_model, optimizer, epoch, val_loss,
                                    os.path.join(model_path, "lstm_cnn_attention_soc.pth"),
                                    final_metrics)
                
                # Also save metrics separately as JSON for easy loading
                import json
                with open(os.path.join(model_path, "lstm_cnn_attention_soc_metrics.json"), 'w') as f:
                    json.dump(final_metrics, f, indent=2)
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        X_test_float32 = X_test.astype(np.float32)
        y_test_float32 = y_test.astype(np.float32)
        cnn_results = evaluate_soc_model(cnn_model, X_test_float32, y_test_float32, device=device)
    
    print("All SoC training completed!")


if __name__ == "__main__":
    main()
