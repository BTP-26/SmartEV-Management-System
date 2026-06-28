
import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from shared.config import get_config
from shared.train_utils import MetricsTracker, calculate_classification_metrics, save_model_checkpoint, create_data_loaders, EarlyStopper
from shared.dataset_loader import get_dataset_loader
from modules.braking.models.multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention
from modules.braking.models.genetic_algorithm_optimizer import GeneticAlgorithmOptimizer


def train_baseline_model(X_train, y_train, X_val, y_val, device="cpu", config=None):
    print("training baseline model...")
    
    unique_classes = len(np.unique(y_train))
    if unique_classes == 1:
        print("error: only one class in dataset.")
        return None
    
    num_classes = len(np.unique(y_train))
    
    model = MultitaskLSTMCNNAttention(
        input_dim=X_train.shape[2], 
        lstm_hidden=64, 
        num_lstm_layers=1,
        cnn_channels=32,
        dropout_rate=0.0
    )
    
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model = model.to(device)
    
    y_train_int = y_train.astype(np.int64)
    y_val_int = y_val.astype(np.int64)
    
    train_dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train_int).long()
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val).float(),
        torch.from_numpy(y_val_int).long()
    )
    
    training_config = config.get_training_config() if config else {}
    batch_size = training_config.get('batch_size', 32)
    learning_rate = training_config.get('learning_rate', 0.001)
    epochs = training_config.get('epochs.braking_baseline', 3)
    patience = training_config.get('patience', 2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    wait = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            int_output, class_output = model(batch_x)
            int_output_single = int_output[:, 0:1]
            batch_y_float = batch_y.float().unsqueeze(1)
            loss = criterion(int_output_single, batch_y_float)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                int_output, class_output = model(batch_x)
                int_output_single = int_output[:, 0:1]
                batch_y_float = batch_y.float().unsqueeze(1)
                loss = criterion(int_output_single, batch_y_float)
                val_loss += loss.item()
                
                predictions = int_output_single.squeeze()
                threshold = 0.5
                predicted_binary = (predictions > threshold).long()
                target_binary = batch_y
                total += batch_y.size(0)
                correct += (predicted_binary == target_binary).sum().item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        val_acc = 100 * correct / total
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            paths = config.get_paths_config()
            model_path = paths['models']['braking']
            os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_path, "lstm_cnn_attention_baseline.pth"))
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print("baseline model training complete!")
    return model


def train_multitask_model(X_train, y_class_train, y_int_train, X_val, y_class_val, y_int_val, device="cpu", config=None):
    """train multitask model with both classification and regression"""
    print("training multitask model...")
    
    model = MultitaskLSTMCNNAttention(input_dim=X_train.shape[2])
    model = model.to(device)
    
    if len(y_int_train) < len(y_class_train):
        y_int_train_padded = np.zeros(len(y_class_train))
        y_int_train_padded[:len(y_int_train)] = y_int_train
        y_int_train = y_int_train_padded
    
    if len(y_int_val) < len(y_class_val):
        y_int_val_padded = np.zeros(len(y_class_val))
        y_int_val_padded[:len(y_int_val)] = y_int_val
        y_int_val = y_int_val_padded
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_class_train, dtype=torch.long),
        torch.tensor(y_int_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_class_val, dtype=torch.long),
        torch.tensor(y_int_val, dtype=torch.float32)
    )
    
    # Get training config
    training_config = config.get_training_config() if config else {}
    batch_size = training_config.get('batch_size', 32)
    learning_rate = training_config.get('learning_rate', 0.001)
    epochs = training_config.get('epochs.braking_multitask', 3)
    patience = training_config.get('patience', 2)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    wait = 0
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y_cls, batch_y_int in train_loader:
            batch_x, batch_y_cls, batch_y_int = batch_x.to(device), batch_y_cls.to(device), batch_y_int.to(device)
            
            optimizer.zero_grad()
            cls_outputs, int_outputs = model(batch_x)
            
            cls_loss = criterion_cls(cls_outputs, batch_y_cls)
            reg_loss = criterion_reg(int_outputs, batch_y_int)
            total_loss = cls_loss + 0.5 * reg_loss  # Weighted loss
            
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        cls_correct = 0
        cls_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y_cls, batch_y_int in val_loader:
                batch_x, batch_y_cls, batch_y_int = batch_x.to(device), batch_y_cls.to(device), batch_y_int.to(device)
                cls_outputs, int_outputs = model(batch_x)
                
                loss_cls = criterion_cls(cls_outputs, batch_y_cls)
                loss_reg = criterion_reg(int_outputs, batch_y_int)
                loss = 0.7 * loss_cls + 0.3 * loss_reg
                val_loss += loss.item()
                
                _, predicted = torch.max(cls_outputs.data, 1)
                cls_total += batch_y_cls.size(0)
                cls_correct += (predicted == batch_y_cls).sum().item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        val_acc = 100 * cls_correct / cls_total
        
        # Calculate additional metrics
        val_accuracy = cls_correct / cls_total
        
        # Update metrics tracker
        metrics_tracker.update(train_loss, val_loss, val_accuracy=val_accuracy, epoch_time=0)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
            paths = config.get_paths_config()
            model_path = paths['models']['braking']
            os.makedirs(model_path, exist_ok=True)
            
            # Prepare metrics to save with model
            final_metrics = {
                'val_accuracy': val_accuracy,
                'best_val_loss': best_val_loss,
                'training_epochs': epoch + 1,
                'model_type': 'multitask_lstm_cnn_attention'
            }
            
            # Save model with metrics
            save_model_checkpoint(model, optimizer, epoch, val_loss, 
                                os.path.join(model_path, "final_multitask_model.pth"), 
                                final_metrics)
            
            # Also save metrics separately as JSON for easy loading
            import json
            with open(os.path.join(model_path, "final_multitask_model_metrics.json"), 'w') as f:
                json.dump(final_metrics, f, indent=2)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    print("Multitask model training complete!")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Braking Intention Models")
    parser.add_argument("--baseline", action="store_true", help="Train baseline LSTM-CNN-Attention model only")
    parser.add_argument("--multitask", action="store_true", help="Train multitask model only")
    parser.add_argument("--ga", action="store_true", help="Run genetic algorithm optimization")
    parser.add_argument("--device", default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    config = get_config()
    
    if args.device == "auto":
        device = config.get_device()
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    set_seed(config.get('system.seed', 42))
    
    try:
        print("Loading braking datasets...")
        dataset_loader = get_dataset_loader()
        dataset_info = dataset_loader.get_dataset_info('braking')
        
        print(f"Using {dataset_info['source']} dataset")
        print(f"Dataset info: {dataset_info}")
        
        (X_train, X_val, X_test, 
         y_int_train, y_int_val, y_int_test,
         y_class_train, y_class_val, y_class_test) = dataset_loader.load_braking_dataset()
        
        y_train = y_class_train
        y_val = y_class_val
        print("Braking dataset loaded successfully")
        
        if y_int_train is None or y_int_val is None:
            print("Intensity labels not found for multitask training")
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        
    except FileNotFoundError as e:
        print(f"Dataset files not found: {e}")
        print("Please run dataset generation first!")
        return
    
    if args.baseline or (not args.multitask and not args.ga):
        result = train_baseline_model(X_train, y_train, X_val, y_val, device, config)
        if result is None:
            print("Baseline training failed due to single-class dataset.")
            return
    
    if args.multitask and y_int_train is not None:
        train_multitask_model(X_train, y_train, y_int_train, X_val, y_val, y_int_val, device, config)
    elif args.multitask:
        print("Multitask training requires intensity labels")
    
    if args.ga:
        print("Running Genetic Algorithm Optimization...")
        os.chdir("modules/braking/models")
        from modules.braking.models.genetic_algorithm_optimizer import GeneticAlgorithmOptimizer
        ga_optimizer = GeneticAlgorithmOptimizer()
        ga_optimizer.run_optimization()
        os.chdir("../../../")
    
    print("All training completed!")


if __name__ == "__main__":
    main()

