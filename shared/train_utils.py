import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
import numpy as np
import os
import time
from typing import Dict, Tuple, Optional


class EarlyStopper:
    """early stopping utility to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def should_stop(self, val_loss: float) -> bool:
        """check if training should stop"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class MetricsTracker:
    """track training and validation metrics"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
        self.val_rmse = []
        self.val_mae = []
        self.epoch_times = []
        
    def update(self, train_loss: float, val_loss: float, val_accuracy: float = None, 
               val_f1: float = None, val_rmse: float = None, val_mae: float = None,
               epoch_time: float = None):
        """update metrics for current epoch"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if val_accuracy is not None:
            self.val_accuracies.append(val_accuracy)
        if val_f1 is not None:
            self.val_f1_scores.append(val_f1)
        if val_rmse is not None:
            self.val_rmse.append(val_rmse)
        if val_mae is not None:
            self.val_mae.append(val_mae)
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
    
    def get_best_epoch(self, metric: str = 'val_loss') -> int:
        """get epoch with best validation metric"""
        if metric == 'val_loss':
            return np.argmin(self.val_losses)
        elif metric == 'val_accuracy' and self.val_accuracies:
            return np.argmax(self.val_accuracies)
        elif metric == 'val_f1' and self.val_f1_scores:
            return np.argmax(self.val_f1_scores)
        else:
            return np.argmin(self.val_losses)
    
    def print_summary(self):
        """print training summary"""
        if not self.val_losses:
            return
            
        best_epoch = self.get_best_epoch()
        print(f"\ntraining summary")
        print(f"total epochs: {len(self.train_losses)}")
        print(f"best epoch: {best_epoch + 1}")
        print(f"best val loss: {min(self.val_losses):.6f}")
        
        if self.val_accuracies:
            print(f"best val accuracy: {max(self.val_accuracies):.4f}")
        if self.val_f1_scores:
            print(f"best val f1: {max(self.val_f1_scores):.4f}")
        if self.val_rmse:
            print(f"best val rmse: {min(self.val_rmse):.4f}")
        if self.epoch_times:
            print(f"total training time: {sum(self.epoch_times):.2f}s")


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """calculate classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted)
    }


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape)
    }


def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         epoch: int, loss: float, path: str, metrics: Dict = None):
    """save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics or {}
    }
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint, path)


def load_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         path: str, device: torch.device) -> Tuple[int, float, Dict]:
    """load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint.get('metrics', {})


def create_data_loaders(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None,
                       batch_size: int = 32, shuffle_train: bool = True) -> Tuple[DataLoader, Optional[DataLoader]]:
    """create pytorch data loaders"""
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32 if y_train.ndim == 1 else torch.long)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32 if y_val.ndim == 1 else torch.long)
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
