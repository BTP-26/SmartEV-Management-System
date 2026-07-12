
# B6: Weighted-Objective Genetic Algorithm Optimizer for SoC Estimation
#
# This was previously described as "multi-objective" (class/function names, prints, output
# files all said "multi_objective"). In fact this is a fixed weighted-sum scalarization
# (paper Eq. 7: f = w1*f1 + w2*f2 + w3*f3, w=[0.5, 0.3, 0.2], matching this file's
# w_rmse/w_efficiency/w_robustness defaults exactly) - selection/elitism (_tournament())
# uses only this one combined scalar. There is no Pareto front, no non-dominated sorting,
# no NSGA-II anywhere in this file, even though NSGA-II is cited in the paper's related
# work. Renamed throughout for honesty (Reviewer 3.3). Optimizes a single scalarized
# objective built from three real, individually-tracked-but-not-Pareto-ranked terms: RMSE,
# computational efficiency (inference time), and temperature robustness.
#
# See run_weight_sensitivity_sweep() for how the specific choice of weights [0.5, 0.3, 0.2]
# affects which hyperparameters end up selected (roadmap B6 step 2).

import json
import math
import os
import sys
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.soc.models.lstm_cnn_attention_soc import (
    LSTMCNNAttentionSoC, train_soc_model, evaluate_soc_model
)
from shared.dataset_loader import get_dataset_loader


@dataclass
class WeightedObjectiveSoCHyperParams:
    """Extended hyperparameters for the weighted-objective GA search."""
    learning_rate: float
    batch_size: int
    lstm_hidden: int
    num_lstm_layers: int
    dropout_rate: float
    cnn_channels: int
    # Additional parameters for efficiency optimization
    attention_heads: int
    use_layer_norm: bool
    bidirectional: bool
    
    def as_dict(self) -> Dict:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "lstm_hidden": self.lstm_hidden,
            "num_lstm_layers": self.num_lstm_layers,
            "dropout_rate": self.dropout_rate,
            "cnn_channels": self.cnn_channels,
            "attention_heads": self.attention_heads,
            "use_layer_norm": self.use_layer_norm,
            "bidirectional": self.bidirectional,
        }


class WeightedObjectiveSoCGAOptimizer:
    """Weighted-objective GA optimizer for SoC estimation (fixed weighted-sum
    scalarization, not Pareto/NSGA-II - see module docstring)."""
    
    def __init__(
        self,
        X_train, y_train, X_val, y_val,
        population_size: int = 2,
        generations: int = 1,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        max_epochs: int = 5,
        device=None,
        # Objective weights
        w_rmse: float = 0.5,
        w_efficiency: float = 0.3,
        w_robustness: float = 0.2,
    ):
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = min(tournament_size, population_size)
        self.max_epochs = max_epochs
        if device:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Mac GPU
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")        
        # Objective weights (should sum to 1.0)
        self.w_rmse = w_rmse
        self.w_efficiency = w_efficiency
        self.w_robustness = w_robustness
        
        self._cache: Dict[Tuple, Tuple[float, float, float]] = {}
        
        # Extended search space for the weighted-objective GA search
        self.lr_choices = [1e-4, 5e-4, 1e-3, 5e-3]
        self.batch_choices = [32, 64, 128, 256]
        self.hidden_choices = [32, 64, 128, 256]
        self.layers_choices = [1, 2, 3]
        self.dropout_choices = [0.1, 0.2, 0.3, 0.4]
        self.cnn_channel_choices = [16, 32, 64, 128]
        self.attention_heads_choices = [1, 2, 4, 8]
        self.layer_norm_choices = [True, False]
        self.bidirectional_choices = [True, False]
    
    def _random_hp(self) -> WeightedObjectiveSoCHyperParams:
        """Generate random hyperparameters."""
        return WeightedObjectiveSoCHyperParams(
            learning_rate=random.choice(self.lr_choices),
            batch_size=random.choice(self.batch_choices),
            lstm_hidden=random.choice(self.hidden_choices),
            num_lstm_layers=random.choice(self.layers_choices),
            dropout_rate=random.choice(self.dropout_choices),
            cnn_channels=random.choice(self.cnn_channel_choices),
            attention_heads=random.choice(self.attention_heads_choices),
            use_layer_norm=random.choice(self.layer_norm_choices),
            bidirectional=random.choice(self.bidirectional_choices),
        )
    
    def _encode(self, hp: WeightedObjectiveSoCHyperParams) -> Tuple:
        """Encode hyperparameters for caching."""
        return tuple(sorted(hp.as_dict().items()))
    
    def _create_model(self, hp: WeightedObjectiveSoCHyperParams) -> nn.Module:
        """Create model with given hyperparameters."""
        # Create a modified version of the SoC model with additional parameters
        class EnhancedLSTMCNNAttentionSoC(LSTMCNNAttentionSoC):
            def __init__(self, input_dim=3, cnn_channels=64, lstm_hidden=128, 
                        num_lstm_layers=2, dropout=0.2, attention_heads=1,
                        use_layer_norm=False, bidirectional=False):
                super().__init__(input_dim, cnn_channels, lstm_hidden, 
                               num_lstm_layers, dropout)
                
                # Enhanced attention mechanism
                self.attention_heads = attention_heads
                self.use_layer_norm = use_layer_norm
                self.bidirectional = bidirectional
                
                # Modify LSTM for bidirectional support
                if bidirectional:
                    self.lstm = nn.LSTM(
                        input_size=cnn_channels,
                        hidden_size=lstm_hidden,
                        num_layers=num_lstm_layers,
                        dropout=dropout if num_lstm_layers > 1 else 0,
                        batch_first=True,
                        bidirectional=True
                    )
                    # Adjust attention input size for bidirectional
                    attention_input_size = lstm_hidden * 2
                else:
                    attention_input_size = lstm_hidden
                
                # Multi-head attention
                if attention_heads > 1:
                    self.attention = nn.MultiheadAttention(
                        embed_dim=attention_input_size,
                        num_heads=attention_heads,
                        dropout=dropout,
                        batch_first=True
                    )
                else:
                    # Keep original attention for single head
                    self.attention = nn.Linear(attention_input_size, 1)
                
                # Layer normalization
                if use_layer_norm:
                    self.layer_norm = nn.LayerNorm(attention_input_size)
                
                # Adjust final layers for bidirectional
                final_input_size = lstm_hidden * 2 if bidirectional else lstm_hidden
                self.regressor = nn.Sequential(
                    nn.Linear(final_input_size, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1)
                )
            
            def forward(self, x):
                # CNN feature extraction
                x = x.transpose(1, 2)  # (batch, channels, seq_len)
                x = self.cnn(x)
                x = x.transpose(1, 2)  # (batch, seq_len, channels)
                
                # LSTM processing
                x, _ = self.lstm(x)
                
                # Layer normalization
                if hasattr(self, 'layer_norm'):
                    x = self.layer_norm(x)
                
                # Attention mechanism
                if self.attention_heads > 1:
                    # Multi-head attention
                    attn_output, _ = self.attention(x, x, x)
                    # Global average pooling
                    context = torch.mean(attn_output, dim=1)
                else:
                    # Original attention mechanism
                    weights = torch.softmax(self.attention(x), dim=1)
                    context = torch.sum(weights * x, dim=1)
                
                # Final regression
                output = self.regressor(context)
                return output.squeeze(-1)
        
        return EnhancedLSTMCNNAttentionSoC(
            cnn_channels=hp.cnn_channels,
            lstm_hidden=hp.lstm_hidden,
            num_lstm_layers=hp.num_lstm_layers,
            dropout=hp.dropout_rate,
            attention_heads=hp.attention_heads,
            use_layer_norm=hp.use_layer_norm,
            bidirectional=hp.bidirectional,
        )
    
    def _measure_inference_time(self, model: nn.Module, n_samples: int = 100) -> float:
        """Measure average inference time per sample."""
        model.eval()
        
        # Create test batch
        test_batch = torch.randn(n_samples, 50, 3).to(self.device)
        
        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_batch[:1])
        
        # Measure inference time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):  # Average over multiple runs
                _ = model(test_batch)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time_per_sample = (end_time - start_time) / (10 * n_samples)
        return avg_time_per_sample
    
    def _evaluate_robustness(self, model: nn.Module) -> float:
        """Evaluate model robustness across different temperature conditions."""
        model.eval()
        
        # Simulate different temperature conditions by adding noise
        # Temperature is typically the 3rd feature in battery data
        robustness_scores = []
        
        temperature_variations = [
            0.0,    # Normal condition
            0.1,    # Slight temperature variation
            -0.1,   # Slight temperature variation
            0.2,    # Moderate temperature variation
            -0.2,   # Moderate temperature variation
        ]
        
        with torch.no_grad():
            base_batch = torch.randn(50, 50, 3).to(self.device)
            base_output = model(base_batch)
            
            for temp_var in temperature_variations:
                # Add temperature variation to the 3rd feature
                temp_batch = base_batch.clone()
                temp_batch[:, :, 2] += temp_var  # Add to temperature channel
                
                temp_output = model(temp_batch)
                
                # Calculate relative change
                relative_change = torch.mean(torch.abs(temp_output - base_output) / torch.abs(base_output))
                robustness_scores.append(relative_change.item())
        
        # Lower average relative change = higher robustness
        avg_robustness_loss = np.mean(robustness_scores)
        return avg_robustness_loss
    
    def _weighted_fitness(self, hp: WeightedObjectiveSoCHyperParams) -> Tuple[float, float, float, float]:
        """Calculate weighted-sum fitness (RMSE, efficiency, robustness, combined). The
        three individual terms are tracked for reporting, but only `combined` (the
        weighted sum) drives selection/elitism - see _tournament()."""
        key = self._encode(hp)
        if key in self._cache:
            return self._cache[key]
        
        print(f"  Evaluating: lr={hp.learning_rate:.4f}, hidden={hp.lstm_hidden}, "
              f"layers={hp.num_lstm_layers}, cnn={hp.cnn_channels}, "
              f"batch={hp.batch_size}, heads={hp.attention_heads}")
        
        try:
            # Create and train model
            model = self._create_model(hp)
            model.to(self.device)
            
            # Train model
            _, history = train_soc_model(
                model, self.X_train, self.y_train, self.X_val, self.y_val,
                lr=hp.learning_rate, batch_size=hp.batch_size,
                epochs=self.max_epochs, patience=5,
                device=self.device,
                save_path="modules/soc/models/tmp_weighted_obj_soc.pth",
            )
            
            # Objective 1: RMSE (lower is better)
            best_rmse = min(history["val_rmse"])
            rmse_fitness = -best_rmse  # Negative for maximization
            
            # Objective 2: Computational efficiency (lower inference time is better)
            avg_inference_time = self._measure_inference_time(model)
            efficiency_fitness = -avg_inference_time  # Negative for maximization
            
            # Objective 3: Temperature robustness (lower robustness loss is better)
            robustness_loss = self._evaluate_robustness(model)
            robustness_fitness = -robustness_loss  # Negative for maximization
            
            # Combined fitness (weighted sum)
            combined_fitness = (
                self.w_rmse * rmse_fitness +
                self.w_efficiency * efficiency_fitness +
                self.w_robustness * robustness_fitness
            )
            
            fitness_tuple = (combined_fitness, rmse_fitness, efficiency_fitness, robustness_fitness)
            self._cache[key] = fitness_tuple
            
            print(f"    RMSE: {best_rmse:.4f}, Time: {avg_inference_time*1000:.2f}ms, "
                  f"Robustness: {robustness_loss:.4f}, Combined: {combined_fitness:.4f}")
            
            return fitness_tuple
            
        except Exception as e:
            print(f"    Error evaluating hyperparameters: {e}")
            # Return worst possible fitness
            worst_fitness = (-float('inf'), -float('inf'), -float('inf'), -float('inf'))
            self._cache[key] = worst_fitness
            return worst_fitness
    
    def _tournament(self, pop, fits) -> WeightedObjectiveSoCHyperParams:
        """Tournament selection based on combined fitness."""
        idx = random.sample(range(len(pop)), self.tournament_size)
        best = max(idx, key=lambda i: fits[i][0])  # Use combined fitness
        return pop[best]
    
    def _crossover(self, p1: WeightedObjectiveSoCHyperParams, p2: WeightedObjectiveSoCHyperParams):
        """Crossover operation for hyperparameters."""
        g1 = list(p1.as_dict().values())
        g2 = list(p2.as_dict().values())
        
        # Uniform crossover
        child1 = []
        child2 = []
        for v1, v2 in zip(g1, g2):
            if random.random() < 0.5:
                child1.append(v1)
                child2.append(v2)
            else:
                child1.append(v2)
                child2.append(v1)
        
        def build(g):
            return WeightedObjectiveSoCHyperParams(
                learning_rate=g[0], batch_size=int(g[1]),
                lstm_hidden=int(g[2]), num_lstm_layers=int(g[3]),
                dropout_rate=float(g[4]), cnn_channels=int(g[5]),
                attention_heads=int(g[6]), use_layer_norm=bool(g[7]),
                bidirectional=bool(g[8]),
            )
        
        return build(child1), build(child2)
    
    def _mutate(self, hp: WeightedObjectiveSoCHyperParams) -> WeightedObjectiveSoCHyperParams:
        """Mutation operation for hyperparameters."""
        d = hp.as_dict()
        
        if random.random() < self.mutation_rate:
            d["learning_rate"] = random.choice(self.lr_choices)
        if random.random() < self.mutation_rate:
            d["batch_size"] = random.choice(self.batch_choices)
        if random.random() < self.mutation_rate:
            d["lstm_hidden"] = random.choice(self.hidden_choices)
        if random.random() < self.mutation_rate:
            d["num_lstm_layers"] = random.choice(self.layers_choices)
        if random.random() < self.mutation_rate:
            d["dropout_rate"] = random.choice(self.dropout_choices)
        if random.random() < self.mutation_rate:
            d["cnn_channels"] = random.choice(self.cnn_channel_choices)
        if random.random() < self.mutation_rate:
            d["attention_heads"] = random.choice(self.attention_heads_choices)
        if random.random() < self.mutation_rate:
            d["use_layer_norm"] = random.choice(self.layer_norm_choices)
        if random.random() < self.mutation_rate:
            d["bidirectional"] = random.choice(self.bidirectional_choices)
        
        return WeightedObjectiveSoCHyperParams(**d)
    
    def run(self):
        """Run the weighted-objective GA optimization loop."""
        print(f"Starting Weighted-Objective GA Optimization...")
        print(f"Population: {self.population_size}, Generations: {self.generations}")
        print(f"Objective weights - RMSE: {self.w_rmse}, Efficiency: {self.w_efficiency}, Robustness: {self.w_robustness}")
        
        # Initialize population
        population = [self._random_hp() for _ in range(self.population_size)]
        fitnesses = [self._weighted_fitness(hp) for hp in population]
        
        # Track best individuals for each objective
        best_combined_idx = int(np.argmax([f[0] for f in fitnesses]))
        best_combined_hp = population[best_combined_idx]
        best_combined_fitness = fitnesses[best_combined_idx]
        
        best_rmse_idx = int(np.argmax([f[1] for f in fitnesses]))
        best_rmse_hp = population[best_rmse_idx]
        best_rmse_fitness = fitnesses[best_rmse_idx]
        
        best_efficiency_idx = int(np.argmax([f[2] for f in fitnesses]))
        best_efficiency_hp = population[best_efficiency_idx]
        best_efficiency_fitness = fitnesses[best_efficiency_idx]
        
        best_robustness_idx = int(np.argmax([f[3] for f in fitnesses]))
        best_robustness_hp = population[best_robustness_idx]
        best_robustness_fitness = fitnesses[best_robustness_idx]
        
        # Track fitness history
        history = {
            'combined': [best_combined_fitness[0]],
            'rmse': [best_rmse_fitness[1]],
            'efficiency': [best_efficiency_fitness[2]],
            'robustness': [best_robustness_fitness[3]]
        }
        
        print(f"\nGeneration 0 - Best Combined: {best_combined_fitness[0]:.4f}, "
              f"Best RMSE: {-best_rmse_fitness[1]:.4f}")
        
        # Evolution loop
        for gen in range(1, self.generations + 1):
            # Create new population with elitism
            new_pop = [best_combined_hp]  # Elitism: keep best combined
            
            while len(new_pop) < self.population_size:
                # Selection
                p1 = self._tournament(population, fitnesses)
                p2 = self._tournament(population, fitnesses)
                
                # Crossover
                c1, c2 = self._crossover(p1, p2)
                
                # Mutation
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.population_size:
                    new_pop.append(self._mutate(c2))
            
            # Evaluate new population
            population = new_pop
            fitnesses = [self._weighted_fitness(hp) for hp in population]
            
            # Update best individuals
            gen_best_combined_idx = int(np.argmax([f[0] for f in fitnesses]))
            if fitnesses[gen_best_combined_idx][0] > best_combined_fitness[0]:
                best_combined_fitness = fitnesses[gen_best_combined_idx]
                best_combined_hp = population[gen_best_combined_idx]
            
            gen_best_rmse_idx = int(np.argmax([f[1] for f in fitnesses]))
            if fitnesses[gen_best_rmse_idx][1] > best_rmse_fitness[1]:
                best_rmse_fitness = fitnesses[gen_best_rmse_idx]
                best_rmse_hp = population[gen_best_rmse_idx]
            
            gen_best_efficiency_idx = int(np.argmax([f[2] for f in fitnesses]))
            if fitnesses[gen_best_efficiency_idx][2] > best_efficiency_fitness[2]:
                best_efficiency_fitness = fitnesses[gen_best_efficiency_idx]
                best_efficiency_hp = population[gen_best_efficiency_idx]
            
            gen_best_robustness_idx = int(np.argmax([f[3] for f in fitnesses]))
            if fitnesses[gen_best_robustness_idx][3] > best_robustness_fitness[3]:
                best_robustness_fitness = fitnesses[gen_best_robustness_idx]
                best_robustness_hp = population[gen_best_robustness_idx]
            
            # Update history
            history['combined'].append(best_combined_fitness[0])
            history['rmse'].append(best_rmse_fitness[1])
            history['efficiency'].append(best_efficiency_fitness[2])
            history['robustness'].append(best_robustness_fitness[3])
            
            print(f"Generation {gen}/{self.generations} - "
                  f"Best Combined: {best_combined_fitness[0]:.4f}, "
                  f"Best RMSE: {-best_rmse_fitness[1]:.4f}")
        
        # Prepare results
        results = {
            'best_combined': {
                'hyperparams': best_combined_hp.as_dict(),
                'fitness': best_combined_fitness,
                'rmse': -best_combined_fitness[1],
                'inference_time': -best_combined_fitness[2],
                'robustness_loss': -best_combined_fitness[3]
            },
            'best_rmse': {
                'hyperparams': best_rmse_hp.as_dict(),
                'fitness': best_rmse_fitness,
                'rmse': -best_rmse_fitness[1],
                'inference_time': -best_rmse_fitness[2],
                'robustness_loss': -best_rmse_fitness[3]
            },
            'best_efficiency': {
                'hyperparams': best_efficiency_hp.as_dict(),
                'fitness': best_efficiency_fitness,
                'rmse': -best_efficiency_fitness[1],
                'inference_time': -best_efficiency_fitness[2],
                'robustness_loss': -best_efficiency_fitness[3]
            },
            'best_robustness': {
                'hyperparams': best_robustness_hp.as_dict(),
                'fitness': best_robustness_fitness,
                'rmse': -best_robustness_fitness[1],
                'inference_time': -best_robustness_fitness[2],
                'robustness_loss': -best_robustness_fitness[3]
            },
            'history': history
        }
        
        return results


def run_weighted_objective_soc_ga():
    """Run the weighted-objective GA optimization for SoC estimation (Eq. 7's
    fixed weighted-sum scalarization, not Pareto/NSGA-II - see module docstring)."""
    # Load the same real, [0,1]-scaled dataset the deployed model uses, instead of the
    # orphaned *_soc.npy convention that no preprocessing script in the repo produced.
    dataset_loader = get_dataset_loader()
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_loader.load_soc_dataset()

    # SPEEDUP: sample smaller dataset (full 50-step window length is kept, matching the
    # deployed model and this file's own _measure_inference_time/_evaluate_robustness,
    # which already assume seq_len=50)
    sample_size = min(50000, len(X_train))
    idx = np.random.choice(len(X_train), sample_size, replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    ga = MultiObjectiveSoCGAOptimizer(
        X_train, y_train, X_val, y_val,
        population_size=2, generations=2,
        mutation_rate=0.2, max_epochs=3,
        w_rmse=0.5, w_efficiency=0.3, w_robustness=0.2,
    )
    
    results = ga.run()

    print(f"\n WEIGHTED-OBJECTIVE GA OPTIMIZATION RESULTS")
    
    # Best combined (primary result)
    best_combined = results['best_combined']
    print(f"\n🏆 BEST COMBINED SOLUTION:")
    print(f"   RMSE: {best_combined['rmse']:.4f}")
    print(f"   Inference Time: {best_combined['inference_time']*1000:.2f}ms")
    print(f"   Robustness Loss: {best_combined['robustness_loss']:.4f}")
    print(f"   Hyperparameters: {best_combined['hyperparams']}")
    
    # Best individual objectives
    print(f"\n🎯 BEST INDIVIDUAL OBJECTIVES:")
    print(f"   Best RMSE: {results['best_rmse']['rmse']:.4f}")
    print(f"   Best Efficiency: {results['best_efficiency']['inference_time']*1000:.2f}ms")
    print(f"   Best Robustness: {results['best_robustness']['robustness_loss']:.4f}")

    # Convert numpy types to regular Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    serializable_results = convert_numpy_types(results)
    
    # Save results
    os.makedirs("modules/soc/models", exist_ok=True)
    with open("modules/soc/models/weighted_objective_ga_results.json", "w") as f:
        json.dump(serializable_results, f, indent=4)
    print("Results saved: modules/soc/models/weighted_objective_ga_results.json")

    # Plot fitness curves
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(results['history']['combined'], marker="o")
    plt.title("Combined Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot([-f for f in results['history']['rmse']], marker="o")
    plt.title("Best RMSE")
    plt.xlabel("Generation")
    plt.ylabel("RMSE")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot([-f for f in results['history']['efficiency']], marker="o")
    plt.title("Best Inference Time")
    plt.xlabel("Generation")
    plt.ylabel("Time (s)")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot([-f for f in results['history']['robustness']], marker="o")
    plt.title("Best Robustness Loss")
    plt.xlabel("Generation")
    plt.ylabel("Robustness Loss")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("assets/img", exist_ok=True)
    plt.savefig("assets/img/weighted_objective_ga_fitness.png", dpi=150)
    plt.close()
    print("Fitness plots saved: assets/img/weighted_objective_ga_fitness.png")

    return results


def run_weight_sensitivity_sweep(
    X_train, y_train, X_val, y_val,
    n_candidates: int = 6,
    weight_triples: Optional[List[Tuple[str, float, float, float]]] = None,
    max_epochs: int = 3,
    device=None,
) -> pd.DataFrame:
    """B6 step 2: how much does the paper's weight choice (Eq. 7: w=[0.5, 0.3, 0.2])
    actually matter? Trains a small candidate pool ONCE (the only expensive part - real
    but lightweight-scale training, `max_epochs` matching this file's existing small
    defaults), then re-weights each candidate's already-computed (rmse, efficiency,
    robustness) fitness under several weight triples ARITHMETICALLY (no retraining). This
    isolates weight-sensitivity from GA search-noise and is far cheaper than re-running a
    full GA search per triple. Returns one row per weight triple: which candidate it
    selects and that candidate's real RMSE/inference-time/robustness."""
    weight_triples = weight_triples or [
        ("paper (0.5/0.3/0.2)", 0.5, 0.3, 0.2),
        ("rmse-heavy", 0.8, 0.1, 0.1),
        ("efficiency-heavy", 0.1, 0.8, 0.1),
        ("robustness-heavy", 0.1, 0.1, 0.8),
        ("equal", 1 / 3, 1 / 3, 1 / 3),
    ]

    optimizer = WeightedObjectiveSoCGAOptimizer(
        X_train, y_train, X_val, y_val, max_epochs=max_epochs, device=device,
    )
    candidates = [optimizer._random_hp() for _ in range(n_candidates)]
    print(f"Training {n_candidates} candidates once for the weight-sensitivity sweep...")
    # (combined, rmse_fitness, efficiency_fitness, robustness_fitness) per candidate -
    # `combined` here uses the optimizer's own default weights and is unused below; the
    # sweep recomputes combined scores itself from the raw per-objective terms for every
    # weight triple.
    raw_fitness = [optimizer._weighted_fitness(hp) for hp in candidates]

    rows = []
    for label, w_rmse, w_eff, w_rob in weight_triples:
        combined_scores = [w_rmse * f[1] + w_eff * f[2] + w_rob * f[3] for f in raw_fitness]
        best_idx = int(np.argmax(combined_scores))
        best_hp = candidates[best_idx]
        best_f = raw_fitness[best_idx]
        rows.append({
            "weight_label": label, "w_rmse": w_rmse, "w_efficiency": w_eff, "w_robustness": w_rob,
            "selected_candidate_idx": best_idx,
            "rmse": -best_f[1], "inference_time_ms": -best_f[2] * 1000, "robustness_loss": -best_f[3],
            "hyperparams": best_hp.as_dict(),
        })
        print(f"  {label}: candidate #{best_idx} selected "
              f"(RMSE={-best_f[1]:.4f}, latency={-best_f[2]*1000:.2f}ms, robustness={-best_f[3]:.4f})")

    n_distinct = len(set(row["selected_candidate_idx"] for row in rows))
    print(f"\n{n_distinct}/{len(weight_triples)} weight triples selected a distinct candidate "
          f"({'weights matter' if n_distinct > 1 else 'same candidate won under every triple tried'}).")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_weighted_objective_soc_ga()
