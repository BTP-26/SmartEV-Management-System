import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from .multitask_lstm_cnn_attention import MultitaskLSTMCNNAttention


class MultitaskHardDataset(Dataset):

    def __init__(self, X: np.ndarray, y_class: np.ndarray, y_intensity: np.ndarray):
        assert len(X) == len(y_class) == len(
            y_intensity
        ), "Inconsistent dataset lengths for multitask data."
        self.X = X.astype(np.float32)
        self.y_class = y_class.astype(np.int64)
        self.y_intensity = y_intensity.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])
        y_cls = torch.tensor(self.y_class[idx], dtype=torch.long)
        y_int = torch.tensor(self.y_intensity[idx], dtype=torch.float32)
        return x, y_cls, y_int


@dataclass
class HyperParams:
    learning_rate: float
    batch_size: int
    lstm_hidden_size: int
    num_lstm_layers: int
    dropout_rate: float
    cnn_filters: int

    def as_dict(self) -> Dict:
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "lstm_hidden_size": self.lstm_hidden_size,
            "num_lstm_layers": self.num_lstm_layers,
            "dropout_rate": self.dropout_rate,
            "cnn_filters": self.cnn_filters,
        }


class GeneticAlgorithmOptimizer:

    def __init__(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        population_size: int = 20,
        generations: int = 3,
        mutation_rate: float = 0.1,
        tournament_size: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_epochs: int = 5,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.device = torch.device(device)
        self.max_epochs = max_epochs

        # Hyperparameter search space
        self.lr_min = 1e-4
        self.lr_max = 1e-2
        self.batch_size_choices = [16, 32, 64, 128]
        self.lstm_hidden_choices = [64, 128, 256]
        self.num_lstm_layers_choices = [1, 2, 3]
        self.dropout_min = 0.1
        self.dropout_max = 0.5
        self.cnn_filter_choices = [32, 64, 128]

        # Cache to avoid retraining identical configurations
        self._fitness_cache: Dict[Tuple, float] = {}

    # GA Core 
    def _random_hparams(self) -> HyperParams:
        # Sample learning rate log-uniformly
        log_lr = random.uniform(math.log10(self.lr_min), math.log10(self.lr_max))
        lr = 10 ** log_lr
        return HyperParams(
            learning_rate=lr,
            batch_size=random.choice(self.batch_size_choices),
            lstm_hidden_size=random.choice(self.lstm_hidden_choices),
            num_lstm_layers=random.choice(self.num_lstm_layers_choices),
            dropout_rate=random.uniform(self.dropout_min, self.dropout_max),
            cnn_filters=random.choice(self.cnn_filter_choices),
        )

    def _encode(self, hp: HyperParams) -> Tuple:
        #Immutable key for caching.#
        d = hp.as_dict()
        return tuple(sorted(d.items()))

    def _tournament_select(self, population: List[HyperParams], fitnesses: List[float]) -> HyperParams:
        indices = random.sample(range(len(population)), self.tournament_size)
        best_idx = max(indices, key=lambda i: fitnesses[i])
        return population[best_idx]

    def _crossover(self, parent1: HyperParams, parent2: HyperParams) -> Tuple[HyperParams, HyperParams]:
        genes1 = list(parent1.as_dict().values())
        genes2 = list(parent2.as_dict().values())
        point = random.randint(1, len(genes1) - 1)
        child1_genes = genes1[:point] + genes2[point:]
        child2_genes = genes2[:point] + genes1[point:]

        def build_hp(genes: List) -> HyperParams:
            return HyperParams(
                learning_rate=float(genes[0]),
                batch_size=int(genes[1]),
                lstm_hidden_size=int(genes[2]),
                num_lstm_layers=int(genes[3]),
                dropout_rate=float(genes[4]),
                cnn_filters=int(genes[5]),
            )

        return build_hp(child1_genes), build_hp(child2_genes)

    def _mutate(self, hp: HyperParams) -> HyperParams:
        d = hp.as_dict()

        # Each gene mutates independently
        if random.random() < self.mutation_rate:
            log_lr = random.uniform(math.log10(self.lr_min), math.log10(self.lr_max))
            d["learning_rate"] = 10 ** log_lr

        if random.random() < self.mutation_rate:
            d["batch_size"] = random.choice(self.batch_size_choices)

        if random.random() < self.mutation_rate:
            d["lstm_hidden_size"] = random.choice(self.lstm_hidden_choices)

        if random.random() < self.mutation_rate:
            d["num_lstm_layers"] = random.choice(self.num_lstm_layers_choices)

        if random.random() < self.mutation_rate:
            d["dropout_rate"] = random.uniform(self.dropout_min, self.dropout_max)

        if random.random() < self.mutation_rate:
            d["cnn_filters"] = random.choice(self.cnn_filter_choices)

        return HyperParams(**d)

    # Model Training / Evaluation
    def _build_model(self, hp: HyperParams) -> nn.Module:
        model = MultitaskLSTMCNNAttention(
            input_dim=3,
            cnn_channels=hp.cnn_filters,
            lstm_hidden=hp.lstm_hidden_size,
            num_lstm_layers=hp.num_lstm_layers,
            dropout_rate=hp.dropout_rate,
        )
        return model.to(self.device)

    def _fitness(self, hp: HyperParams) -> float:
        key = self._encode(hp)
        if key in self._fitness_cache:
            return self._fitness_cache[key]

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=hp.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=hp.batch_size,
            shuffle=False,
            drop_last=False,
        )

        model = self._build_model(hp)
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        # Classification-focused loss weighting
        lambda_cls = 0.8
        lambda_reg = 1.0 - lambda_cls

        optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)

        for _ in range(self.max_epochs):
            model.train()
            for x, y_cls, y_int in train_loader:
                x = x.to(self.device)
                y_cls = y_cls.to(self.device)
                y_int = y_int.to(self.device)

                optimizer.zero_grad()
                logits, intensity = model(x)

                loss_cls = ce_loss(logits, y_cls)
                loss_reg = mse_loss(intensity, y_int)
                loss = lambda_cls * loss_cls + lambda_reg * loss_reg

                loss.backward()
                optimizer.step()

        # Fitness = validation macro F1
        model.eval()
        all_preds: List[int] = []
        all_targets: List[int] = []

        with torch.no_grad():
            for x, y_cls, _ in val_loader:
                x = x.to(self.device)
                y_cls = y_cls.to(self.device)
                logits, _ = model(x)
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(y_cls.cpu().numpy().tolist())

        f1 = f1_score(all_targets, all_preds, average="macro")
        self._fitness_cache[key] = f1
        return f1

    # Public API 
    def run(self) -> Tuple[HyperParams, float, List[float]]:

        population: List[HyperParams] = [self._random_hparams() for _ in range(self.population_size)]
        fitnesses: List[float] = [self._fitness(ind) for ind in population]

        best_idx = int(np.argmax(fitnesses))
        best_individual = population[best_idx]
        best_fitness = fitnesses[best_idx]
        fitness_curve: List[float] = [best_fitness]

        for gen in range(1, self.generations + 1):
            new_population: List[HyperParams] = []

            # Elitism: always carry the best individual forward
            new_population.append(best_individual)

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)

                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = new_population
            fitnesses = [self._fitness(ind) for ind in population]

            gen_best_idx = int(np.argmax(fitnesses))
            gen_best = population[gen_best_idx]
            gen_best_f1 = fitnesses[gen_best_idx]

            if gen_best_f1 > best_fitness:
                best_fitness = gen_best_f1
                best_individual = gen_best

            fitness_curve.append(best_fitness)
            print(
                f"Generation {gen}/{self.generations} "
                f"- Best macro F1 so far: {best_fitness:.4f}"
            )

        return best_individual, best_fitness, fitness_curve


def load_multitask_hard_datasets(
    data_dir: str = "data",
) -> Tuple[Dataset, Dataset]:
    X_train = np.load(os.path.join(data_dir, "X_train_realistic.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val_realistic.npy"))

    y_class_train = np.load(os.path.join(data_dir, "y_class_train_realistic.npy"))
    y_class_val = np.load(os.path.join(data_dir, "y_class_val_realistic.npy"))

    y_int_train = np.load(os.path.join(data_dir, "y_int_train_realistic.npy"))
    y_int_val = np.load(os.path.join(data_dir, "y_int_val_realistic.npy"))

    train_ds = MultitaskHardDataset(X_train, y_class_train, y_int_train)
    val_ds = MultitaskHardDataset(X_val, y_class_val, y_int_val)

    return train_ds, val_ds


def save_best_hyperparams(
    hparams: HyperParams,
    fitness: float,
    output_path: str = "models/best_ga_hyperparams.json",
) -> None:
    payload = {
        "best_f1_macro": fitness,
        "hyperparams": hparams.as_dict(),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"Best hyperparameters saved to {output_path}")


def plot_fitness_curve(
    fitness_curve: List[float],
    output_path: str = "assets/img/ga_fitness_curve.png",
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(fitness_curve)), fitness_curve, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Best Macro F1 (Validation)")
    plt.title("GA Hyperparameter Optimization - Fitness Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Fitness curve saved to {output_path}")


def run_ga_optimization():
    train_ds, val_ds = load_multitask_hard_datasets()
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    ga = GeneticAlgorithmOptimizer(
        train_dataset=train_ds,
        val_dataset=val_ds,
        population_size=20,
        generations=10,
        mutation_rate=0.15,
        tournament_size=3,
        max_epochs=15,
    )

    best_hp, best_f1, fitness_curve = ga.run()
    save_best_hyperparams(best_hp, best_f1)
    plot_fitness_curve(fitness_curve)


if __name__ == "__main__":
    run_ga_optimization()