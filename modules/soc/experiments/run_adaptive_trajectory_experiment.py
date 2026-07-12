"""B4 driver: expose the existing adaptive-ensemble trajectory/ablation functionality as a
runnable experiment (roadmap B4 steps 2-3: "log weights per timestep across the test cycle",
"add fixed-vs-adaptive ablation").

Trains the adaptive ensemble via the existing, UNMODIFIED
AdaptiveEnsembleSoC.train_ensemble() + GAEnsembleOptimizer.optimize_weights() (the same two
steps train_adaptive_ensemble() already performs), picks a cycle CONFIRMED to be in the B5
leak-free TEST split (closing the B4-audit gap where the trajectory was previously run on a
cycle never confirmed held-out), and calls the existing run_adaptive_trajectory() /
plot_weight_trajectory() / compute_fixed_vs_adaptive_ablation() to produce:
  - assets/img/weight_trajectory.png
  - modules/soc/models/weight_trajectory.csv
  - modules/soc/models/ensemble_ablation.csv

This script adds no new adaptation logic - it only wires together existing, unmodified
functions/classes from adaptive_ensemble.py and battery_rls_identification.py.

Usage:
  python modules/soc/experiments/run_adaptive_trajectory_experiment.py
  python modules/soc/experiments/run_adaptive_trajectory_experiment.py --seed 1 --train-subset 5000
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from shared.dataset_loader import get_dataset_loader
from shared.train_utils import set_seed
from modules.soc.models.adaptive_ensemble import (
    AdaptiveEnsembleSoC, GAEnsembleOptimizer, CoulombCountingReferenceSignal,
    run_adaptive_trajectory, plot_weight_trajectory, compute_fixed_vs_adaptive_ablation,
)
from modules.soc.models.battery_rls_identification import list_cycle_folders, load_cycle


def get_held_out_test_cycle(dataset_loader):
    """Pick the first real cycle confirmed to be in the B5 leak-free TEST split."""
    split_meta = dataset_loader.load_soc_split_metadata()
    test_session_ids = set(split_meta["test"]["session_id"].tolist())

    for folder_path in list_cycle_folders():
        folder_type = os.path.basename(os.path.dirname(folder_path))
        folder_name = os.path.basename(folder_path)
        session_id = f"{folder_type}/{folder_name}"
        if session_id not in test_session_ids:
            continue
        cycle = load_cycle(folder_path)
        if cycle is not None:
            return cycle, session_id

    raise RuntimeError(
        "No readable cycle found in the B5 test split - cannot run a confirmed held-out "
        "trajectory. Check modules/soc/data/session_id_test_real.npy exists and is current."
    )


def main():
    parser = argparse.ArgumentParser(
        description="B4: adaptive-ensemble weight trajectory + fixed-vs-adaptive ablation "
                     "on a confirmed held-out (B5 test-split) cycle."
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed override, for multi-seed experiments (SV-1).")
    parser.add_argument("--train-subset", type=int, default=30000,
                        help="Training subset size for the ensemble sub-models (matches "
                             "adaptive_ensemble.py's existing train_adaptive_ensemble() default).")
    parser.add_argument("--ga-population", type=int, default=10,
                        help="GA population size for offline weight tuning (matches "
                             "adaptive_ensemble.py's existing default).")
    parser.add_argument("--ga-generations", type=int, default=5,
                        help="GA generations for offline weight tuning (matches "
                             "adaptive_ensemble.py's existing default).")
    parser.add_argument("--step", type=int, default=10,
                        help="Window step for the trajectory walk (matches "
                             "run_adaptive_trajectory()'s existing default).")
    parser.add_argument("--figure-path", type=str, default="assets/img/weight_trajectory.png")
    parser.add_argument("--trajectory-csv", type=str,
                        default="modules/soc/models/weight_trajectory.csv")
    parser.add_argument("--ablation-csv", type=str,
                        default="modules/soc/models/ensemble_ablation.csv")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_loader = get_dataset_loader()
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_loader.load_soc_dataset()

    subset_size = min(args.train_subset, len(X_train))
    idx = np.random.choice(len(X_train), subset_size, replace=False)
    X_train_sub, y_train_sub = X_train[idx], y_train[idx]
    print(f"Training ensemble sub-models on {subset_size} windows (seed={args.seed})...")

    # Existing, unmodified training + offline GA weight tuning - the same two steps
    # train_adaptive_ensemble() already performs. No new adaptation logic here.
    ensemble = AdaptiveEnsembleSoC()
    ensemble.train_ensemble(X_train_sub, y_train_sub, X_val, y_val)
    ga_optimizer = GAEnsembleOptimizer(population_size=args.ga_population, generations=args.ga_generations)
    ga_optimizer.optimize_weights(ensemble, X_val, y_val)
    print(f"Offline-tuned starting weights: {ensemble.weights.as_array()}")

    cycle, session_id = get_held_out_test_cycle(dataset_loader)
    print(f"Using confirmed held-out TEST-split cycle: {session_id} ({len(cycle['t'])} samples)")

    log_df = run_adaptive_trajectory(
        ensemble, cycle, reference_signal=CoulombCountingReferenceSignal(), step=args.step,
    )
    print(f"Trajectory: {len(log_df)} steps")

    plot_weight_trajectory(log_df, args.figure_path)  # creates its own output directory

    os.makedirs(os.path.dirname(args.trajectory_csv), exist_ok=True)
    log_df.to_csv(args.trajectory_csv, index=False)
    print(f"Saved: {args.trajectory_csv}")

    ablation_df = compute_fixed_vs_adaptive_ablation(log_df)
    ablation_df.to_csv(args.ablation_csv, index=False)
    print(f"Saved: {args.ablation_csv}")
    print(ablation_df.to_string(index=False))


if __name__ == "__main__":
    main()
