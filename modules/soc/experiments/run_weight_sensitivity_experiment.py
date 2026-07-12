"""B6 driver: expose the existing GA weight-sensitivity sweep as a runnable, results-saving
experiment (roadmap B6 step 2: "sweep the 3 weights and report a sensitivity table").

run_weight_sensitivity_sweep() (modules/soc/models/multi_objective_ga_optimizer.py) already
implements the sweep and returns a DataFrame, but no existing entry point calls it or saves
its output. This script calls it (unmodified) and persists the result to CSV.

This script adds no new GA/search logic - it only wires together the existing, unmodified
run_weight_sensitivity_sweep() with dataset loading and a CSV writer.

Usage:
  python modules/soc/experiments/run_weight_sensitivity_experiment.py
  python modules/soc/experiments/run_weight_sensitivity_experiment.py --seed 1 --n-candidates 12
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
from modules.soc.models.multi_objective_ga_optimizer import run_weight_sensitivity_sweep


def main():
    parser = argparse.ArgumentParser(
        description="B6: GA weight-sensitivity sweep (Eq. 7's [0.5, 0.3, 0.2] vs. alternative "
                     "weight triples), saved to CSV."
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed override, for multi-seed experiments (SV-1).")
    parser.add_argument("--train-subset", type=int, default=50000,
                        help="Training subset size for the candidate pool (matches "
                             "multi_objective_ga_optimizer.py's existing "
                             "run_weighted_objective_soc_ga() default).")
    parser.add_argument("--n-candidates", type=int, default=6,
                        help="Number of hyperparameter candidates to train once and re-weight "
                             "(matches run_weight_sensitivity_sweep()'s existing default).")
    parser.add_argument("--max-epochs", type=int, default=3,
                        help="Training epochs per candidate (matches "
                             "run_weight_sensitivity_sweep()'s existing default).")
    parser.add_argument("--output", type=str,
                        default="modules/soc/models/weight_sensitivity_table.csv")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_loader = get_dataset_loader()
    X_train, X_val, X_test, y_train, y_val, y_test = dataset_loader.load_soc_dataset()

    subset_size = min(args.train_subset, len(X_train))
    idx = np.random.choice(len(X_train), subset_size, replace=False)
    X_train_sub, y_train_sub = X_train[idx], y_train[idx]
    print(f"Sweeping over {args.n_candidates} candidates trained on {subset_size} windows "
          f"(seed={args.seed})...")

    # Existing, unmodified sweep - this script only adds dataset loading + persistence.
    sweep_df = run_weight_sensitivity_sweep(
        X_train_sub, y_train_sub, X_val, y_val,
        n_candidates=args.n_candidates, max_epochs=args.max_epochs,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    sweep_df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(sweep_df.drop(columns=["hyperparams"]).to_string(index=False))


if __name__ == "__main__":
    main()
