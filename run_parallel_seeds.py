#!/usr/bin/env python3
"""
Parallel Seed Runner for Grokking Ablation Studies
Runs multiple seeds simultaneously on the same GPU using torch.multiprocessing.

Usage:
    python run_parallel_seeds.py \
        --conditions baseline full_steerage_v2 \
        --num_seeds 20 \
        --parallel 3 \
        --batch_size 512 \
        --use_amp \
        --results_dir ./results_parallel

This will run 3 seeds at a time, maximizing GPU utilization on RTX 3060.
"""

import argparse
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path
import sys
from typing import Dict, Any

# Import the core experiment runner
from grokking_experiment import run_experiment


def _run_single_seed_task(task):
    """
    Top-level task runner to avoid pickling issues with local functions.
    task = (seed, condition_name, condition_args, base_args, results_dir)
    """
    import copy
    seed, condition_name, condition_args, base_args, results_dir = task

    args = copy.deepcopy(base_args)
    for k, v in condition_args.items():
        setattr(args, k, v)
    args.seed = seed

    # Pool workers are daemonic and cannot spawn DataLoader worker processes.
    # The dataset is tiny (p^2 rows), so num_workers=0 costs nothing here.
    args.num_workers = 0
    args.persistent_workers = False

    run_dir = results_dir / condition_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(run_dir)

    print(f"\n[PID {mp.current_process().pid}] Running: {condition_name} | Seed: {seed}")

    try:
        history = run_experiment(args)
        from run_ablation_experiments import compute_grokking_metrics
        metrics = compute_grokking_metrics(history)
        metrics.update({"condition": condition_name, "seed": seed})

        import numpy as np
        np.savez(run_dir / "history.npz", **{
            k: v for k, v in history.items() if isinstance(v, (list, np.ndarray))
        })
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[PID {mp.current_process().pid}] ✓ Completed: {condition_name} seed {seed}")
        return metrics
    except Exception as e:
        print(f"[PID {mp.current_process().pid}] ✗ ERROR in {condition_name} seed {seed}: {e}")
        error_info = {"condition": condition_name, "seed": seed, "error": str(e)}
        with open(run_dir / "error.txt", "w") as f:
            f.write(str(e))
        return error_info


def main():
    parser = argparse.ArgumentParser(
        description="Run Grokking Ablation Experiments with Parallel Seeds"
    )

    parser.add_argument("--conditions", nargs="+", required=True,
                        help="List of conditions to run")
    parser.add_argument("--num_seeds", type=int, default=10,
                        help="Number of seeds per condition")
    parser.add_argument("--parallel", type=int, default=3,
                        help="Number of seeds to run in parallel (default: 3)")
    parser.add_argument("--results_dir", type=str, default="./results_parallel",
                        help="Directory to store results")

    # Base arguments (passed through to grokking_experiment.py)
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--train_frac", type=float, default=0.45)
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--wd", type=float, default=0.40)
    parser.add_argument("--lr", type=float, default=0.0005)

    # Performance flags
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--compile_model", action="store_true")

    # Steerage mechanism flags
    parser.add_argument("--use_polarity_steering", action="store_true")
    parser.add_argument("--polarity_lambda", type=float, default=0.05)
    parser.add_argument("--use_holonomy_reg", action="store_true")
    parser.add_argument("--holonomy_lambda", type=float, default=0.03)
    parser.add_argument("--use_stabilizer", action="store_true")
    parser.add_argument("--use_mirror_closure", action="store_true")
    parser.add_argument("--mirror_lambda", type=float, default=0.01)
    parser.add_argument("--use_internal_mirror_closure", action="store_true")
    parser.add_argument("--num_internal_agents", type=int, default=4)
    parser.add_argument("--internal_mirror_lambda", type=float, default=0.05)
    parser.add_argument("--use_epistemic_self_improvement", action="store_true")
    parser.add_argument("--epistemic_lambda", type=float, default=0.03)
    parser.add_argument("--epistemic_ema_beta", type=float, default=0.995)
    parser.add_argument("--epistemic_start_step", type=int, default=2000)

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Define experimental conditions (same as run_ablation_experiments.py)
    conditions = {
        "baseline": {
            "use_polarity_steering": False,
            "use_holonomy_reg": False,
            "use_stabilizer": False,
            "use_mirror_closure": False,
            "use_internal_mirror_closure": False,
            "use_epistemic_self_improvement": False,
        },
        "full_steerage_v1": {
            "use_polarity_steering": True,
            "use_holonomy_reg": True,
            "use_stabilizer": True,
            "use_mirror_closure": True,
            "use_internal_mirror_closure": False,
            "use_epistemic_self_improvement": False,
        },
        "full_steerage": {
            "use_polarity_steering": True,
            "use_holonomy_reg": True,
            "use_stabilizer": True,
            "use_mirror_closure": True,
            "use_internal_mirror_closure": True,
            "use_epistemic_self_improvement": False,
            "num_internal_agents": 4,
            "internal_mirror_lambda": 0.05,
        },
        "full_steerage_v2": {
            "use_polarity_steering": True,
            "use_holonomy_reg": True,
            "use_stabilizer": True,
            "use_mirror_closure": True,
            "use_internal_mirror_closure": True,
            "use_epistemic_self_improvement": True,
            "num_internal_agents": 4,
            "internal_mirror_lambda": 0.05,
            "epistemic_lambda": 0.03,
            "epistemic_ema_beta": 0.995,
            "epistemic_start_step": 2000,
        },
        "full_steerage_v3": {
            "use_polarity_steering": True,
            "use_holonomy_reg": True,
            "use_stabilizer": True,
            "use_mirror_closure": True,
            "use_internal_mirror_closure": True,
            "use_epistemic_self_improvement": True,
            "use_polarity_navigation": True,
            "num_internal_agents": 4,
            "internal_mirror_lambda": 0.05,
            "epistemic_lambda": 0.03,
            "epistemic_ema_beta": 0.995,
            "epistemic_start_step": 2000,
            "polarity_navigation_lambda": 0.02,
            "polarity_noise_strong": 0.15,
            "polarity_noise_weak": 0.02,
        },
        "full_steerage_v4": {
            "use_polarity_steering": True,
            "use_holonomy_reg": True,
            "use_stabilizer": True,
            "use_mirror_closure": True,
            "use_internal_mirror_closure": True,
            "use_epistemic_self_improvement": True,
            "use_polarity_navigation": True,
            "use_resilience_reg": True,
            "num_internal_agents": 4,
            "internal_mirror_lambda": 0.05,
            "epistemic_lambda": 0.03,
            "epistemic_ema_beta": 0.995,
            "epistemic_start_step": 2000,
            "polarity_navigation_lambda": 0.02,
            "polarity_noise_strong": 0.15,
            "polarity_noise_weak": 0.02,
            "resilience_lambda": 0.01,
            "resilience_noise_level": 0.05,
            "resilience_start_step": 3000,
        },
    }

    # Validate conditions
    conditions_to_run = []
    for cond_name in args.conditions:
        if cond_name not in conditions:
            print(f"Warning: Unknown condition '{cond_name}', skipping.")
            continue
        conditions_to_run.append(cond_name)

    if not conditions_to_run:
        print("No valid conditions specified. Exiting.")
        return

    print(f"\n{'='*70}")
    print(f"PARALLEL SEED RUNNER")
    print(f"{'='*70}")
    print(f"Conditions: {conditions_to_run}")
    print(f"Seeds per condition: {args.num_seeds}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*70}\n")

    # Create list of all (condition, seed) tasks
    tasks = []
    for cond_name in conditions_to_run:
        cond_args = conditions[cond_name]
        for seed in range(args.num_seeds):
            tasks.append((cond_name, cond_args, seed))

    print(f"Total tasks: {len(tasks)}")

    # Build task list as tuples for the top-level function
    task_list = [(seed, cond_name, cond_args, args, results_dir)
                 for cond_name, cond_args, seed in tasks]

    with mp.Pool(processes=args.parallel) as pool:
        results = pool.map(_run_single_seed_task, task_list)

    print(f"\n{'='*70}")
    print(f"ALL TASKS COMPLETED")
    print(f"{'='*70}")
    print(f"Results saved to: {results_dir}")

    # Optional: Aggregate results
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"\nSuccessful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed seeds:")
        for f in failed:
            print(f"  {f['condition']} seed {f['seed']}: {f['error'][:80]}...")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method("spawn", force=True)
    main()