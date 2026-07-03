#!/usr/bin/env python3
"""
Experimental Runner for Grokking Ablation Studies
Version 0.9 - Fixed + Complete Pipeline
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Dict, Any

def _maybe_reexec_with_project_venv() -> None:
    project_root = Path(__file__).resolve().parent
    venv_root = project_root / ".venv"
    venv_python = project_root / ".venv" / "bin" / "python3"
    if not venv_python.exists():
        return

    if Path(sys.prefix).resolve() == venv_root.resolve():
        return

    required_modules = ("numpy", "pandas", "scipy", "torch")
    if all(importlib.util.find_spec(module) is not None for module in required_modules):
        return

    os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])


_maybe_reexec_with_project_venv()

import numpy as np
import pandas as pd
from scipy import stats

from grokking_experiment import run_experiment


def get_experimental_conditions() -> Dict[str, Dict[str, Any]]:
    """Define your experimental conditions here."""
    return {
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
    }


def compute_grokking_metrics(history: dict) -> dict:
    """Compute key grokking and Benford-related metrics."""
    steps = np.array(history["steps"])
    test_accs = np.array(history["test_accs"])
    weight_norms = np.array(history.get("weight_norms", [np.nan] * len(steps)))
    benford_chi2 = np.array(history.get("benford_chi2", [np.nan] * len(steps)))
    benford_p1 = np.array(history.get("benford_p1", [np.nan] * len(steps)))

    metrics = {
        "final_test_acc": float(test_accs[-1]) if len(test_accs) > 0 else np.nan,
        "final_test_loss": float(history.get("test_losses", [np.nan])[-1]),
    }

    # Steps to reach accuracy thresholds
    above_50 = np.where(test_accs >= 0.5)[0]
    above_90 = np.where(test_accs >= 0.9)[0]
    metrics["steps_to_0.5"] = int(steps[above_50[0]]) if len(above_50) > 0 else None
    metrics["steps_to_0.9"] = int(steps[above_90[0]]) if len(above_90) > 0 else None

    # Weight norm reduction
    if len(weight_norms) > 1 and not np.isnan(weight_norms[0]):
        metrics["weight_norm_reduction"] = float(
            (weight_norms[0] - weight_norms[-1]) / weight_norms[0]
        )

    # Benford improvement metrics
    if len(benford_chi2) > 1 and not np.isnan(benford_chi2[0]):
        metrics["benford_chi2_reduction"] = float(
            (benford_chi2[0] - benford_chi2[-1]) / benford_chi2[0]
        )

    if len(benford_p1) > 1 and not np.isnan(benford_p1[0]):
        metrics["p1_change"] = float(benford_p1[-1] - benford_p1[0])

    return metrics


def run_single_experiment(
    condition_name: str,
    condition_args: Dict[str, Any],
    seed: int,
    base_args: argparse.Namespace,
    results_dir: Path
) -> Dict[str, Any]:

    print(f"\n{'='*60}")
    print(f"▶ Running: {condition_name} | Seed: {seed}")
    print(f"{'='*60}")

    args = argparse.Namespace(**vars(base_args))
    for k, v in condition_args.items():
        setattr(args, k, v)
    args.seed = seed

    run_dir = results_dir / condition_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(run_dir)

    try:
        history = run_experiment(args)
        metrics = compute_grokking_metrics(history)
        metrics.update({"condition": condition_name, "seed": seed})

        # Save full history
        np.savez(run_dir / "history.npz", **{
            k: v for k, v in history.items() if isinstance(v, (list, np.ndarray))
        })

        # Save computed metrics
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Completed | Final Test Acc: {metrics.get('final_test_acc', 'N/A'):.4f}")
        return metrics

    except Exception as e:
        print(f"✗ ERROR in {condition_name} seed {seed}: {e}")
        error_info = {"condition": condition_name, "seed": seed, "error": str(e)}
        with open(run_dir / "error.txt", "w") as f:
            f.write(str(e))
        return error_info


def perform_statistical_comparison(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Pairwise statistical tests using Mann-Whitney U test."""
    conditions = df["condition"].unique()
    results = []

    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i + 1:]:
            data1 = df[df["condition"] == cond1][metric].dropna()
            data2 = df[df["condition"] == cond2][metric].dropna()

            if len(data1) < 2 or len(data2) < 2:
                continue

            stat, pval = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            n1, n2 = len(data1), len(data2)
            effect_size = 1 - (2 * stat) / (n1 * n2)

            results.append({
                "comparison": f"{cond1} vs {cond2}",
                "metric": metric,
                "p_value": round(pval, 4),
                "effect_size": round(effect_size, 4),
                "significant": pval < 0.05
            })

    return pd.DataFrame(results)


def generate_comparison_plots(df: pd.DataFrame, results_dir: Path):
    """Generate bar plots comparing conditions."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="viridis")

    for metric in ["steps_to_0.9", "final_test_acc", "weight_norm_reduction", "benford_chi2_reduction"]:
        if metric not in df.columns or df[metric].dropna().empty:
            continue

        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x="condition", y=metric, errorbar="sd", capsize=0.15)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=5)

        plt.title(f"{metric.replace('_', ' ').title()} by Condition", fontsize=14)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        plot_path = results_dir / f"plot_{metric}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {plot_path}")


def aggregate_and_analyze(results_dir: Path):
    """Aggregate results, run statistics, and generate plots."""
    all_metrics = []

    for condition_dir in sorted(results_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        for seed_dir in condition_dir.glob("seed_*"):
            metrics_file = seed_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    all_metrics.append(json.load(f))

    if not all_metrics:
        print("No results found to analyze.")
        return

    df = pd.DataFrame(all_metrics)
    df.to_csv(results_dir / "all_results.csv", index=False)

    print("\n" + "="*75)
    print("EXPERIMENT SUMMARY")
    print("="*75)

    key_metrics = ["final_test_acc", "steps_to_0.9", "weight_norm_reduction", "benford_chi2_reduction"]
    available = [m for m in key_metrics if m in df.columns]

    if available:
        summary = df.groupby("condition")[available].agg(["mean", "std"]).round(4)
        print(summary)
        summary.to_csv(results_dir / "summary.csv")

    # Statistical comparisons
    print("\n" + "="*75)
    print("STATISTICAL COMPARISONS")
    print("="*75)

    for metric in ["steps_to_0.9", "final_test_acc", "weight_norm_reduction", "benford_chi2_reduction"]:
        if metric in df.columns and df[metric].notna().any():
            comp_df = perform_statistical_comparison(df, metric)
            if not comp_df.empty:
                print(f"\n{metric}:")
                print(comp_df.to_string(index=False))
                comp_df.to_csv(results_dir / f"stats_{metric}.csv", index=False)

    # Generate plots
    print("\nGenerating comparison plots...")
    generate_comparison_plots(df, results_dir)

    print(f"\nAll results saved to: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Grokking Ablation Experiments")

    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--all_conditions", action="store_true")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--results_dir", type=str, default="./ablation_results")

    # Base arguments required by grokking_experiment.py
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--train_frac", type=float, default=0.45)
    parser.add_argument("--max_steps", type=int, default=15000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--wd", type=float, default=0.40)
    parser.add_argument("--lr", type=float, default=0.0006)
    parser.add_argument("--polarity_lambda", type=float, default=0.05)
    parser.add_argument("--holonomy_lambda", type=float, default=0.03)
    parser.add_argument("--holonomy_target", type=float, default=0.95)
    parser.add_argument("--holonomy_check_interval", type=int, default=100)
    parser.add_argument("--epistemic_lambda", type=float, default=0.03)
    parser.add_argument("--epistemic_ema_beta", type=float, default=0.995)
    parser.add_argument("--epistemic_start_step", type=int, default=2000)

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    conditions = get_experimental_conditions()

    if args.all_conditions:
        conditions_to_run = list(conditions.keys())
    else:
        conditions_to_run = args.conditions or list(conditions.keys())[:2]

    for condition_name in conditions_to_run:
        if condition_name not in conditions:
            continue
        condition_args = conditions[condition_name]

        for seed in range(args.num_seeds):
            run_single_experiment(condition_name, condition_args, seed, args, results_dir)

    aggregate_and_analyze(results_dir)


if __name__ == "__main__":
    main()