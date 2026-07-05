# Grokking Ablation Experiments

This project provides a framework for systematically testing how seven **steerage mechanisms** (Mirror-Closure, Polarity Gradient Steering, Holonomy Regularization, Stabilizer Modules, Internal Multi-Agent Mirror Closure, Epistemic Self-Improvement Loss, and Polarity Navigation Regularization) affect **grokking** behavior in neural networks.

It also tracks **Benford’s Law** metrics on model weights as a potential signal of internal reorganization during the grokking transition.

## The Seven Steerage Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **Mirror-Closure** | Weight-level symmetry regularization (`||W − Wᵀ||`) encouraging consistency in square weight matrices. |
| **Polarity Gradient Steering** | Directional projection loss using dual EMA prototypes (anomaly vs resolution) to steer representations toward generalization. |
| **Holonomy Regularization** | Contractive dynamics penalty that constrains expansion ratios in hidden representations. |
| **Stabilizer Module** | Residual coherence block (`x + MLP(x)`) that helps maintain internal consistency during training. |
| **Internal Multi-Agent Mirror Closure** | Internal consistency regularization via multi-view agent agreement. Multiple noisy views of hidden activations are generated; their mean is treated as a consensus target, and each view is pulled toward it via MSE loss. This encourages internal representational stability without external supervision. |
| **Epistemic Self-Improvement Loss** | Maintains an EMA target of hidden representations and pulls current hidden states toward this slowly improving internal target (post-memorization), encouraging self-referential epistemic improvement. |
| **Polarity Navigation Regularization** | Penalizes imbalance between empirical and deductive views by creating two noisy versions of the hidden state (weak noise = empirical, strong noise = deductive), computing auxiliary losses on both, and adding `\|Loss_empirical − Loss_deductive\|` as a regularizer. Encourages the model to maintain balanced performance across different levels of representational robustness. |

## Project Structure

```
grokking_experiment/
├── grokking_experiment.py          # Core training engine (returns structured metrics + history)
├── run_ablation_experiments.py     # Ablation orchestrator (multi-condition, multi-seed)
├── requirements.txt
├── README.md
└── ablation_results/               # Auto-generated
    ├── baseline/
    │   └── seed_*/
    ├── full_steerage/
    ├── full_steerage_v2/
    │   └── seed_*/
    ├── all_results.csv
    ├── summary.csv
    ├── stats_*.csv
    └── plot_*.png
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the Full Ablation Pipeline

```bash
python run_ablation_experiments.py --all_conditions --num_seeds 5
```

This will:
- Run all configured conditions (including `full_steerage_v2`)
- Use 5 random seeds per condition
- Automatically compute grokking and Benford metrics
- Run statistical comparisons (Mann-Whitney U + effect sizes)
- Generate comparison plots

### 1b. Run the Exact 4-Condition Incremental Study Set

This matches the planned progression exactly:
- `baseline`: no steerage mechanisms
- `full_steerage_v1`: original four mechanisms
- `full_steerage`: v1 + Internal Multi-Agent Mirror Closure
- `full_steerage_v2`: full_steerage + Epistemic Self-Improvement Loss

```bash
python3 run_ablation_experiments.py \
  --conditions baseline full_steerage_v1 full_steerage full_steerage_v2 \
  --num_seeds 80
```

### 2. Run Specific Conditions

```bash
python3 run_ablation_experiments.py --conditions baseline full_steerage_v1 full_steerage full_steerage_v2 --num_seeds 3
```

### 3. Run a Single Experiment Directly

```bash
python grokking_experiment.py \
  --p 97 \
  --hidden_dim 1024 \
  --max_steps 15000 \
  --use_polarity_steering \
  --use_holonomy_reg \
  --use_stabilizer \
  --use_mirror_closure \
  --use_internal_mirror_closure \
  --num_internal_agents 4 \
  --internal_mirror_lambda 0.05 \
  --use_epistemic_self_improvement \
  --epistemic_lambda 0.03 \
  --epistemic_ema_beta 0.995 \
  --epistemic_start_step 2000
```

## Output

After running experiments, the `ablation_results/` folder will contain:

| File | Description |
|------|-------------|
| `all_results.csv` | Raw results from all runs |
| `summary.csv` | Mean ± std across seeds |
| `stats_*.csv` | Statistical comparisons (p-values + effect sizes) |
| `plot_*.png` | Bar plots comparing conditions |

## Key Metrics Tracked

| Metric | Meaning |
|--------|---------|
| `steps_to_0.5` / `steps_to_0.9` | How fast grokking occurs |
| `final_test_acc` | Final generalization performance |
| `weight_norm_reduction` | Measure of simplicity bias |
| `benford_chi2_reduction` | Improvement in parameter distribution quality |

## Performance Tuning (RTX 3060 / Modern GPUs)

For maximum GPU utilization during long ablation studies, use these optimizations:

```bash
python grokking_experiment.py \
  --batch_size 512 \
  --num_workers 8 \
  --use_amp \
  --compile_model
```

**Key Flags:**

| Flag | Default | Effect |
|------|---------|--------|
| `--batch_size 512` | 512 | Larger batches improve GPU occupancy |
| `--num_workers 8` | 8 | Parallel data loading (match to CPU cores) |
| `--use_amp` | True | Automatic Mixed Precision (~2× faster on RTX 30-series) |
| `--compile_model` | False | PyTorch 2.0+ graph compilation (+10–20% speedup) |

**Expected speedup:** ~2.5–3× faster training vs default settings on RTX 3060.

### Parallel Seed Execution

For even higher throughput, use the parallel seed runner:

```bash
python run_parallel_seeds.py \
  --conditions baseline full_steerage_v2 \
  --num_seeds 30 \
  --parallel 3 \
  --batch_size 512 \
  --use_amp \
  --results_dir ./results_parallel
```

This runs 3 seeds simultaneously on the same GPU using `torch.multiprocessing`.

## Author

**Jonathan "Batch" Batchelor**

**Goal:** Understand and improve grokking through principled steerage mechanisms and distributional diagnostics.











