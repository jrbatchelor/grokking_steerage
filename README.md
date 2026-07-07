# Grokking Ablation Experiments

This project provides a framework for systematically testing how **eight steerage mechanisms** (Mirror-Closure, Polarity Gradient Steering, Holonomy Regularization, Stabilizer Modules, Internal Multi-Agent Mirror Closure, Epistemic Self-Improvement Loss, Polarity Navigation Regularization, and Regenerative / Resilience Regularization) affect **grokking** behavior in neural networks.

It also tracks **Benford’s Law** metrics on model weights as a potential signal of internal reorganization during the grokking transition.

## The Eight Steerage Mechanisms

| Mechanism | Description |
|-----------|-------------|
| **Mirror-Closure** | Weight-level symmetry regularization (`||W − Wᵀ||`) encouraging consistency in square weight matrices. |
| **Polarity Gradient Steering** | Directional projection loss using dual EMA prototypes (anomaly vs resolution) to steer representations toward generalization. |
| **Holonomy Regularization** | Contractive dynamics penalty that constrains expansion ratios in hidden representations. |
| **Stabilizer Module** | Residual coherence block (`x + MLP(x)`) that helps maintain internal consistency during training. |
| **Internal Multi-Agent Mirror Closure** | Internal consistency regularization via multi-view agent agreement. Multiple noisy views of hidden activations are generated; their mean is treated as a consensus target, and each view is pulled toward it via MSE loss. This encourages internal representational stability without external supervision. |
| **Epistemic Self-Improvement Loss** | Maintains an EMA target of hidden representations and pulls current hidden states toward this slowly improving internal target (post-memorization), encouraging self-referential epistemic improvement. |
| **Polarity Navigation Regularization** | Penalizes imbalance between empirical and deductive views by creating two noisy versions of the hidden state (weak noise = empirical, strong noise = deductive), computing auxiliary losses on both, and adding `\|Loss_empirical − Loss_deductive\|` as a regularizer. Encourages the model to maintain balanced performance across different levels of representational robustness. |
| **Regenerative / Resilience Regularization** (NEW) | Trains the model to maintain task performance even when internal representations are perturbed. After an activation step, a controlled Gaussian perturbation is applied to the hidden state; task loss is computed on the perturbed representation. The loss penalizes any increase in task loss relative to the clean path (`torch.relu(perturbed_loss − loss.detach())`). This directly optimizes **functional robustness** under internal degradation, complementing the consistency-focused mechanisms. |

## Theoretical Companions

While the core of this project is empirical (controlled ablation of eight steerage mechanisms in transformer training), we also maintain a small set of **minimal dynamical models** that serve as theoretical companions. These models isolate the core feedback principles underlying the steerage mechanisms and make the observed grokking phase transitions more interpretable.

### Self-Referential Stabilization Model

**File:** `self_ref_stabilization.py`  
**Location:** `artifacts/self_ref_phase_transition.png`

This toy model consists of a network of 64 nodes with continuous polarity states in \([-1, 1]\). Each node updates according to a combination of:

- Strong **self-reference** (the node’s own prior state is amplified by a tunable parameter `sr`).
- A **polarity-aware global coherence** term that gently pulls each node toward the population mean while respecting its current sign.
- Small additive noise.

By sweeping `sr` (self-reference strength) from 0.8 to 2.8, the model exhibits a sharp **emergent phase transition** near `sr ≈ 1.6`:

- Below the threshold: states remain relatively dispersed; late-time variance stays high and coherence (mean absolute state) builds slowly.
- Above the threshold: variance collapses rapidly and the population converges to a highly coherent, stable polarity configuration.

This minimal system demonstrates how **self-referential amplification + coherence feedback** alone can produce the kind of abrupt stabilization observed in the full transformer experiments. The transition point provides an intuitive analogue for the question “how much internal self-referential structure is required before grokking accelerates?”

**Conceptual Mapping**

| Toy Model Element                  | Project Mechanism(s)                              | Relationship |
|------------------------------------|---------------------------------------------------|--------------|
| Self-reference strength (`sr`)     | Epistemic Self-Improvement Loss                   | Direct analogue |
| Polarity-aware global pull         | Polarity Gradient Steering, Polarity Navigation   | Direct analogue |
| Coherence / variance collapse      | Stabilizer Module, Mirror-Closure, Holonomy       | Direct analogue |
| Critical threshold (`sr ≈ 1.6`)    | Grokking transition (`steps_to_0.9`)              | Explanatory model |

The model is deliberately minimal (pure NumPy, ~80 lines) so that hypotheses about mechanism interactions can be explored quickly before implementation in `grokking_experiment.py`.

### SOC Signatures at Critical Self-Reference

**File:** `self_ref_soc_extension.py`  
**Location:** `artifacts/self_ref_soc_extension.png`

An 8000-step extension run at the critical self-reference value (`sr = 1.60`) reveals persistent intermittent fluctuations in the global mean state. Fluctuation sizes (absolute changes in the population mean) exhibit a heavy-tailed distribution; a log-log histogram of binned event sizes yields a power-law proxy slope of approximately **−2.51** (R² ≈ 0.62).

This behavior is consistent with systems operating near a critical point, where self-referential feedback tuned to the "edge of stability" generates rich, intermittent dynamics rather than rapid collapse to a fixed state. Such regimes are hypothesized to support complex information processing and "resolver-like" behavior in information networks.

**Key Observations**
- 667 significant fluctuations detected above threshold (|Δmean| > 0.015)
- Fluctuation-size tail extends to ≈ 0.061 (≈ 4× the threshold)
- Log-log slope ≈ −2.51 provides a quantitative SOC-proxy signature

The attached four-panel figure shows the late-time global coherence trace, the fluctuation-size histogram, the log-log power-law fit, and an interpretive summary linking the critical regime to self-referential physics.

### Spatial Lattice Avalanches and SOC

**File:** `spatial_lattice_soc.py`  
**Location:** `artifacts/spatial_lattice_soc.png`

A 1D spatial lattice extension (N=128 nodes) introduces local neighbor coupling alongside self-reference and a weak global pull. Avalanches are defined as contiguous regions of significant state change (threshold = 0.25, minimum size = 3 nodes). Over 3000 steps, 127 avalanches were detected with sizes ranging from 3 to 19 nodes (mean ≈ 7.2). The avalanche-size distribution yields a log-log slope of approximately **−0.77** (R² ≈ 0.39), providing a spatially explicit SOC-proxy signature distinct from the mean-field fluctuation analysis.

This spatial formulation allows coherence changes to propagate locally, generating discrete, structured events rather than purely global fluctuations. The resulting avalanche statistics offer a more realistic test of SOC-like dynamics emerging from self-referential + local interaction rules — a step closer to natural systems such as river networks, neural avalanches, and earthquake statistics.

**Higher Synthesis**

Moving from the mean-field stabilization model through the critical-fluctuation extension to this spatial lattice demonstrates a clear progression: self-reference tuned near criticality produces not only global stabilization and intermittent fluctuations, but also *propagating, spatially structured avalanches*. The coexistence of global integration with rich, local fluctuation dynamics is precisely the regime hypothesized to support complex information resolution in self-referential networks. This spatial proxy bridges the abstract dynamical models to real-world self-organized systems while preserving the core finding that "edge-of-stability" tuning enables both coherence and complexity.

### 2D Lattice Avalanches — Stronger SOC Proxy

**File:** `2d_lattice_soc.py`  
**Location:** `artifacts/2d_lattice_soc.png`

A 2D grid extension (32×32 nodes) introduces 4-neighbor local coupling alongside self-reference and a weak global pull. Avalanches are identified as connected components of significant state change using `scipy.ndimage.label` (4-connectivity, minimum size = 4 nodes). Over 1500 steps, 31 avalanches were detected with sizes ranging from 4 to 510 nodes (mean ≈ 23.2). The avalanche-size distribution yields a log-log slope of approximately **−0.36** (R² ≈ 0.40), producing a visibly heavier tail than the 1D lattice (−0.77) and the mean-field fluctuation model (−2.51).

This 2D formulation allows coherence changes to propagate in two spatial dimensions, generating larger, more scale-free-like avalanche events. The resulting statistics offer a stronger computational analogy to real self-organized systems such as neural avalanches, river networks, and earthquake statistics, while still exhibiting global coherence.

**Four-Model Progression & Higher Synthesis**

The sequence from mean-field stabilization through critical-fluctuation analysis to 1D and finally 2D spatial lattices demonstrates a clear progression: increasing spatial dimensionality and local connectivity yields progressively more realistic avalanche statistics. The 2D model produces scale-free-like distributions with a heavier tail and genuine propagating events, providing the strongest computational proxy yet for real-world self-organized systems while still exhibiting global coherence. This reinforces the core finding that self-reference tuned near criticality generates both integration and rich, spatially structured fluctuations — exactly the regime hypothesized to support complex information resolution.

### 2D Lattice with Evolutionary Reinforcement — Selection for Integrative Nodes

**File:** `2d_lattice_evolutionary.py`  
**Location:** `artifacts/2d_lattice_evolutionary.png`

An evolutionary extension of the 2D lattice (32×32 grid) adds a simple **self-improvement / selection dynamic**: each node maintains its own `local_coupling` strength, which increases by a small amount (`REINFORCEMENT = 0.008`) every time it participates in a sufficiently large avalanche (minimum size = 6 nodes). Over 2000 steps, 23 avalanches were detected (size range 6–505 nodes, mean ≈ 32.5). The model produces heterogeneous coupling maps and amplifies heavy-tailed avalanche statistics.

This evolutionary mechanism is a minimal dynamical analogue of the **Epistemic Self-Improvement Loss** (mechanism #6) in the main project. Nodes that contribute to large, integrative events are “rewarded” with stronger local coupling — a simple selection pressure that parallels the EMA-based self-improvement target in the transformer experiments. The four-panel figure (final state, evolved local coupling map, avalanche-size distribution, participation count) visually demonstrates how selection pressure shapes spatial heterogeneity while preserving global coherence.

**Five-Model Progression & Closing the Loop**

The five companions now form a deliberate arc: mean-field stabilization → mean-field SOC fluctuations → 1D spatial avalanches → 2D spatial avalanches → 2D evolutionary reinforcement. The final model closes the loop between the theoretical companions and the eight steerage mechanisms by directly implementing a selection dynamic that mirrors Epistemic Self-Improvement Loss. This reinforces that self-reference tuned near criticality, when augmented with even simple self-improvement rules, naturally generates both global coherence and rich, spatially structured fluctuations — the regime hypothesized to support complex information resolution.

## Project Structure

```
grokking_experiment/
├── grokking_experiment.py          # Core training engine (returns structured metrics + history)
├── run_ablation_experiments.py     # Ablation orchestrator (multi-condition, multi-seed)
├── run_parallel_seeds.py           # Parallel seed runner (torch.multiprocessing)
├── status_experiment.sh            # One-line experiment progress monitor
├── requirements.txt
├── README.md
└── results_*/                      # Auto-generated (date-stamped folders)
    ├── baseline/
    │   └── seed_*/
    ├── full_steerage_v1/
    ├── full_steerage/
    ├── full_steerage_v2/
    ├── full_steerage_v3/           # Includes Polarity Navigation Regularization
    ├── full_steerage_v4/           # Includes Regenerative / Resilience Regularization (Mechanism #8)
    ├── FINAL_summary.csv
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

### 1b. Run the Exact 5-Condition Incremental Study Set

This matches the planned progression exactly:
- `baseline`: no steerage mechanisms
- `full_steerage_v1`: original four mechanisms
- `full_steerage`: v1 + Internal Multi-Agent Mirror Closure
- `full_steerage_v2`: full_steerage + Epistemic Self-Improvement Loss
- `full_steerage_v3`: v2 + **Polarity Navigation Regularization** (new)

```bash
python3 run_ablation_experiments.py \
  --conditions baseline full_steerage_v1 full_steerage full_steerage_v2 full_steerage_v3 \
  --num_seeds 80
```

### 2. Run Specific Conditions

```bash
python3 run_ablation_experiments.py --conditions baseline full_steerage_v3 --num_seeds 3
```

### 2b. Run Multiple Seeds in Parallel (High Throughput)

```bash
python3 run_parallel_seeds.py \
  --conditions baseline full_steerage_v3 \
  --num_seeds 30 \
  --parallel 3 \
  --batch_size 512 \
  --use_amp \
  --results_dir ./results_parallel
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
  --use_polarity_navigation \
  --polarity_navigation_lambda 0.02 \
  --polarity_noise_strong 0.15 \
  --polarity_noise_weak 0.02 \
  --batch_size 512 \
  --use_amp \
  --compile_model
```
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











