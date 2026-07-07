#!/usr/bin/env python3
"""
Toy model: Self-referential information network with emergent stabilization.
Proxy for consciousness/info-resolver concepts and phase-transition onset.
Nodes have polarity states [-1, 1]. Update includes strong self-reference + global coherence feedback.
Sweep self-reference strength; measure late-time stability and coherence.
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
N = 64  # nodes for decent stats
STEPS = 300
SWEEP = np.linspace(0.8, 2.8, 25)  # self_ref strength parameter

results = []
all_histories = {}  # for a couple of key points

for sr in SWEEP:
    states = np.random.uniform(-0.8, 0.8, N)
    late_vars = []
    late_coherences = []
    history_for_plot = []
    
    for t in range(STEPS):
        mean_s = np.mean(states)
        # Self-referential + polarity alignment term:
        # new = tanh( sr * self + coupling * (global_mean alignment incentive) + noise )
        # Strong self-ref encourages persistence of individual polarity; global term encourages coherence
        new_states = np.tanh(
            sr * states +
            1.8 * (mean_s - states) * np.sign(states) +   # polarity-aware pull to coherent mean
            0.08 * np.random.randn(N)
        )
        states = new_states
        
        if t > STEPS - 80:
            late_vars.append(np.var(states))
            late_coherences.append(np.mean(np.abs(states)))
        
        if sr in [1.0, 1.6, 2.2] and t % 20 == 0:  # sample histories for key sr
            if sr not in all_histories:
                all_histories[sr] = []
            all_histories[sr].append(states.copy())
    
    stability = 1.0 / (1.0 + np.mean(late_vars))  # high when variance collapses
    coherence = np.mean(late_coherences)
    results.append((sr, stability, coherence))

# Results
print("=== Toy Self-Referential Stabilization Model Results ===")
print("sr (self-ref strength) | Stability (low var) | Coherence (mean |s|)")
for r in results:
    print(f"{r[0]:5.2f}                  | {r[1]:.4f}             | {r[2]:.4f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
srs, stabs, cohs = zip(*results)

axes[0].plot(srs, stabs, 'o-', color='#1f77b4', linewidth=2, markersize=6)
axes[0].axvline(x=1.6, color='red', linestyle='--', alpha=0.7, label='Approx. transition')
axes[0].set_xlabel('Self-Reference Strength (sr)', fontsize=11)
axes[0].set_ylabel('Stability Metric (inverse late variance)', fontsize=11)
axes[0].set_title('Emergent Stabilization vs Self-Reference', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(srs, cohs, 's-', color='#ff7f0e', linewidth=2, markersize=6)
axes[1].axvline(x=1.6, color='red', linestyle='--', alpha=0.7)
axes[1].set_xlabel('Self-Reference Strength (sr)', fontsize=11)
axes[1].set_ylabel('Coherence (mean |state|)', fontsize=11)
axes[1].set_title('Coherence Build-Up (Polarity Alignment)', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/batch/Documents/grokking-experiment/artifacts/self_ref_phase_transition.png', dpi=160, bbox_inches='tight')
print("\nPlot saved to /home/batch/Documents/grokking-experiment/artifacts/self_ref_phase_transition.png")

# Quick check on sample histories for transition behavior
print("\nSample late-time state std at key sr values (lower = more stable):")
for sr_key in [1.0, 1.6, 2.2]:
    if sr_key in all_histories:
        late = np.array(all_histories[sr_key][-5:])
        print(f"sr={sr_key}: mean std across nodes/time = {np.mean(np.std(late, axis=1)):.4f}")
