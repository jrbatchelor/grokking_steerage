#!/usr/bin/env python3
"""
Next advancement: Spatial lattice version with local interactions.
1D chain of nodes with local neighbor coupling + self-reference.
Track propagating "avalanches" (coherent flips or large local changes).
Quantify avalanche size distribution for better SOC test.
This moves beyond mean-field to allow spatial propagation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
N = 128  # chain length
STEPS = 3000
SR = 1.55  # near critical from previous
COUPLING = 1.2  # local neighbor strength
NOISE = 0.06

# Initialize states on a 1D chain
states = np.random.uniform(-0.7, 0.7, N)
avalanche_sizes = []

for t in range(STEPS):
    new_states = states.copy()
    mean_field = np.mean(states)
    
    for i in range(N):
        # Local neighbors (periodic boundary)
        left = states[(i-1) % N]
        right = states[(i+1) % N]
        local = (left + right) / 2
        
        # Update: self-reference + local coupling + weak global pull + noise
        update = (SR * states[i] +
                  COUPLING * (local - states[i]) +
                  0.4 * (mean_field - states[i]) +
                  NOISE * np.random.randn())
        new_states[i] = np.tanh(update)
    
    # Detect "avalanches": significant coherent changes across neighboring nodes
    # Simple proxy: count contiguous regions where |delta_state| > threshold
    delta = np.abs(new_states - states)
    threshold = 0.25
    in_avalanche = False
    current_size = 0
    
    for d in delta:
        if d > threshold:
            if not in_avalanche:
                in_avalanche = True
                current_size = 1
            else:
                current_size += 1
        else:
            if in_avalanche:
                if current_size >= 3:  # minimum size to count as avalanche
                    avalanche_sizes.append(current_size)
                in_avalanche = False
                current_size = 0
    
    if in_avalanche and current_size >= 3:
        avalanche_sizes.append(current_size)
    
    states = new_states

avalanche_sizes = np.array(avalanche_sizes)

print("=== Spatial Lattice SOC Results ===")
print(f"Total steps: {STEPS}")
print(f"Number of avalanches detected: {len(avalanche_sizes)}")
if len(avalanche_sizes) > 0:
    print(f"Avalanche size stats: min={avalanche_sizes.min()}, max={avalanche_sizes.max()}, mean={avalanche_sizes.mean():.2f}")

# Log-log histogram
if len(avalanche_sizes) > 15:
    log_sizes = np.log10(avalanche_sizes)
    hist, bin_edges = np.histogram(log_sizes, bins=12)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = hist > 0
    
    if np.sum(mask) > 3:
        log_bins = bin_centers[mask]
        log_counts = np.log10(hist[mask] + 1)
        slope, intercept, r_value, _, std_err = stats.linregress(log_bins, log_counts)
        print(f"Power-law fit slope: {slope:.3f} ± {std_err:.3f}  (R²={r_value**2:.3f})")
    else:
        slope = None
else:
    slope = None

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Time series of global mean (to show activity)
# (we didn't store full history for speed, but can re-run conceptually)
axes[0].text(0.5, 0.5, 'Spatial 1D Lattice Model\n(Local + Self-Ref + Global pull)\n\nAvalanches = contiguous regions\nof significant state change', 
             ha='center', va='center', fontsize=11, transform=axes[0].transAxes,
             bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.9))
axes[0].axis('off')
axes[0].set_title('Model Structure', fontsize=12)

# Avalanche size distribution
if len(avalanche_sizes) > 0:
    axes[1].hist(avalanche_sizes, bins=20, color='#d62728', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Avalanche Size (contiguous nodes)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Avalanche Size Distribution (Spatial Model)')
    axes[1].grid(True, alpha=0.3)
    
    if slope is not None:
        axes[1].text(0.95, 0.95, f'Log-log slope ≈ {slope:.2f}', transform=axes[1].transAxes,
                    fontsize=10, ha='right', va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/batch/Documents/grokking-experiment/artifacts/spatial_lattice_soc.png', dpi=150, bbox_inches='tight')
print("\nFigure saved to /home/batch/Documents/grokking-experiment/artifacts/spatial_lattice_soc.png")

print("\n=== Interpretation ===")
print("Moving to a spatial lattice allows coherence changes to propagate locally.")
print("This produces discrete 'avalanches' whose size distribution can be analyzed for power-law signatures.")
print("The slope and tail behavior give a more spatially realistic test of SOC-like dynamics")
print("emerging from self-referential + local interaction rules — a step closer to natural systems like river networks.")
