"""
Extra results plots beyond the main Step 5 deployment curve:
  1. Per-seed robustness: does adaptation help in every independent
     drift realization, or just on average?
  2. AP density comparison across candidate datasets (the sparsity
     hypothesis that motivated the dataset switch).
  3. Side-by-side headline comparison: UJIIndoorLoc vs SOD-HCXY.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "../outputs_sod"

with open("../seed_data.json") as f:
    seed_data = json.load(f)
frozen = np.array(seed_data["frozen"])
adapted = np.array(seed_data["adapted"])
improvement = frozen - adapted

# ---------------- Plot 1: per-seed robustness ----------------
fig, ax = plt.subplots(figsize=(8, 5))
seeds = np.arange(1, len(frozen) + 1)
colors = ["tab:green" if v > 0 else "tab:red" for v in improvement]
ax.bar(seeds, improvement, color=colors, edgecolor="black", linewidth=0.5)
ax.axhline(0, color="black", linewidth=1)
ax.axhline(improvement.mean(), color="tab:blue", linestyle="--", linewidth=1.5,
           label=f"mean = +{improvement.mean():.2f}m")
ax.set_xlabel("Random drift seed")
ax.set_ylabel("Improvement (frozen error - adapted error), m")
ax.set_title(f"Adaptation benefit across {len(frozen)} independent drift realizations\n"
             f"({(improvement > 0).sum()}/{len(frozen)} seeds positive)")
ax.set_xticks(seeds)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_seed_robustness.png", dpi=150)
plt.close(fig)

# ---------------- Plot 2: AP density comparison ----------------
datasets = ["UJIIndoorLoc\n(520 APs)", "SOD-HCXY\n(56 APs)", "SOD-SYL\n(46 APs)", "SOD-CETC331\n(52 APs)"]
visible_pct = [3.8, 22.3, 38.8, 49.4]
groups_ok = [True, True, False, False]  # enough session groups for leakage-free CV

fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = ["tab:blue" if ok else "lightgray" for ok in groups_ok]
bars = ax.bar(datasets, visible_pct, color=bar_colors, edgecolor="black", linewidth=0.5)
bars[1].set_color("tab:orange")  # highlight the chosen dataset
for i, v in enumerate(visible_pct):
    ax.text(i, v + 1, f"{v}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Mean AP visibility per fingerprint (%)")
ax.set_title("AP density across candidate datasets\n(gray = insufficient session groups for leakage-free split)")
ax.set_ylim(0, 58)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/dataset_density_comparison.png", dpi=150)
plt.close(fig)

# ---------------- Plot 3: UJI vs SOD-HCXY headline comparison ----------------
fig, ax = plt.subplots(figsize=(8, 5))
labels = ["Clean\nbaseline", "Frozen\nunder drift", "Adapted\nunder drift"]
uji_vals = [8.05, 13.60, 13.37]
sod_vals = [8.47, float(frozen.mean()), float(adapted.mean())]

x = np.arange(len(labels))
w = 0.35
ax.bar(x - w / 2, uji_vals, w, label="UJIIndoorLoc (sparse, 3.8% visible)", color="tab:gray")
ax.bar(x + w / 2, sod_vals, w, label="SOD-HCXY (dense, 22.3% visible)", color="tab:orange")
for i, v in enumerate(uji_vals):
    ax.text(i - w / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
for i, v in enumerate(sod_vals):
    ax.text(i + w / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Localization error (m)")
ax.set_title("Sparse vs. dense RSSI: baseline and drift-adaptation comparison")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/uji_vs_sod_comparison.png", dpi=150)
plt.close(fig)

print("Saved: per_seed_robustness.png, dataset_density_comparison.png, uji_vs_sod_comparison.png")
