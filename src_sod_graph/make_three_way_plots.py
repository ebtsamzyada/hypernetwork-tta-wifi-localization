"""
Three-way comparison plots across all pipeline variants:
  - UJIIndoorLoc, plain MLP (src/)
  - SOD-HCXY, plain MLP (src_sod/)
  - SOD-HCXY, graph/set-attention encoder (src_sod_graph/)
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "../outputs_sod_graph"

with open("../seed_data.json") as f:
    mlp_seeds = json.load(f)
with open("../seed_data_graph.json") as f:
    graph_seeds = json.load(f)

mlp_frozen, mlp_adapted = np.array(mlp_seeds["frozen"]), np.array(mlp_seeds["adapted"])
graph_frozen, graph_adapted = np.array(graph_seeds["frozen"]), np.array(graph_seeds["adapted"])

# ---------------- Plot 1: three-way baseline/frozen/adapted bars ----------------
fig, ax = plt.subplots(figsize=(9, 5.5))
labels = ["Clean\nbaseline", "Frozen\nunder drift\n(15-seed avg)", "Adapted\nunder drift\n(15-seed avg)"]
uji_vals = [8.05, 13.60, 13.37]
sod_mlp_vals = [8.47, float(mlp_frozen.mean()), float(mlp_adapted.mean())]
sod_graph_vals = [6.53, float(graph_frozen.mean()), float(graph_adapted.mean())]

x = np.arange(len(labels))
w = 0.25
ax.bar(x - w, uji_vals, w, label="UJIIndoorLoc, plain MLP", color="tab:gray")
ax.bar(x, sod_mlp_vals, w, label="SOD-HCXY, plain MLP", color="tab:orange")
ax.bar(x + w, sod_graph_vals, w, label="SOD-HCXY, graph encoder", color="tab:blue")
for offset, vals in [(-w, uji_vals), (0, sod_mlp_vals), (w, sod_graph_vals)]:
    for i, v in enumerate(vals):
        ax.text(i + offset, v + 0.15, f"{v:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Localization error (m)")
ax.set_title("Three pipeline variants: baseline and drift-adaptation comparison")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/three_way_comparison.png", dpi=150)
plt.close(fig)

# ---------------- Plot 2: per-seed robustness, MLP vs graph encoder ----------------
fig, ax = plt.subplots(figsize=(9, 5.5))
seeds = np.arange(1, 16)
w = 0.38
mlp_imp = mlp_frozen - mlp_adapted
graph_imp = graph_frozen - graph_adapted
ax.bar(seeds - w / 2, mlp_imp, w, label=f"Plain MLP (mean +{mlp_imp.mean():.2f}m)", color="tab:orange")
ax.bar(seeds + w / 2, graph_imp, w, label=f"Graph encoder (mean +{graph_imp.mean():.2f}m)", color="tab:blue")
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Random drift seed")
ax.set_ylabel("Improvement (frozen - adapted), m")
ax.set_title("Per-seed adaptation benefit: plain MLP vs. graph encoder\n(both 15/15 seeds positive)")
ax.set_xticks(seeds)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/per_seed_mlp_vs_graph.png", dpi=150)
plt.close(fig)

# ---------------- Plot 3: relative improvement % across all variants ----------------
fig, ax = plt.subplots(figsize=(7.5, 5))
variants = ["UJIIndoorLoc\nplain MLP", "SOD-HCXY\nplain MLP", "SOD-HCXY\ngraph encoder"]
rel_improvement = [
    (13.60 - 13.37) / 13.60 * 100,
    (mlp_frozen.mean() - mlp_adapted.mean()) / mlp_frozen.mean() * 100,
    (graph_frozen.mean() - graph_adapted.mean()) / graph_frozen.mean() * 100,
]
colors = ["tab:gray", "tab:orange", "tab:blue"]
bars = ax.bar(variants, rel_improvement, color=colors, edgecolor="black", linewidth=0.5)
for i, v in enumerate(rel_improvement):
    ax.text(i, v + 0.15, f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("Relative TTA improvement (%)")
ax.set_title("Relative hypernetwork-adaptation benefit across pipeline variants")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/relative_improvement_comparison.png", dpi=150)
plt.close(fig)

print("Saved: three_way_comparison.png, per_seed_mlp_vs_graph.png, relative_improvement_comparison.png")
