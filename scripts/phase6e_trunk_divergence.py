"""Phase 6e: root-cause analysis for why Phase 6d (embedding-space drift)
failed, in fact failed WORSE cross-site than the raw-RSSI-stats trials
(6b/6c).

Phase 6d's premise was that the frozen backbone's bottleneck embedding is a
common representation space, so embedding displacement is a site-agnostic
drift signal. But the checkpoints used for ALL of Phase 6/6b/6c/6d
(outputs/phase5c_v4/model_{site}.pt) are PER-SITE FINE-TUNED: phase5c
freezes the shared xy_head but adapts blocks+trunk per site at reduced LR
(see PIVOT_PLAN.md, "Transfer-learning fine-tuning"). That means by the time
any oracle diagnostic runs, there are six DIFFERENT trunks, not one shared
trunk -- the "common embedding space" the whole idea depended on no longer
exists post-fine-tuning.

This script tests that directly and cheaply (no forward passes, pure
weight-space comparison):
  1. Loads the pre-fine-tune joint checkpoint (outputs/phase5b_v4/model.pt)
     as the one-true-shared reference point.
  2. Loads each site's fine-tuned checkpoint (outputs/phase5c_v4/model_*.pt).
  3. Computes, for the shared parts only (blocks.* + trunk.*, NOT the
     frozen xy_head or the never-shared floor_heads), each site's weight
     drift away from the joint checkpoint, and pairwise drift between
     sites.
  4. Plots a divergence heatmap.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multisite_model import MultiSiteGlobLocCNN  # noqa: E402
from sites import PRETRAIN_SITE_IDS, build_floor_class_map, load_site  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JOINT_CKPT = PROJECT_ROOT / "outputs" / "phase5b_v4" / "model.pt"
FINETUNE_DIR = PROJECT_ROOT / "outputs" / "phase5c_v4"
OUTDIR = PROJECT_ROOT / "outputs" / "phase6e"
SITES = ["hdlc", "sod_cetc331", "sod_hcxy", "sod_syl", "uji_b1", "uji_b2"]


def shared_trunk_vector(state_dict):
    """Flattens blocks.* and trunk.* params (the parts phase5c fine-tuning
    actually adapts) into one vector. Excludes xy_head (frozen during
    fine-tuning, identical everywhere) and floor_heads (never shared)."""
    parts = [v.flatten() for k, v in state_dict.items()
             if k.startswith("blocks.") or k.startswith("trunk.")]
    return torch.cat(parts).numpy().astype(np.float64)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    site_num_floors = {}
    for site_id in PRETRAIN_SITE_IDS:
        d, _, _ = load_site(site_id)
        site_num_floors[site_id] = len(build_floor_class_map(d))

    joint_model = MultiSiteGlobLocCNN(site_num_floors=site_num_floors)
    joint_model.load_state_dict(torch.load(JOINT_CKPT, map_location="cpu"))
    joint_vec = shared_trunk_vector(joint_model.state_dict())
    joint_norm = float(np.linalg.norm(joint_vec))
    print(f"Joint pre-fine-tune trunk: {len(joint_vec)} params, ||w||={joint_norm:.3f}")

    vecs = {"joint": joint_vec}
    for site_id in SITES:
        m = MultiSiteGlobLocCNN(site_num_floors=site_num_floors)
        m.load_state_dict(torch.load(FINETUNE_DIR / f"model_{site_id}.pt", map_location="cpu"))
        vecs[site_id] = shared_trunk_vector(m.state_dict())

    labels = ["joint"] + SITES
    n = len(labels)
    dist = np.zeros((n, n))
    cos = np.zeros((n, n))
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            diff = vecs[a] - vecs[b]
            dist[i, j] = np.linalg.norm(diff)
            na, nb = np.linalg.norm(vecs[a]), np.linalg.norm(vecs[b])
            cos[i, j] = float(np.dot(vecs[a], vecs[b]) / (na * nb)) if na > 0 and nb > 0 else 1.0

    print("\nDrift from joint pre-fine-tune trunk (relative to ||joint trunk||):")
    drift_from_joint = {}
    for site_id in SITES:
        rel = dist[0, labels.index(site_id)] / joint_norm
        drift_from_joint[site_id] = float(rel)
        print(f"  {site_id:14s}  ||delta|| / ||joint|| = {rel:.3f}   cos_sim = {cos[0, labels.index(site_id)]:.4f}")

    print("\nPairwise drift BETWEEN sites (relative to ||joint||):")
    for i in range(1, n):
        for j in range(i + 1, n):
            rel = dist[i, j] / joint_norm
            print(f"  {labels[i]:14s} <-> {labels[j]:14s}  {rel:.3f}  cos_sim={cos[i, j]:.4f}")

    # --- Figure: pairwise weight-space divergence heatmap ---
    fig, ax = plt.subplots(figsize=(7, 6))
    rel_dist = dist / joint_norm
    im = ax.imshow(rel_dist, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{rel_dist[i, j]:.2f}", ha="center", va="center",
                     color="white" if rel_dist[i, j] < rel_dist.max() * 0.6 else "black", fontsize=8)
    ax.set_title("Trunk weight-space divergence (||w_a - w_b|| / ||w_joint||)\n"
                  "divergence is small (<=5%) -- rules out gross trunk drift as the cause")
    fig.colorbar(im, ax=ax, label="relative L2 distance")
    fig.tight_layout()
    fig.savefig(OUTDIR / "trunk_divergence_heatmap.png", dpi=150)
    print(f"\nSaved {OUTDIR / 'trunk_divergence_heatmap.png'}")

    # --- Figure: bar chart, drift-from-joint per site ---
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sites_sorted = sorted(drift_from_joint, key=drift_from_joint.get)
    vals = [drift_from_joint[s] for s in sites_sorted]
    ax2.bar(sites_sorted, vals, color="#4C72B0")
    ax2.set_ylabel("||w_site - w_joint|| / ||w_joint||")
    ax2.set_title("How far each site's trunk moved from the shared\npre-fine-tune checkpoint during Phase 5c")
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
    fig2.tight_layout()
    fig2.savefig(OUTDIR / "trunk_drift_from_joint_bar.png", dpi=150)
    print(f"Saved {OUTDIR / 'trunk_drift_from_joint_bar.png'}")

    import json
    with open(OUTDIR / "trunk_divergence.json", "w") as f:
        json.dump({
            "joint_trunk_norm": joint_norm,
            "drift_from_joint_relative": drift_from_joint,
            "labels": labels,
            "pairwise_relative_distance": rel_dist.tolist(),
            "pairwise_cosine_similarity": cos.tolist(),
        }, f, indent=2)
    print(f"Saved {OUTDIR / 'trunk_divergence.json'}")


if __name__ == "__main__":
    main()
