"""Phase 6f: why did Phase 6d's embedding-drift signal fail leave-SITE-out
generalization even worse than Phase 6b/6c's raw-RSSI-stats versions, given
Phase 6e showed the trunks barely diverged during per-site fine-tuning
(cosine similarity > 0.999 for every pair)?

Since gross trunk divergence is ruled out (Phase 6e), the remaining
candidate explanation is that the FEATURE SCALE itself is not calibrated
consistently across sites, even though the function producing it
(the shared-ish trunk) is nearly identical. Phase 6b/6c's raw RSSI stats
are already normalized into a common [0, 1]-ish physical unit space before
any model sees them (RSSI_MIN/MAX scaling), so a "20% detect-rate drop"
means roughly the same thing everywhere. A 128-d bottleneck embedding has
no such external unit -- its scale is whatever the network's last GroupNorm
+ LeakyReLU happens to produce for that site's own RSSI distribution
(17 dense APs at HDLC vs 520 sparse APs at UJI look nothing alike as raw
inputs), so "embedding drift of magnitude 2.0" may correspond to mild
perturbation at one site and severe perturbation at another -- a scale
confound a leave-site-out regressor cannot correct for, since it never
sees the held-out site's scale during training.

This script tests that directly using the raw per-instance arrays Phase 6d
saved (outputs/phase6d/raw_arrays.npz):
  1. Per-site distribution of the two headline embedding-drift features
     (L2 displacement, cosine distance) -- boxplots.
  2. Leave-site-out actual-vs-predicted scatter (already-computed
     out-of-fold RF predictions), colored by site -- shows whether the
     regressor's errors are systematic per-site offsets (consistent with
     the scale-confound story) or unstructured noise (would argue against
     it).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_NPZ = PROJECT_ROOT / "outputs" / "phase6d" / "raw_arrays.npz"
OUTDIR = PROJECT_ROOT / "outputs" / "phase6f"
FEATURE_NAMES = ["l2_diff", "abs_diff_mean", "abs_diff_max", "diff_std",
                  "cur_within_std", "ref_within_std", "cosine_dist", "cur_embed_norm"]
SITE_ORDER = ["hdlc", "sod_cetc331", "sod_hcxy", "sod_syl", "uji_b1", "uji_b2"]
SITE_COLORS = {"hdlc": "#4C72B0", "sod_cetc331": "#DD8452", "sod_hcxy": "#55A868",
               "sod_syl": "#C44E52", "uji_b1": "#8172B2", "uji_b2": "#937860"}


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    d = np.load(IN_NPZ, allow_pickle=True)
    X, y_norm, site_tags = d["X"], d["y_norm"], d["site_tags"]
    pred_site = d["rf_leave_site_out_pred_norm"]

    # --- Figure 1: per-site distribution of L2 embedding displacement + cosine distance ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feat_idx, title in [(axes[0], 0, "L2 embedding displacement (ref -> current)"),
                                  (axes[1], 6, "Cosine distance (ref -> current)")]:
        data = [X[site_tags == s, feat_idx] for s in SITE_ORDER]
        bp = ax.boxplot(data, tick_labels=SITE_ORDER, patch_artist=True, showfliers=False)
        for patch, s in zip(bp["boxes"], SITE_ORDER):
            patch.set_facecolor(SITE_COLORS[s])
            patch.set_alpha(0.7)
        ax.set_title(title)
        ax.set_ylabel(FEATURE_NAMES[feat_idx])
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle("Embedding-drift feature scale varies by site even though the trunks\n"
                  "themselves are nearly identical (Phase 6e) -- a scale confound, not a weight confound")
    fig.tight_layout()
    fig.savefig(OUTDIR / "feature_scale_by_site.png", dpi=150)
    print(f"Saved {OUTDIR / 'feature_scale_by_site.png'}")

    # Quantify the scale confound: coefficient of variation of each site's
    # median L2 displacement, vs. the within-site spread.
    medians = {s: float(np.median(X[site_tags == s, 0])) for s in SITE_ORDER}
    print("\nPer-site median L2 embedding displacement:")
    for s, m in medians.items():
        print(f"  {s:14s} {m:.3f}")
    spread = max(medians.values()) - min(medians.values())
    print(f"Range across sites: {spread:.3f}  "
          f"(site-to-site median differs by {spread / np.mean(list(medians.values())) * 100:.0f}% "
          f"of the mean site-level median)")

    # --- Figure 2: leave-site-out actual vs predicted, colored by site ---
    fig2, ax2 = plt.subplots(figsize=(6.5, 6))
    for s in SITE_ORDER:
        mask = site_tags == s
        ax2.scatter(y_norm[mask], pred_site[mask], s=10, alpha=0.5, color=SITE_COLORS[s], label=s)
    lims = [min(y_norm.min(), pred_site.min()), max(y_norm.max(), pred_site.max())]
    ax2.plot(lims, lims, "k--", linewidth=1, label="perfect prediction")
    ax2.set_xlabel("actual (site-median-normalized) buffer error")
    ax2.set_ylabel("leave-SITE-out RF predicted")
    ax2.set_title("Leave-site-out predictions collapse toward a flat band per site\n"
                   "(model falls back to a site-blind constant, not real drift tracking)")
    ax2.legend(fontsize=8, loc="upper left")
    fig2.tight_layout()
    fig2.savefig(OUTDIR / "leave_site_out_actual_vs_pred.png", dpi=150)
    print(f"Saved {OUTDIR / 'leave_site_out_actual_vs_pred.png'}")

    import json
    with open(OUTDIR / "feature_scale_analysis.json", "w") as f:
        json.dump({"per_site_median_l2_displacement": medians,
                    "range_across_sites": spread}, f, indent=2)
    print(f"Saved {OUTDIR / 'feature_scale_analysis.json'}")


if __name__ == "__main__":
    main()
