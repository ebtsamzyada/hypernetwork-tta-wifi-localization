"""Phase 5d: error analysis. Where are the remaining errors coming from?

For each pretraining site, loads the BEST available checkpoint (the
early-stopped ones for uji_b1/uji_b2, phase5c_v2 for the rest -- see
PIVOT_PLAN.md for why those are the best per-site artifacts), runs
inference on that site's own test split, and produces:
  1. A spatial heatmap of xy error (does error cluster in certain areas,
     e.g. building edges or AP-sparse regions?).
  2. A floor confusion matrix, for multi-floor sites (which floors get
     mistaken for which -- tests the "floors stack directly on top of
     each other" hypothesis for UJI specifically).
  3. A pooled scatter of xy error vs. how many positioned APs were
     actually visible in that row (tests the "sparse AP-position coverage
     drives error" hypothesis quantitatively, not just narratively).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import HyperParams  # noqa: E402
from multisite_model import MultiSiteGlobLocCNN  # noqa: E402
from sites import (  # noqa: E402
    RSSI_MAX, RSSI_MIN, PRETRAIN_SITE_IDS,
    attach_floor_class, build_floor_class_map, filter_locatable_rows, load_site,
)
from splits import grouped_split_by_column  # noqa: E402
from virtual_space import PIXEL_METERS, build_virtual_space, generate_image  # noqa: E402

torch.set_num_threads(1)

OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "phase5d_error_analysis"
FLOOR_HEIGHT_M = 3.5
SEED = 42

CHECKPOINTS = {
    "hdlc": Path("outputs/phase5c_v2/model_hdlc.pt"),
    "sod_cetc331": Path("outputs/phase5c_v2/model_sod_cetc331.pt"),
    "sod_hcxy": Path("outputs/phase5c_v2/model_sod_hcxy.pt"),
    "sod_syl": Path("outputs/phase5c_v2/model_sod_syl.pt"),
    "uji_b1": Path("outputs/phase5c_uji_early_stop/model_uji_b1.pt"),
    "uji_b2": Path("outputs/phase5c_uji_early_stop/model_uji_b2.pt"),
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def run_diagnostic_inference(model, df, wap_pos, wap_cols, site_id, hp):
    """Returns a list of per-row dicts with x, y, floor_id, pred_floor_class,
    xy_err_m, n_positioned_visible -- everything needed for the plots."""
    model.eval()
    records = []
    for _, row in df.iterrows():
        x, y = float(row["x"]), float(row["y"])
        floor_id = int(row["floor_id"])
        floor_class = int(row["floor_class"])
        rssi_row = row[wap_cols].to_numpy(dtype=np.float32)

        visible_idx = np.where(rssi_row > RSSI_MIN)[0]
        n_positioned_visible = sum(1 for i in visible_idx if wap_cols[i] in wap_pos)

        ap_info, (vx, vy), (x_min, y_min), img_hw = build_virtual_space(
            x, y, rssi_row, wap_cols, wap_pos, hp.k_strongest, hp.margin_m, hp.max_extent_m,
            rssi_min=RSSI_MIN, rssi_max=RSSI_MAX,
        )
        img = torch.from_numpy(generate_image(ap_info, img_hw)).unsqueeze(0)
        xy_pred, floor_pred = model(img, site_id)
        pred_xy = xy_pred.numpy()[0] * PIXEL_METERS + np.array([x_min, y_min])
        xy_err = float(np.linalg.norm(pred_xy - np.array([x, y])))
        num_floors = model.site_num_floors[site_id]
        raw_floor = float(floor_pred.item())
        pred_floor_class = int(round(min(max(raw_floor, 0.0), num_floors - 1)))

        records.append({
            "x": x, "y": y, "floor_id": floor_id,
            "true_floor_class": floor_class, "pred_floor_class": pred_floor_class,
            "xy_err_m": xy_err, "n_positioned_visible": n_positioned_visible,
        })
    return records


def plot_spatial_heatmap(records, site_id, outpath):
    floors = sorted(set(r["floor_id"] for r in records))
    fig, axes = plt.subplots(1, len(floors), figsize=(5.5 * len(floors), 5), squeeze=False)
    axes = axes[0]
    all_err = [r["xy_err_m"] for r in records]
    vmax = np.percentile(all_err, 95)
    for ax, floor_id in zip(axes, floors):
        recs = [r for r in records if r["floor_id"] == floor_id]
        xs = [r["x"] for r in recs]
        ys = [r["y"] for r in recs]
        errs = [r["xy_err_m"] for r in recs]
        sc = ax.scatter(xs, ys, c=errs, cmap="inferno_r", vmin=0, vmax=vmax, s=40, edgecolors="k", linewidths=0.3)
        ax.set_title(f"{site_id} floor {floor_id} (n={len(recs)})")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        fig.colorbar(sc, ax=ax, label="xy error (m)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def plot_floor_confusion(records, site_id, outpath):
    floor_classes = sorted(set(r["true_floor_class"] for r in records) | set(r["pred_floor_class"] for r in records))
    n = len(floor_classes)
    idx = {c: i for i, c in enumerate(floor_classes)}
    confusion = np.zeros((n, n), dtype=int)
    for r in records:
        confusion[idx[r["true_floor_class"]], idx[r["pred_floor_class"]]] += 1

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(floor_classes)
    ax.set_yticks(range(n)); ax.set_yticklabels(floor_classes)
    ax.set_xlabel("Predicted floor class")
    ax.set_ylabel("True floor class")
    ax.set_title(f"{site_id} floor confusion")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                     color="white" if confusion[i, j] > confusion.max() / 2 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)
    return confusion, floor_classes


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    hp = HyperParams()

    print("Re-deriving splits/site_num_floors for all 6 pretraining sites...")
    site_data, site_num_floors = {}, {}
    for site_id in PRETRAIN_SITE_IDS:
        df, wap_pos, wap_cols = load_site(site_id)
        floor_map = build_floor_class_map(df)
        df = attach_floor_class(df, floor_map)
        df = filter_locatable_rows(df, wap_pos, wap_cols, hp.k_strongest, site_id)
        _, _, test_df = grouped_split_by_column(df, "_group", val_frac=0.15, test_frac=0.15, seed=SEED)
        site_data[site_id] = (test_df, wap_pos, wap_cols)
        site_num_floors[site_id] = len(floor_map)

    all_records = {}
    for site_id in PRETRAIN_SITE_IDS:
        print(f"\n--- {site_id} ---")
        test_df, wap_pos, wap_cols = site_data[site_id]

        model = MultiSiteGlobLocCNN(site_num_floors=site_num_floors)
        model.load_state_dict(torch.load(PROJECT_ROOT / CHECKPOINTS[site_id], map_location="cpu"))

        records = run_diagnostic_inference(model, test_df, wap_pos, wap_cols, site_id, hp)
        all_records[site_id] = records

        plot_spatial_heatmap(records, site_id, OUTDIR / f"{site_id}_spatial_error.png")
        print(f"[{site_id}] saved spatial heatmap")

        floors_here = set(r["floor_id"] for r in records)
        if len(floors_here) > 1:
            confusion, floor_classes = plot_floor_confusion(records, site_id, OUTDIR / f"{site_id}_floor_confusion.png")
            print(f"[{site_id}] saved floor confusion matrix, classes={floor_classes}")
            print(f"[{site_id}] confusion matrix:\n{confusion}")
        else:
            print(f"[{site_id}] single floor, skipping confusion matrix")

    print("\n" + "=" * 78)
    print("AP-sparsity hypothesis: xy error vs. n positioned-visible APs")
    print("=" * 78)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(PRETRAIN_SITE_IDS)))
    for site_id, color in zip(PRETRAIN_SITE_IDS, colors):
        recs = all_records[site_id]
        n_pos = np.array([r["n_positioned_visible"] for r in recs])
        errs = np.array([r["xy_err_m"] for r in recs])
        if len(set(n_pos)) > 1:
            corr = np.corrcoef(n_pos, errs)[0, 1]
        else:
            corr = float("nan")
        print(f"[{site_id:12s}] n_positioned_visible: min={n_pos.min()} max={n_pos.max()} "
              f"mean={n_pos.mean():.1f}  |  corr(n_positioned, xy_err) = {corr:+.3f}")
        # jitter x slightly per site so overlapping integer counts are visible
        jitter = (np.random.default_rng(hash(site_id) % 2**32).random(len(n_pos)) - 0.5) * 0.3
        ax.scatter(n_pos + jitter, errs, s=8, alpha=0.4, color=color, label=site_id)
    ax.set_xlabel("# positioned APs visible in this reading")
    ax.set_ylabel("xy error (m)")
    ax.set_yscale("log")
    ax.set_title("Error vs. AP-position coverage, all sites pooled")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(OUTDIR / "ap_sparsity_vs_error.png", dpi=140)
    plt.close(fig)
    print(f"\nSaved cross-site AP-sparsity plot to {OUTDIR / 'ap_sparsity_vs_error.png'}")
    print(f"All figures saved to {OUTDIR}")


if __name__ == "__main__":
    main()
