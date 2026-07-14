"""Phase 5c: per-site fine-tuning on top of the Phase 5b v2 joint backbone.

Freezes xy_head (preserve the validated zero-shot xy transfer -- the whole
point of the shared representation), fine-tunes the trunk + that site's own
floor_head at a reduced LR, per site, starting from a FRESH copy of the v2
checkpoint each time (so one site's fine-tune never contaminates another's).
Produces one specialized checkpoint per site; the original v2 checkpoint
remains the "foundation model" artifact for new/zero-shot sites.
"""
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import HyperParams  # noqa: E402
from multisite_dataset import MultiSiteDataset  # noqa: E402
from multisite_model import MultiSiteGlobLocCNN  # noqa: E402
from multisite_train import run_epoch_multisite  # noqa: E402
from naive_baseline import naive_constant_baseline  # noqa: E402
from sites import (  # noqa: E402
    PRETRAIN_SITE_IDS, attach_floor_class, build_floor_class_map,
    filter_locatable_rows, load_site,
)
from splits import grouped_split_by_column  # noqa: E402

V2_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase5b_v2"
OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "phase5c_v2"
FINETUNE_EPOCHS = 10  # v1 was 5 epochs + xy-only model selection (outputs/phase5c,
                       # kept for comparison). v2 doubles epochs and selects on
                       # combined 3D error instead -- see PIVOT_PLAN.md
                       # "model-selection bugfix".
FINETUNE_LR = 3e-5  # v2's base LR (1e-4) / ~3, reduced since this is
                     # adapting an already-converged shared backbone, not
                     # training from scratch
SEED = 42


def naive_baseline_generic(train_df, eval_df):
    tr = train_df.rename(columns={"floor_id": "floor"})
    ev = eval_df.rename(columns={"floor_id": "floor"})
    return naive_constant_baseline(tr, ev)


def finetune_one_site(base_state, site_num_floors, site_id, train_df, val_df, test_df,
                       wap_pos, wap_cols, hp, device="cpu"):
    model = MultiSiteGlobLocCNN(site_num_floors=site_num_floors).to(device)
    model.load_state_dict(base_state)

    for p in model.xy_head.parameters():
        p.requires_grad = False

    train_ds = MultiSiteDataset({site_id: train_df}, {site_id: (wap_pos, wap_cols)}, hp, augment=True, seed=SEED)
    val_ds = MultiSiteDataset({site_id: val_df}, {site_id: (wap_pos, wap_cols)}, hp, augment=False, seed=SEED)
    test_ds = MultiSiteDataset({site_id: test_df}, {site_id: (wap_pos, wap_cols)}, hp, augment=False, seed=SEED)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=FINETUNE_LR)

    best_val, best_state = float("inf"), None
    for ep in range(1, FINETUNE_EPOCHS + 1):
        train_pooled, _ = run_epoch_multisite(model, train_ds, hp, optimizer, device)
        val_pooled, val_per_site = run_epoch_multisite(model, val_ds, hp, optimizer=None, device=device)
        m = val_per_site[site_id]
        print(f"    ft epoch {ep}/{FINETUNE_EPOCHS} | train_xy {train_pooled['median_xy_err_m']:.2f}m | "
              f"val_xy {m['median_xy_err_m']:.2f}m val_floor_acc {m['floor_accuracy']:.3f} "
              f"val_3d {m['median_3d_err_m']:.2f}m")
        # Selection on combined 3D error (xy + floor), not xy alone -- see
        # PIVOT_PLAN.md "model-selection bugfix".
        val_selection_metric = val_pooled["median_3d_err_m"]
        if not math.isnan(val_selection_metric) and val_selection_metric < best_val:
            best_val = val_selection_metric
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_per_site = run_epoch_multisite(model, test_ds, hp, optimizer=None, device=device)
    return model, test_per_site[site_id]


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hp = HyperParams()

    print("Loading v2 joint checkpoint + re-deriving splits/site_num_floors...")
    site_data = {}
    site_num_floors = {}
    for site_id in PRETRAIN_SITE_IDS:
        df, wap_pos, wap_cols = load_site(site_id)
        floor_map = build_floor_class_map(df)
        df = attach_floor_class(df, floor_map)
        df = filter_locatable_rows(df, wap_pos, wap_cols, hp.k_strongest, site_id)
        train_df, val_df, test_df = grouped_split_by_column(df, "_group", val_frac=0.15, test_frac=0.15, seed=SEED)
        site_data[site_id] = (train_df, val_df, test_df, wap_pos, wap_cols)
        site_num_floors[site_id] = len(floor_map)

    base_model = MultiSiteGlobLocCNN(site_num_floors=site_num_floors).to(device)
    base_model.load_state_dict(torch.load(V2_DIR / "model.pt", map_location=device))
    base_state = copy.deepcopy(base_model.state_dict())

    with open(V2_DIR / "results.json") as f:
        v2_results = json.load(f)["per_site_test"]

    print("\n" + "=" * 78)
    print("Per-site fine-tuning (xy_head frozen, trunk + own floor_head adapted)")
    print("=" * 78)
    summary = {}
    for site_id in PRETRAIN_SITE_IDS:
        print(f"\n--- {site_id} ---")
        train_df, val_df, test_df, wap_pos, wap_cols = site_data[site_id]
        naive = naive_baseline_generic(train_df, test_df)

        t0 = time.time()
        model, test_m = finetune_one_site(base_state, site_num_floors, site_id, train_df, val_df, test_df,
                                           wap_pos, wap_cols, hp, device=device)
        dt = time.time() - t0

        v2_m = v2_results[site_id]["model"]
        print(f"[{site_id:12s}] v2 (joint)   xy={v2_m['median_xy_err_m']:6.2f}m floor_acc={v2_m['floor_accuracy']:.3f}")
        print(f"[{site_id:12s}] v3 (finetune) xy={test_m['median_xy_err_m']:6.2f}m floor_acc={test_m['floor_accuracy']:.3f}  "
              f"({dt:.0f}s)")
        print(f"[{site_id:12s}] naive         xy={naive['median_xy_err_m']:6.2f}m floor_acc={naive['floor_accuracy']:.3f}")

        summary[site_id] = {"v2_joint": v2_m, "v3_finetuned": test_m, "naive": naive}
        torch.save(model.state_dict(), OUTDIR / f"model_{site_id}.pt")

    print("\n" + "=" * 78)
    print("Summary: v2 (joint) vs v3 (fine-tuned) vs naive")
    print("=" * 78)
    for site_id, s in summary.items():
        print(f"{site_id:12s} | v2 floor_acc={s['v2_joint']['floor_accuracy']:.3f} -> "
              f"v3 floor_acc={s['v3_finetuned']['floor_accuracy']:.3f} "
              f"(naive={s['naive']['floor_accuracy']:.3f})")

    with open(OUTDIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
