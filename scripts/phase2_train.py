"""Phase 2: base foundation network (GlobLoc CNN+SPP, WiFi-only), trained
on the Phase 1 grouped (leakage-free) split, reported against the Phase 1
naive constant baseline. Official-split numbers are NOT used for training
or model selection -- see PIVOT_PLAN.md on why that split is contaminated.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loading import (  # noqa: E402
    attach_floor_class,
    build_floor_class_map,
    filter_locatable_rows,
    load_raw_layout_csv,
)
from dataset import GlobLocDataset, HyperParams  # noqa: E402
from model import GlobLocCNN  # noqa: E402
from naive_baseline import naive_constant_baseline  # noqa: E402
from splits import assert_disjoint_point_groups, grouped_split  # noqa: E402
from train import run_epoch, train_model  # noqa: E402

LAYOUT1_DIR = (
    Path(__file__).resolve().parents[1]
    / "data" / "raw" / "Hybrid-fingerprint Data with Layout Change (HDLC)" / "Layout 1"
)
OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "phase2"


def main(epochs=10, seed=42):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_raw, wap_pos, wap_cols = load_raw_layout_csv(LAYOUT1_DIR / "Layout 1 - Raw - Training.csv")
    test_raw, _, _ = load_raw_layout_csv(LAYOUT1_DIR / "Layout 1 - Raw - Testing.csv")
    full_df = pd.concat([train_raw, test_raw], ignore_index=True)

    floor_map = build_floor_class_map(full_df)
    full_df = attach_floor_class(full_df, floor_map)
    num_floors = len(floor_map)
    print(f"Floors: {sorted(floor_map.keys())} -> {num_floors} classes")

    full_df = filter_locatable_rows(full_df, wap_pos, wap_cols, HyperParams().k_strongest, "full pool")

    train_df, val_df, test_df = grouped_split(full_df, val_frac=0.15, test_frac=0.15, seed=seed)
    assert_disjoint_point_groups(train_df, val_df, test_df)
    print(f"Grouped split -> train={len(train_df)} val={len(val_df)} test={len(test_df)} rows")

    naive = naive_constant_baseline(train_df, test_df)
    print(f"\nNaive constant baseline (grouped test): median_xy_err={naive['median_xy_err_m']:.2f}m "
          f"floor_acc={naive['floor_accuracy']:.3f}\n")

    hp = HyperParams()
    train_ds = GlobLocDataset(train_df, wap_pos, wap_cols, hp, augment=True, seed=seed)
    val_ds = GlobLocDataset(val_df, wap_pos, wap_cols, hp, augment=False, seed=seed)
    test_ds = GlobLocDataset(test_df, wap_pos, wap_cols, hp, augment=False, seed=seed)

    model = GlobLocCNN(num_floors=num_floors).to(device)

    t0 = time.time()
    model, history, best_val = train_model(model, train_ds, val_ds, hp, epochs=epochs, device=device)
    print(f"\nTraining took {time.time() - t0:.1f}s for {epochs} epochs")

    test_metrics, _ = run_epoch(model, test_ds, hp, optimizer=None, device=device)
    print(f"\n=== FINAL: grouped test set (leakage-free spatial holdout) ===")
    print(f"  Model  -- median_xy_err={test_metrics['median_xy_err_m']:.2f}m "
          f"floor_acc={test_metrics['floor_accuracy']:.3f} "
          f"median_3d_err={test_metrics['median_3d_err_m']:.2f}m")
    print(f"  Naive  -- median_xy_err={naive['median_xy_err_m']:.2f}m "
          f"floor_acc={naive['floor_accuracy']:.3f}")

    if test_metrics["median_xy_err_m"] >= naive["median_xy_err_m"]:
        print("\n*** WARNING: model does NOT beat the naive constant-xy baseline "
              "on the grouped test set. Do not trust this model. ***")
    if test_metrics["floor_accuracy"] <= naive["floor_accuracy"]:
        print("\n*** WARNING: model does NOT beat the naive majority-floor baseline "
              "on the grouped test set. Do not trust this model. ***")

    with open(OUTDIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(OUTDIR / "results.json", "w") as f:
        json.dump({
            "hyperparams": hp.as_dict(),
            "naive_baseline": naive,
            "test_metrics": test_metrics,
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        }, f, indent=2)
    torch.save(model.state_dict(), OUTDIR / "model.pt")
    print(f"\nSaved outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
