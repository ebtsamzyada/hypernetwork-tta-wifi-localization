"""Phase 3: drift sanity check. HDLC's Layout 1 -> Layout 2 -> Layout 3 is a
*real*, physically-caused RSSI drift (partition boards added as obstructions;
Layout 2 = boards on ground+2nd floor only, Layout 3 = boards on every
floor), not a synthetic injector -- see PIVOT_PLAN.md Item 1.

This script freezes the Phase 2 model (trained only on Layout 1) and
evaluates it, unmodified, on the SAME 58 held-out physical points across all
three layouts. Using the same points across conditions isolates the effect
of layout change from a "some points are just harder" confound. If this
degrades monotonically (L1 test < L2 < L3), the frozen baseline is a valid
target for the Phase 4/5 hypernetwork TTA correction.
"""
import json
import sys
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
from splits import grouped_split  # noqa: E402
from train import run_epoch  # noqa: E402

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "Hybrid-fingerprint Data with Layout Change (HDLC)"
PHASE2_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase2"
OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "phase3"


def _point_id(row):
    return f"{row['floor']}_{row['x']}_{row['y']}"


def load_layout_pool(layout_dir, layout_label):
    train_raw, wap_pos, wap_cols = load_raw_layout_csv(layout_dir / f"{layout_label} - Raw - Training.csv")
    test_raw, _, _ = load_raw_layout_csv(layout_dir / f"{layout_label} - Raw - Testing.csv")
    return pd.concat([train_raw, test_raw], ignore_index=True), wap_pos, wap_cols


def main(seed=42):
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # ---- reconstruct the exact Phase 2 grouped split to recover the same
    # 58 held-out test points (same function, same seed, same input pool).
    l1_pool, wap_pos, wap_cols = load_layout_pool(DATA_ROOT / "Layout 1", "Layout 1")
    floor_map = build_floor_class_map(l1_pool)
    l1_pool = attach_floor_class(l1_pool, floor_map)
    l1_pool = filter_locatable_rows(l1_pool, wap_pos, wap_cols, HyperParams().k_strongest, "Layout 1 pool")
    _, _, l1_test_df = grouped_split(l1_pool, val_frac=0.15, test_frac=0.15, seed=seed)
    held_out_point_ids = set(_point_id(r) for _, r in l1_test_df.iterrows())
    print(f"Reusing {len(held_out_point_ids)} held-out physical points from Phase 2's grouped test split.\n")

    # ---- load Layout 2 / Layout 3 pools, restrict to the SAME held-out points,
    # reusing Layout 1's wap_positions (physically shared -- verified identical).
    conditions = {"Layout 1 (reference, no obstruction)": l1_test_df}
    for label, dirname in [("Layout 2 (partial obstruction)", "Layout 2"), ("Layout 3 (full obstruction)", "Layout 3")]:
        pool, _, _ = load_layout_pool(DATA_ROOT / dirname, dirname)
        pool = attach_floor_class(pool, floor_map)
        pool = filter_locatable_rows(pool, wap_pos, wap_cols, HyperParams().k_strongest, dirname)
        subset = pool[pool.apply(lambda r: _point_id(r) in held_out_point_ids, axis=1)].reset_index(drop=True)
        conditions[label] = subset

    # ---- load the frozen Phase 2 model
    model = GlobLocCNN(num_floors=len(floor_map))
    model.load_state_dict(torch.load(PHASE2_DIR / "model.pt", map_location="cpu"))
    model.eval()
    hp = HyperParams()

    results = {}
    print(f"{'Condition':45s} {'n':>6s} {'median_xy_err_m':>16s} {'floor_acc':>10s} {'median_3d_err_m':>16s}")
    for label, df in conditions.items():
        ds = GlobLocDataset(df, wap_pos, wap_cols, hp, augment=False, seed=seed)
        metrics, _ = run_epoch(model, ds, hp, optimizer=None, device="cpu")
        results[label] = {**metrics, "n": len(df)}
        print(f"{label:45s} {len(df):6d} {metrics['median_xy_err_m']:16.2f} "
              f"{metrics['floor_accuracy']:10.3f} {metrics['median_3d_err_m']:16.2f}")

    xy_errs = [results[k]["median_xy_err_m"] for k in conditions]
    monotonic = all(xy_errs[i] <= xy_errs[i + 1] for i in range(len(xy_errs) - 1))
    print(f"\nMonotonic degradation (xy error) as severity increases: {monotonic}")
    if not monotonic:
        print("*** WARNING: drift did not degrade the frozen model monotonically. "
              "Investigate before treating this as a valid drift testbed for TTA. ***")

    with open(OUTDIR / "drift_results.json", "w") as f:
        json.dump({"held_out_points": len(held_out_point_ids), "results": results,
                   "monotonic": monotonic}, f, indent=2)
    print(f"\nSaved to {OUTDIR / 'drift_results.json'}")


if __name__ == "__main__":
    main()
