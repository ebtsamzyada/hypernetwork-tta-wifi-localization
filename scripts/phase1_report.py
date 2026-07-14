"""Phase 1 sanity report: load HDLC Layout 1, build both the official
(row-level) split and a genuine grouped (leave-points-out) split, assert
disjointness on the grouped one, and report the naive constant baseline on
both -- so any future model result has an honest floor to beat, and the
row-level-leakage gap is visible on every run, not just in PIVOT_PLAN.md.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loading import build_floor_class_map, load_raw_layout_csv  # noqa: E402
from naive_baseline import naive_constant_baseline  # noqa: E402
from splits import (  # noqa: E402
    assert_disjoint_point_groups,
    grouped_split,
    official_split_point_overlap,
)

LAYOUT1_DIR = (
    Path(__file__).resolve().parents[1]
    / "data" / "raw" / "Hybrid-fingerprint Data with Layout Change (HDLC)" / "Layout 1"
)


def main():
    print("=" * 70)
    print("Loading HDLC Layout 1 raw CSVs (WiFi-only)")
    print("=" * 70)
    train_df, wap_pos, wap_cols = load_raw_layout_csv(LAYOUT1_DIR / "Layout 1 - Raw - Training.csv")
    test_df, _, _ = load_raw_layout_csv(LAYOUT1_DIR / "Layout 1 - Raw - Testing.csv")

    floor_map = build_floor_class_map(pd.concat([train_df, test_df], ignore_index=True))
    print(f"\nFloors present: {sorted(floor_map.keys())} -> classes {list(floor_map.values())}")

    print("\n" + "=" * 70)
    print("Official (row-level) split -- as published by the dataset authors")
    print("=" * 70)
    official_split_point_overlap(train_df, test_df)
    official_baseline = naive_constant_baseline(train_df, test_df)
    print(f"Naive constant baseline (official split): {official_baseline}")

    print("\n" + "=" * 70)
    print("Grouped (leave-physical-points-out) split -- built from the full pool")
    print("=" * 70)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    g_train, g_val, g_test = grouped_split(full_df, val_frac=0.15, test_frac=0.15, seed=42)
    assert_disjoint_point_groups(g_train, g_val, g_test)
    print(f"train={len(g_train)} rows, val={len(g_val)} rows, test={len(g_test)} rows")

    grouped_baseline = naive_constant_baseline(g_train, g_test)
    print(f"Naive constant baseline (grouped split): {grouped_baseline}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        f"Naive median xy error -- official split: "
        f"{official_baseline['median_xy_err_m']:.2f} m | grouped split: "
        f"{grouped_baseline['median_xy_err_m']:.2f} m"
    )
    print(
        f"Naive floor accuracy  -- official split: "
        f"{official_baseline['floor_accuracy']:.3f} | grouped split: "
        f"{grouped_baseline['floor_accuracy']:.3f}"
    )
    print(
        "\nAny Phase 2 model must beat BOTH baselines on BOTH splits to be "
        "meaningful. The grouped split is the one that actually measures "
        "spatial generalization -- see PIVOT_PLAN.md."
    )


if __name__ == "__main__":
    main()
