"""Phase 5a: multi-site loading + grouped-split validation, BEFORE any
multi-site model training code. Mirrors Phase 1's discipline (Phase 1
report, PIVOT_PLAN.md) but across all 7 sites: 6 pretraining-pool sites
get their own leakage-free grouped split + naive baseline; the held-out
zero-shot site (uji_b0) is loaded and inventoried only, no split needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from naive_baseline import naive_constant_baseline  # noqa: E402
from sites import (  # noqa: E402
    HELDOUT_SITE_ID, PRETRAIN_SITE_IDS,
    attach_floor_class, build_floor_class_map, filter_locatable_rows, load_site,
)
from splits import grouped_split_by_column  # noqa: E402

HP_K_STRONGEST = 5  # matches dataset.HyperParams default


def naive_baseline_multisite(train_df, eval_df):
    """naive_baseline.naive_constant_baseline expects 'x','y','floor' columns
    -- this schema uses 'floor_id', so adapt with a thin rename."""
    tr = train_df.rename(columns={"floor_id": "floor"})
    ev = eval_df.rename(columns={"floor_id": "floor"})
    return naive_constant_baseline(tr, ev)


def main():
    print("=" * 78)
    print("Pretraining-pool sites (own grouped split + naive baseline each)")
    print("=" * 78)
    rows = []
    for site_id in PRETRAIN_SITE_IDS:
        df, wap_pos, wap_cols = load_site(site_id)
        floor_map = build_floor_class_map(df)
        df = attach_floor_class(df, floor_map)
        df = filter_locatable_rows(df, wap_pos, wap_cols, HP_K_STRONGEST, site_id)

        train_df, val_df, test_df = grouped_split_by_column(df, "_group", val_frac=0.15, test_frac=0.15, seed=42)
        naive = naive_baseline_multisite(train_df, test_df)

        rows.append({
            "site_id": site_id, "n_rows": len(df), "n_groups": df["_group"].nunique(),
            "n_floors": len(floor_map), "n_aps": len(wap_cols),
            "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
            "naive_median_xy_err_m": naive["median_xy_err_m"], "naive_floor_acc": naive["floor_accuracy"],
        })
        print()

    print("=" * 78)
    print("Held-out zero-shot site (inventory only, no split)")
    print("=" * 78)
    ho_df, ho_wap_pos, ho_wap_cols = load_site(HELDOUT_SITE_ID)
    ho_floor_map = build_floor_class_map(ho_df)
    ho_df = attach_floor_class(ho_df, ho_floor_map)
    ho_df = filter_locatable_rows(ho_df, ho_wap_pos, ho_wap_cols, HP_K_STRONGEST, HELDOUT_SITE_ID)
    print(f"[{HELDOUT_SITE_ID}] {len(ho_df)} usable rows, {ho_df['_group'].nunique()} groups, "
          f"{len(ho_floor_map)} floors -- reserved for zero-shot eval only, never trained on.")

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    header = f"{'site':14s} {'rows':>7s} {'grps':>6s} {'flrs':>5s} {'APs':>5s} {'train':>7s} {'val':>6s} {'test':>6s} {'naive_xy_m':>11s} {'naive_flracc':>13s}"
    print(header)
    for r in rows:
        print(f"{r['site_id']:14s} {r['n_rows']:7d} {r['n_groups']:6d} {r['n_floors']:5d} {r['n_aps']:5d} "
              f"{r['n_train']:7d} {r['n_val']:6d} {r['n_test']:6d} "
              f"{r['naive_median_xy_err_m']:11.2f} {r['naive_floor_acc']:13.3f}")


if __name__ == "__main__":
    main()
