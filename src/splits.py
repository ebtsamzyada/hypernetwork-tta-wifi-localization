"""Grouped (leave-physical-point-out) splitting, plus a diagnostic for the
HDLC dataset's official row-level split -- see PIVOT_PLAN.md, "Official
train/test split is row-level, not grouped by point".
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def _point_group_ids(df: pd.DataFrame) -> pd.Series:
    return (
        df["floor"].astype(str) + "_" +
        df["x"].astype(str) + "_" +
        df["y"].astype(str)
    )


def assert_disjoint_point_groups(*dfs: pd.DataFrame) -> None:
    group_sets = [set(_point_group_ids(d)) for d in dfs]
    for i in range(len(group_sets)):
        for j in range(i + 1, len(group_sets)):
            overlap = group_sets[i] & group_sets[j]
            assert not overlap, (
                f"Grouped split leaked {len(overlap)} physical point(s) "
                f"across splits {i} and {j}: {sorted(overlap)[:5]}"
            )
    print(f"Disjointness OK: {[len(s) for s in group_sets]} distinct physical "
          f"points, zero overlap across {len(group_sets)} splits.")


def grouped_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """Split by unique physical (floor, x, y) point -- never by row. Returns
    (train_df, val_df, test_df) with an explicit disjointness assertion."""
    groups = _point_group_ids(df)

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    trainval_groups = _point_group_ids(trainval_df)
    relative_val_frac = val_frac / (1 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=seed)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_groups))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    assert_disjoint_point_groups(train_df, val_df, test_df)
    return train_df, val_df, test_df


def grouped_split_by_column(df: pd.DataFrame, group_col: str = "_group",
                             val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """Same as grouped_split, but the group key is an arbitrary precomputed
    column instead of hardcoded (floor,x,y) -- used for the multi-site pool,
    where the leakage-safe group differs per site (physical point for
    HDLC/SODIndoorLoc, capture session for UJIIndoorLoc -- see sites.py)."""
    groups = df[group_col]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(df, groups=groups))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    trainval_groups = trainval_df[group_col]
    relative_val_frac = val_frac / (1 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_frac, random_state=seed)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_groups))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    for a, b, name_a, name_b in [(train_df, val_df, "train", "val"), (train_df, test_df, "train", "test"),
                                  (val_df, test_df, "val", "test")]:
        overlap = set(a[group_col]) & set(b[group_col])
        assert not overlap, f"Grouped split leaked {len(overlap)} group(s) across {name_a}/{name_b}: {sorted(overlap)[:5]}"

    return train_df, val_df, test_df


def official_split_point_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> set:
    """Diagnostic only -- quantifies how much the dataset's published
    Training/Testing CSVs violate grouped-split discipline. Does not raise;
    this is expected/known, just kept visible every run."""
    train_groups = set(_point_group_ids(train_df))
    test_groups = set(_point_group_ids(test_df))
    overlap = train_groups & test_groups
    pct = len(overlap) / max(1, len(test_groups))
    print(
        f"Official split: {len(train_groups)} train points, {len(test_groups)} "
        f"test points, {len(overlap)} points appear in BOTH ({pct:.0%} of test "
        f"points already seen during training) -- this is a row-level split, "
        f"NOT grouped. Treat any metric on this split as same-point "
        f"interpolation, not spatial generalization."
    )
    return overlap
