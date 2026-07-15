"""Grouped (leave-physical-point-out) splitting, plus a diagnostic for the
HDLC dataset's official row-level split -- see PIVOT_PLAN.md, "Official
train/test split is row-level, not grouped by point".
"""
import numpy as np
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


def stratified_grouped_split_by_floor(df: pd.DataFrame, group_col: str = "_group", floor_col: str = "floor_id",
                                       val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42):
    """Same grouping discipline as grouped_split_by_column, but tries to
    give every floor class present in `df` representation in TEST and VAL
    too -- WITHOUT ever starving TRAIN of a floor it could otherwise cover.

    v1 of this function (see PIVOT_PLAN.md "Phase 5d") guaranteed test/val
    coverage unconditionally, which for uji_b2 (only 16 groups, 5 floors)
    meant floor 4's only 2 groups both went to test+val, leaving TRAIN with
    zero floor-4 examples -- the model could then never learn to predict
    that floor at all, which is a WORSE failure mode than the original
    problem (a floor merely absent from test just isn't scored; a floor
    absent from train guarantees wrong predictions whenever it's the true
    answer). Fixed: a group is only pulled into test/val if doing so still
    leaves every floor it touches with >=1 other group remaining for
    train. Floors too scarce to get test/val representation this way are
    reported as train-only, not silently dropped from the log.
    """
    rng = np.random.default_rng(seed)
    group_floors_s = df.groupby(group_col)[floor_col].apply(lambda s: set(s.unique()))
    all_floors = sorted(df[floor_col].unique())
    group_ids = list(group_floors_s.index)
    rng.shuffle(group_ids)
    group_floors = group_floors_s.to_dict()

    def _train_safe(candidate, held_out_so_far):
        """True if pulling `candidate` out of train still leaves every
        floor it touches covered by >=1 OTHER group still in train."""
        current_train = [g for g in group_ids if g not in held_out_so_far and g != candidate]
        for floor in group_floors[candidate]:
            if not any(floor in group_floors[g] for g in current_train):
                return False
        return True

    test_groups, val_groups = set(), set()
    train_only_floors = []

    for floor in all_floors:
        candidates = [g for g in group_ids if floor in group_floors[g]
                      and g not in test_groups and g not in val_groups]
        safe = [g for g in candidates if _train_safe(g, test_groups | val_groups)]
        if safe:
            test_groups.add(safe[0])
        elif floor not in train_only_floors:
            train_only_floors.append(floor)
    for floor in all_floors:
        candidates = [g for g in group_ids if floor in group_floors[g]
                      and g not in test_groups and g not in val_groups]
        safe = [g for g in candidates if _train_safe(g, test_groups | val_groups)]
        if safe:
            val_groups.add(safe[0])
        elif floor not in train_only_floors:
            train_only_floors.append(floor)

    n_total = len(group_ids)
    target_test = max(0, round(test_frac * n_total) - len(test_groups))
    target_val = max(0, round(val_frac * n_total) - len(val_groups))
    remaining = [g for g in group_ids if g not in test_groups and g not in val_groups]
    rng.shuffle(remaining)
    # Fill toward target size, but still only with train-safe groups --
    # the size targets are soft, floor coverage in train is not.
    filled_test, held = 0, set(test_groups | val_groups)
    for g in remaining:
        if filled_test >= target_test:
            break
        if g not in held and _train_safe(g, held):
            test_groups.add(g)
            held.add(g)
            filled_test += 1
    filled_val = 0
    for g in remaining:
        if filled_val >= target_val:
            break
        if g not in held and _train_safe(g, held):
            val_groups.add(g)
            held.add(g)
            filled_val += 1

    train_groups = set(group_ids) - test_groups - val_groups

    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_groups)].reset_index(drop=True)

    for a, b, name_a, name_b in [(train_df, val_df, "train", "val"), (train_df, test_df, "train", "test"),
                                  (val_df, test_df, "val", "test")]:
        overlap = set(a[group_col]) & set(b[group_col])
        assert not overlap, f"Grouped split leaked {len(overlap)} group(s) across {name_a}/{name_b}: {sorted(overlap)[:5]}"

    missing_train = sorted(set(all_floors) - set(train_df[floor_col].unique()))
    assert not missing_train, (
        f"stratified_grouped_split_by_floor has a bug: floor(s) {missing_train} ended up "
        f"missing from TRAIN, which this function exists specifically to prevent."
    )
    missing_test = sorted(set(all_floors) - set(test_df[floor_col].unique()))
    missing_val = sorted(set(all_floors) - set(val_df[floor_col].unique()))
    print(f"Stratified split: {len(all_floors)} floors, {n_total} groups "
          f"-> train={len(train_groups)} val={len(val_groups)} test={len(test_groups)} groups. "
          f"Floors missing from test: {missing_test or 'none'}, from val: {missing_val or 'none'} "
          f"(train coverage guaranteed for all floors; too-scarce-for-test/val: {train_only_floors or 'none'}).")

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
