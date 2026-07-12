"""
Step 1 (SOD-HCXY variant): Leakage-free data loading for the SOD-HCXY
building from SODIndoorLoc (github.com/bijingxue/SODIndoorLoc).

Why this dataset, in addition to UJIIndoorLoc: UJIIndoorLoc has only
~3.8% of its 520 APs visible per fingerprint, which we found dominates
the hypernetwork's ability to separate "which location" from "how much
drift" (buffer statistics are too sparse/location-specific to
generalize). SOD-HCXY has 56 APs with ~22% visibility per fingerprint
(~6x denser) -- this tests whether that sparsity was really the limiting
factor. It is also single-floor (FloorID is constant), so none of
UJIIndoorLoc's floor-conflation issue applies here.

Design decisions:
  - Use the dataset's OFFICIAL train/test split (Training_HCXY_AP_30.csv /
    Testing_HCXY_AP.csv) rather than deriving our own: verified directly
    that the two files share ZERO (ECoord, NCoord) reference points, i.e.
    it's already a clean, point-disjoint held-out test set.
  - UserID + PhoneID is still a valid group key for a validation carve-out
    within train (each user surveyed a disjoint subset of points, 30
    repeated samples each), preventing the same near-duplicate-repeat
    leakage UJIIndoorLoc had.
  - RSSI: same encoding as UJIIndoorLoc (100 = not detected), but the
    actual detected range here is [-97, -9] dBm, not UJI's [-104, 0] --
    use the dataset's own observed floor/ceiling rather than reusing UJI's
    constants.
  - No real continuous TIMESTAMP exists (SampleTimes is just a 1-30 repeat
    counter at a fixed point, not wall-clock time). A synthetic ordering
    is constructed per user session: points are ordered via a greedy
    nearest-neighbor tour (a plausible walking path through that user's
    assigned points), then the 30 repeats at each point are played in
    sequence -- giving drift_simulator.py's chronological ordering
    something coherent to sort by.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data_sod" / "SODIndoorLoc-main" / "HCXY"
TRAIN_CSV = DATA_DIR / "Training_HCXY_AP_30.csv"
TEST_CSV = DATA_DIR / "Testing_HCXY_AP.csv"

NO_SIGNAL_RAW = 100
NO_SIGNAL_FILL = -98  # 1 dBm below the observed detection floor of -97

RSSI_FLOOR = -97.0
RSSI_CEIL = 0.0


def _preprocess_rssi(raw: np.ndarray) -> np.ndarray:
    out = raw.astype(np.float32).copy()
    out[out == NO_SIGNAL_RAW] = NO_SIGNAL_FILL
    return out


def scale_rssi(raw100: np.ndarray, rssi_min: np.ndarray, rssi_max: np.ndarray) -> np.ndarray:
    remapped = _preprocess_rssi(raw100)
    span = np.where(rssi_max - rssi_min > 0, rssi_max - rssi_min, 1.0)
    return ((remapped - rssi_min) / span).astype(np.float32)


@dataclass
class Split:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    wap_cols: list
    rssi_min: np.ndarray
    rssi_max: np.ndarray
    train_groups: np.ndarray
    test_groups: np.ndarray
    X_test_raw100: np.ndarray
    test_timestamps: np.ndarray
    X_train_raw100: np.ndarray
    train_timestamps: np.ndarray
    train_floor: np.ndarray  # constant (single floor) -- kept for interface parity
    test_floor: np.ndarray


def _synthetic_timestamps(df: pd.DataFrame) -> np.ndarray:
    """Per-(UserID,PhoneID)-group, order that user's unique points via a
    greedy nearest-neighbor tour, then play each point's repeats (ordered
    by SampleTimes) in sequence. Returns a per-row float used only for
    sorting within a group -- absolute values / cross-group comparability
    don't matter (drift_simulator orders sessions by their own min value)."""
    ts = np.zeros(len(df), dtype=np.float64)
    for (uid, pid), gdf in df.groupby(["UserID", "PhoneID"]):
        pts = gdf[["ECoord", "NCoord"]].drop_duplicates().to_numpy()
        remaining = list(range(len(pts)))
        order = [remaining.pop(0)]
        while remaining:
            last = pts[order[-1]]
            dists = np.linalg.norm(pts[remaining] - last, axis=1)
            nxt = remaining.pop(int(np.argmin(dists)))
            order.append(nxt)
        point_rank = {tuple(pts[o]): rank for rank, o in enumerate(order)}
        rank_col = gdf.apply(lambda r: point_rank[(r["ECoord"], r["NCoord"])], axis=1)
        group_ts = rank_col.to_numpy() * 100.0 + gdf["SampleTimes"].to_numpy()
        ts[gdf.index.to_numpy()] = group_ts
    return ts


def load_raw():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    return train_df, test_df


def load_split() -> Split:
    train_df, test_df = load_raw()
    mac_cols = [c for c in train_df.columns if c.startswith("MAC")]
    assert mac_cols == [c for c in test_df.columns if c.startswith("MAC")]

    train_groups = (train_df["UserID"].astype(str) + "_" + train_df["PhoneID"].astype(str)).to_numpy()
    test_groups = (test_df["UserID"].astype(str) + "_" + test_df["PhoneID"].astype(str)).to_numpy()

    # Point-level leakage check: the 30 repeats of a point must never split
    # across train/test (verified already disjoint by construction, assert
    # to catch any future data changes).
    train_pts = set(map(tuple, train_df[["ECoord", "NCoord"]].drop_duplicates().to_numpy()))
    test_pts = set(map(tuple, test_df[["ECoord", "NCoord"]].drop_duplicates().to_numpy()))
    assert train_pts.isdisjoint(test_pts), "Leakage: a reference point appears in both splits"

    rssi_train_raw100 = train_df[mac_cols].to_numpy()
    rssi_test_raw100 = test_df[mac_cols].to_numpy()
    rssi_train_remapped = _preprocess_rssi(rssi_train_raw100)
    rssi_min = rssi_train_remapped.min(axis=0)
    rssi_max = rssi_train_remapped.max(axis=0)

    X_train = scale_rssi(rssi_train_raw100, rssi_min, rssi_max)
    X_test = scale_rssi(rssi_test_raw100, rssi_min, rssi_max)
    y_train = train_df[["ECoord", "NCoord"]].to_numpy(dtype=np.float32)
    y_test = test_df[["ECoord", "NCoord"]].to_numpy(dtype=np.float32)

    return Split(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        wap_cols=mac_cols,
        rssi_min=rssi_min,
        rssi_max=rssi_max,
        train_groups=train_groups,
        test_groups=test_groups,
        X_test_raw100=rssi_test_raw100,
        test_timestamps=_synthetic_timestamps(test_df),
        X_train_raw100=rssi_train_raw100,
        train_timestamps=_synthetic_timestamps(train_df),
        train_floor=train_df["FloorID"].to_numpy(),
        test_floor=test_df["FloorID"].to_numpy(),
    )


if __name__ == "__main__":
    split = load_split()
    print(f"Train: {split.X_train.shape}, Test: {split.X_test.shape}")
    print(f"Train groups ({len(set(split.train_groups))}): {sorted(set(split.train_groups))}")
    print(f"Test groups  ({len(set(split.test_groups))}): {sorted(set(split.test_groups))}")
    print(f"Mean AP visibility fraction (train): {(split.X_train_raw100 != NO_SIGNAL_RAW).mean():.4f}")
    print(f"Target (x,y) train range: "
          f"x[{split.y_train[:,0].min():.1f},{split.y_train[:,0].max():.1f}] "
          f"y[{split.y_train[:,1].min():.1f},{split.y_train[:,1].max():.1f}]")
