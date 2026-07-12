"""
Step 1 (CSI variant): Leakage-free data loading for the qiang5love1314
CSI dataset (Lab Dataset area -- single Tx-Rx link, NLOS computer lab).
See ../CSI_PIVOT_PLAN.md for the full rationale and honest differences
from the RSSI datasets this project started with.

Design decisions (documented, not hidden):
  - This is a SINGLE Tx-Rx link (one transmitter, one receiver -- verified
    by rendering lab.pdf and looking at it), not multiple access points.
    There is no "AP heterogeneity/dropout" structure here the way there
    was for RSSI; the localization signal comes from fine-grained
    multipath structure across 3 antennas x 30 subcarrier groups.
  - Each of the 317 reference points has 1500 packets recorded. We treat
    each packet as one training sample (all 1500 sharing that point's
    label), since that's far more data than aggregating per-point, but
    this means the split MUST hold out entire POINTS for test, not
    packets -- 1500 packets at one point are a burst of near-identical
    repeated measurements, exactly the kind of leakage risk burst-sampled
    UJIIndoorLoc taught us to watch for.
  - Coordinates are filename-encoded (e.g. "coordinate715" -> grid [7,15])
    and converted to metres using the room's real layout, both VERIFIED
    against source material (not assumed) -- see CSI_PIVOT_PLAN.md.
  - Features: CSI AMPLITUDE only (|real + 1j*imag|) per (antenna,
    subcarrier) = 3*30 = 90 features per packet. Phase is available but
    left out of this first pass -- amplitude is the less ambiguous
    choice (phase needs calibration/unwrapping to be usable) and this
    project's own discipline has been to keep the first pass simple and
    get an honest baseline before adding complexity.
  - Standardized (zero mean, unit variance) using TRAIN-ONLY statistics,
    analogous to how RSSI was min-max scaled using train-only stats.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io as sio
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data_csi" / "CSI-dataset-for-indoor-localization-main" / "Lab Dataset"
)
REAL_SUBDIRS = ["coordinate 1-100", "coordinate 101-200", "coordinate 201-300", "coordinate 301-317"]
IMAG_SUBDIR = "imaginary_part"

# Verified against lab.pdf (rendered and visually inspected): 13.5m x 11m
# room, grid column/row counts match the decoded X/Y ranges exactly.
ROOM_WIDTH_M = 13.5
ROOM_HEIGHT_M = 11.0
X_GRID_MAX = 21
Y_GRID_MAX = 23

TEST_SIZE = 0.2
RANDOM_STATE = 42


def decode_coords(point_id: int) -> tuple:
    """Filename numeric ID -> (x_metres, y_metres). Last 2 digits = Y grid
    index, remaining leading digits = X grid index -- verified against the
    README's own example (coordinate715 -> grid [7, 15])."""
    s = str(point_id)
    y_grid = int(s[-2:])
    x_grid = int(s[:-2])
    x_m = (x_grid - 1) / (X_GRID_MAX - 1) * ROOM_WIDTH_M
    y_m = (y_grid - 1) / (Y_GRID_MAX - 1) * ROOM_HEIGHT_M
    return x_m, y_m


def find_all_points():
    """Returns list of (point_id, real_mat_path)."""
    points = []
    for d in REAL_SUBDIRS:
        for f in sorted((DATA_DIR / d).glob("coordinate*.mat")):
            m = re.search(r"coordinate(\d+)\.mat", f.name)
            points.append((int(m.group(1)), f))
    return points


def load_point_amplitude(point_id: int, real_path: Path) -> np.ndarray:
    """Returns (1500, 90) amplitude features -- one row per packet."""
    real = sio.loadmat(real_path)["myData"]  # (3, 30, 1500)
    imag_path = DATA_DIR / IMAG_SUBDIR / f"imaginary{point_id}.mat"
    imag = sio.loadmat(imag_path)["myData"]
    complex_csi = real + 1j * imag
    amplitude = np.abs(complex_csi)  # (3, 30, 1500)
    n_antennas, n_subcarriers, n_packets = amplitude.shape
    return amplitude.reshape(n_antennas * n_subcarriers, n_packets).T.astype(np.float32)


@dataclass
class Split:
    X_train: np.ndarray  # (N, 90) standardized amplitude features
    X_test: np.ndarray
    y_train: np.ndarray  # (N, 2) (x, y) in metres
    y_test: np.ndarray
    point_train: np.ndarray  # which reference point each row came from
    point_test: np.ndarray
    feat_mean: np.ndarray
    feat_std: np.ndarray
    num_features: int
    X_test_raw: np.ndarray  # (N, 90) RAW (unstandardized) amplitude -- for drift simulation
    X_train_raw: np.ndarray


def scale_features(X_raw: np.ndarray, feat_mean: np.ndarray, feat_std: np.ndarray) -> np.ndarray:
    """Standardize RAW amplitude using TRAIN-fit stats. Reused by the
    drift simulator so drifted amplitude goes through the exact same
    pipeline the base model was trained on (mirrors the RSSI pipelines'
    scale_rssi)."""
    return ((X_raw - feat_mean) / feat_std).astype(np.float32)


def load_split(test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE,
               packet_stride: int = 5) -> Split:
    """`packet_stride`: keep every Nth packet per point (default 5, so 1500
    -> 300 per point). The 1500 packets at one point are highly correlated
    near-duplicate readings of the same physical channel (diagnosed via a
    train/val naive-baseline comparison: a plain MLP trained on all 1500
    overfit badly -- 0.64m train error vs 5.14m validation error, an 8x
    gap, and lost to a trivial constant-prediction baseline on the held-out
    test points). Subsampling reduces redundant near-duplicates the model
    could otherwise memorize instead of learning genuine spatial structure,
    while still leaving ample training data (300 packets x ~250 points ~=
    75,000 rows)."""
    points = find_all_points()
    rng = np.random.default_rng(random_state)

    all_X, all_y, all_point_ids = [], [], []
    n_dropped_packets = 0
    for point_id, real_path in points:
        feats = load_point_amplitude(point_id, real_path)  # (1500, 90)
        # A small fraction of packets (mostly in the weakest-signal grid
        # cells, farthest from the Tx/Rx pair) have Inf amplitude values --
        # drop only those specific PACKETS (rows), not the whole point, so
        # we don't discard otherwise-good data from 34 affected points.
        bad_rows = ~np.isfinite(feats).all(axis=1)
        if bad_rows.any():
            n_dropped_packets += bad_rows.sum()
            feats = feats[~bad_rows]
        if packet_stride > 1 and len(feats) > packet_stride:
            keep = rng.choice(len(feats), size=len(feats) // packet_stride, replace=False)
            feats = feats[np.sort(keep)]
        x_m, y_m = decode_coords(point_id)
        all_X.append(feats)
        all_y.append(np.tile([x_m, y_m], (feats.shape[0], 1)))
        all_point_ids.append(np.full(feats.shape[0], point_id))

    if n_dropped_packets:
        print(f"Dropped {n_dropped_packets} corrupted (Inf-valued) packets out of "
              f"{len(points) * 1500} total.")

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.float32)
    point_ids = np.concatenate(all_point_ids, axis=0)

    # Grouped split by REFERENCE POINT (not packet) -- see module docstring.
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    tr_idx, te_idx = next(gss.split(X, y, point_ids))
    train_points = set(point_ids[tr_idx])
    test_points = set(point_ids[te_idx])
    assert train_points.isdisjoint(test_points), "Leakage: a reference point appears in both splits"

    X_train_raw, X_test_raw = X[tr_idx], X[te_idx]
    y_train, y_test = y[tr_idx], y[te_idx]

    feat_mean = X_train_raw.mean(axis=0)
    feat_std = X_train_raw.std(axis=0)
    feat_std = np.where(feat_std > 0, feat_std, 1.0)

    X_train = scale_features(X_train_raw, feat_mean, feat_std)
    X_test = scale_features(X_test_raw, feat_mean, feat_std)

    return Split(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        point_train=point_ids[tr_idx], point_test=point_ids[te_idx],
        feat_mean=feat_mean, feat_std=feat_std, num_features=X.shape[1],
        X_test_raw=X_test_raw.astype(np.float32), X_train_raw=X_train_raw.astype(np.float32),
    )


if __name__ == "__main__":
    split = load_split()
    print(f"Train: {split.X_train.shape}, Test: {split.X_test.shape}")
    print(f"Train points: {len(set(split.point_train))}, Test points: {len(set(split.point_test))}")
    print(f"Target (x,y) train range: "
          f"x[{split.y_train[:,0].min():.2f},{split.y_train[:,0].max():.2f}] "
          f"y[{split.y_train[:,1].min():.2f},{split.y_train[:,1].max():.2f}] metres")
