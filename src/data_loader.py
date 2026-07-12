"""
Step 1: Leakage-free data loading for UJIIndoorLoc.

Design decisions (documented, not hidden):
  - We use a single building (BUILDINGID == 2 by default): it has the most
    samples and the most distinct capture sessions (16 USERID x PHONEID
    combos), which is what a grouped split needs to be meaningful.
  - We discard BUILDINGID and predict only (LONGITUDE, LATITUDE) as the
    regression target. LONGITUDE/LATITUDE in UJIIndoorLoc are already
    projected coordinates in metres, not degrees, so MAE is directly
    interpretable in metres. FLOOR is NOT discarded from the pipeline,
    though: building 2's floors share almost the identical (x,y) footprint
    (median distance from a point to the nearest point on a DIFFERENT
    floor is ~0m), so a floor-blind model faces real, quantified ambiguity
    (a floor-oracle check cost ~2.5m of avoidable error). FLOOR is exposed
    to the rest of the pipeline via `train_floor`/`test_floor` so a floor
    classifier (floor_model.py) can supply predicted-floor features to the
    localization network -- see base_model.py's augmented input.
  - UJIIndoorLoc is burst-sampled: each (USERID, PHONEID) pair collected a
    contiguous run of fingerprints at nearly the same timestamps and
    physical waypoints. A random row-level split leaks near-duplicate
    fingerprints from the same walk into both train and test, which is
    why naive splits report near-0m error. We split by GROUP =
    (USERID, PHONEID) using GroupShuffleSplit, so an entire capture
    session goes entirely to train or entirely to test.
  - RSSI preprocessing: UJIIndoorLoc encodes "AP not detected" as 100.
    We remap 100 -> -105 (1 dBm below the observed detection floor of
    -104) so "no signal" is treated as "very weak signal" rather than an
    enormous positive outlier, then min-max scale to [0, 1] using
    statistics fit ONLY on the training split.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "UJIndoorLoc"
ZIP_URL = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
ZIP_PATH = DATA_DIR / "ujiindoorloc.zip"

NO_SIGNAL_RAW = 100
NO_SIGNAL_FILL = -105

BUILDING_ID = 2
TEST_SIZE = 0.2
RANDOM_STATE = 42


def ensure_downloaded() -> None:
    if (RAW_DIR / "trainingData.csv").exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        urlretrieve(ZIP_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)


def load_raw() -> pd.DataFrame:
    ensure_downloaded()
    return pd.read_csv(RAW_DIR / "trainingData.csv")


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
    X_test_raw100: np.ndarray  # test RSSI, untouched (100 = not detected), for drift sim
    test_timestamps: np.ndarray
    X_train_raw100: np.ndarray  # train RSSI, untouched, for meta-training drift episodes
    train_timestamps: np.ndarray
    train_floor: np.ndarray  # int floor label, 0-4 -- used as an auxiliary classifier target
    test_floor: np.ndarray


def _preprocess_rssi(raw: np.ndarray) -> np.ndarray:
    out = raw.astype(np.float32).copy()
    out[out == NO_SIGNAL_RAW] = NO_SIGNAL_FILL
    return out


def scale_rssi(raw100: np.ndarray, rssi_min: np.ndarray, rssi_max: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline (remap 100-sentinel, then min-max scale)
    applied to a RAW (100-sentinel) RSSI array, using stats fit on train.
    Reused by the drift simulator so drifted fingerprints go through the
    exact same pipeline the base model was trained on."""
    remapped = _preprocess_rssi(raw100)
    span = np.where(rssi_max - rssi_min > 0, rssi_max - rssi_min, 1.0)
    return ((remapped - rssi_min) / span).astype(np.float32)


def load_split(
    building_id: int = BUILDING_ID,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Split:
    df = load_raw()
    df = df[df["BUILDINGID"] == building_id].reset_index(drop=True)

    wap_cols = [c for c in df.columns if c.startswith("WAP")]
    rssi = _preprocess_rssi(df[wap_cols].to_numpy())
    targets = df[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=np.float32)

    # Group = one physical capture session. Same phone + same user walking
    # a route and logging fingerprints back-to-back.
    groups = (df["USERID"].astype(str) + "_" + df["PHONEID"].astype(str)).to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(rssi, targets, groups))

    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    assert train_groups.isdisjoint(test_groups), "Leakage: a capture session appears in both splits"

    # Fit RSSI scaling stats on TRAIN ONLY, then apply to both splits.
    rssi_train_raw = rssi[train_idx]
    rssi_min = rssi_train_raw.min(axis=0)
    rssi_max = rssi_train_raw.max(axis=0)

    X_train = scale_rssi(df[wap_cols].to_numpy()[train_idx], rssi_min, rssi_max)
    X_test = scale_rssi(df[wap_cols].to_numpy()[test_idx], rssi_min, rssi_max)
    y_train = targets[train_idx]
    y_test = targets[test_idx]

    return Split(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        wap_cols=wap_cols,
        rssi_min=rssi_min,
        rssi_max=rssi_max,
        train_groups=groups[train_idx],
        test_groups=groups[test_idx],
        X_test_raw100=df[wap_cols].to_numpy()[test_idx],
        test_timestamps=df["TIMESTAMP"].to_numpy()[test_idx],
        X_train_raw100=df[wap_cols].to_numpy()[train_idx],
        train_timestamps=df["TIMESTAMP"].to_numpy()[train_idx],
        train_floor=df["FLOOR"].to_numpy()[train_idx],
        test_floor=df["FLOOR"].to_numpy()[test_idx],
    )


if __name__ == "__main__":
    split = load_split()
    print(f"Train: {split.X_train.shape}, Test: {split.X_test.shape}")
    print(f"Train groups ({len(set(split.train_groups))}): {sorted(set(split.train_groups))}")
    print(f"Test groups  ({len(set(split.test_groups))}): {sorted(set(split.test_groups))}")
    print(f"Test fraction of rows: {len(split.X_test) / (len(split.X_train) + len(split.X_test)):.3f}")
    print(f"Target (x,y) train range: "
          f"x[{split.y_train[:,0].min():.1f},{split.y_train[:,0].max():.1f}] "
          f"y[{split.y_train[:,1].min():.1f},{split.y_train[:,1].max():.1f}]")
