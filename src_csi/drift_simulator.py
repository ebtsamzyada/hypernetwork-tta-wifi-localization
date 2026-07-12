"""
Step 3 (CSI variant): Continuous, accumulating drift simulator.

Key difference from the RSSI drift simulators (src/, src_sod/): those
had multiple SESSIONS, each a genuinely different deployment (different
user/phone), so drift reset at each session boundary. This dataset is a
SINGLE fixed Tx-Rx link -- there is no "different device" boundary to
reset at. The natural reading of "continuous deployment" here is: ONE
system, deployed once, monitoring the room over its whole operational
lifetime while drift (subcarrier-wise amplitude fading, e.g. from
furniture moving, people walking through, seasonal changes) accumulates
monotonically the entire time -- while it happens to be asked to
localize at different positions in the room as time passes.

Deployment order: the 64 held-out TEST reference points are ordered via
a greedy nearest-neighbor spatial tour (a plausible "the deployed system
periodically re-checks each part of the room" path -- there's no real
recorded visiting order across DIFFERENT points to use, unlike the
within-point packets, which keep their original temporal order).

Drift model: a persistent per-(antenna,subcarrier) random-walk additive
offset, applied to amplitude (the only feature currently in use), that
accumulates every timestep and never resets -- clipped to stay
non-negative (amplitude can't be negative) since nothing else bounds it.
`walk_sigma` is deliberately much smaller than the RSSI simulators used:
this timeline has ~19,000 steps (64 points x ~300 packets each) versus
RSSI's few-thousand-step timelines, and CSI amplitude's own scale
(std ~10.5, range ~0-43) is different from RSSI's dBm scale -- picked so
accumulated drift reaches a meaningful fraction of the amplitude's own
spread by the end of the timeline without blowing up early on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_loader import scale_features, Split

AMPLITUDE_FLOOR = 0.0
AMPLITUDE_CEIL = 60.0  # generous headroom above the observed max (~42.7)


@dataclass
class DriftedStream:
    X_raw: np.ndarray     # (T, 90) drifted RAW amplitude
    X_scaled: np.ndarray  # (T, 90) drifted amplitude through the model's preprocessing
    y: np.ndarray         # (T, 2) targets, same order
    order: np.ndarray     # indices into the original split.X_test / y_test
    point_ids: np.ndarray  # which reference point each step came from


def deployment_order(split: Split) -> np.ndarray:
    """Order test rows: points visited via a greedy nearest-neighbor
    spatial tour, packets within a point kept in their original
    (temporal) order."""
    unique_points = np.unique(split.point_test)
    point_coords = {}
    for p in unique_points:
        idx = np.where(split.point_test == p)[0][0]
        point_coords[p] = split.y_test[idx]
    coords_arr = np.array([point_coords[p] for p in unique_points])

    remaining = list(range(len(unique_points)))
    tour = [remaining.pop(0)]
    while remaining:
        last = coords_arr[tour[-1]]
        dists = np.linalg.norm(coords_arr[remaining] - last, axis=1)
        nxt = remaining.pop(int(np.argmin(dists)))
        tour.append(nxt)
    ordered_points = unique_points[tour]

    order_parts = [np.where(split.point_test == p)[0] for p in ordered_points]
    return np.concatenate(order_parts)


def simulate_drift(split: Split, walk_sigma: float = 0.4, seed: int = 0) -> DriftedStream:
    order = deployment_order(split)
    X_raw = split.X_test_raw[order].astype(np.float32).copy()
    y = split.y_test[order]
    point_ids = split.point_test[order]
    T, D = X_raw.shape

    rng = np.random.default_rng(seed)
    drift_offset = np.zeros(D, dtype=np.float32)

    for t in range(T):
        drift_offset += rng.normal(0.0, walk_sigma, size=D).astype(np.float32)
        X_raw[t] = np.clip(X_raw[t] + drift_offset, AMPLITUDE_FLOOR, AMPLITUDE_CEIL)

    X_scaled = scale_features(X_raw, split.feat_mean, split.feat_std)

    return DriftedStream(X_raw=X_raw, X_scaled=X_scaled, y=y, order=order, point_ids=point_ids)


def random_drift_snapshot(X_raw: np.ndarray, rng: np.random.Generator,
                           severity_range=(0.5, 8.0)) -> tuple:
    """Perturb a buffer with ONE fixed random per-feature offset sampled
    at random -- a snapshot of "how much accumulated drift exists right
    now" rather than a step-by-step walk within the buffer itself (the
    buffer is short relative to the full deployment timeline). Used to
    generate diverse meta-training episodes on TRAIN points only. No
    dropout concept here (unlike RSSI): CSI amplitude has no "not
    detected" state for a single always-on link. Returns
    (drifted_raw, severity)."""
    B, D = X_raw.shape
    severity = rng.uniform(*severity_range)
    offset = (rng.normal(0.0, 1.0, size=D) * severity).astype(np.float32)
    drifted = np.clip(X_raw + offset, AMPLITUDE_FLOOR, AMPLITUDE_CEIL)
    return drifted, severity


if __name__ == "__main__":
    from data_loader import load_split

    split = load_split()
    stream = simulate_drift(split)

    print(f"Deployment timeline length: {len(stream.y)} steps")
    orig_raw = split.X_test_raw[stream.order]
    diffs = np.abs(stream.X_raw - orig_raw)
    n_chunks = 5
    chunk_size = len(stream.y) // n_chunks
    for c in range(n_chunks):
        sl = slice(c * chunk_size, (c + 1) * chunk_size)
        print(f"chunk {c}: mean |amplitude drift| = {diffs[sl].mean():.3f}")
