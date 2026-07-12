"""
Step 3: Continuous, accumulating drift simulator for the test stream.

Two coupled degradation processes, both driven by a deployment "progress"
fraction so drift is a continuous timeline, not a static severity dial
applied uniformly to every sample:

  1. Random-walk RSSI attenuation drift. Each AP dimension carries a
     persistent offset that takes a Gaussian random-walk step every
     timestep (`walk_sigma` dBm/step) and accumulates. This models slow
     physical changes in the environment (furniture moved, humidity,
     new obstructions) that shift measured signal strength further and
     further from what the base network was trained on. Only applied to
     APs the ground truth says are actually visible at that fingerprint.

  2. Increasing random AP dropout. A currently-visible AP is flipped to
     "not detected" with probability dropout_prob(t), which ramps
     linearly from `dropout_start` to `dropout_end` over the timeline.
     This models growing hardware/interference failures post-deployment.

(SOD-HCXY variant.) There's no real continuous TIMESTAMP in this dataset
(SampleTimes is just a 1-30 repeat counter at a fixed point) -- data_loader
synthesizes a per-session ordering (nearest-neighbor tour through each
user's assigned points). Each (UserID,PhoneID) session is still treated as
its OWN independent deployment: sessions are concatenated as CONTIGUOUS
blocks (never interleaved), and drift (accumulated random-walk offset +
dropout schedule progress) resets at the start of every session block,
mirroring the UJIIndoorLoc variant's session-boundary handling.

RSSI_FLOOR/CEIL use this dataset's own observed detection range
([-97, -9] dBm), not UJIIndoorLoc's [-104, 0].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_loader import NO_SIGNAL_RAW, Split, scale_rssi

RSSI_FLOOR = -97.0  # weakest real detected RSSI in this dataset
RSSI_CEIL = 0.0


@dataclass
class DriftedStream:
    X_raw100: np.ndarray       # (T, D) drifted RSSI, 100 = not detected
    X_scaled: np.ndarray       # (T, D) drifted RSSI through the model's preprocessing
    y: np.ndarray              # (T, 2) targets, same order
    order: np.ndarray          # indices into the original split.X_test / y_test
    dropout_prob: np.ndarray   # (T,) scheduled dropout probability at each step
    frac_visible: np.ndarray   # (T,) fraction of APs actually visible after drift
    groups: np.ndarray         # (T,) which capture session/site each step belongs to


def chronological_order(split: Split) -> np.ndarray:
    """Group rows by SESSION first (each session = one independent
    deployment at one physical site), sessions ordered by their own
    earliest timestamp, rows within a session by their own timestamp.
    Keeps each site's deployment contiguous instead of interleaving
    concurrently-captured sessions -- see module docstring."""
    groups = split.test_groups
    unique_groups = sorted(set(groups), key=lambda g: split.test_timestamps[groups == g].min())
    order_parts = []
    for g in unique_groups:
        idx = np.where(groups == g)[0]
        order_parts.append(idx[np.argsort(split.test_timestamps[idx], kind="stable")])
    return np.concatenate(order_parts)


def simulate_drift(
    split: Split,
    walk_sigma: float = 0.6,
    dropout_start: float = 0.0,
    dropout_end: float = 0.5,
    seed: int = 0,
) -> DriftedStream:
    order = chronological_order(split)
    raw100 = split.X_test_raw100[order].astype(np.float32).copy()
    y = split.y_test[order]
    groups_ordered = split.test_groups[order]
    T, D = raw100.shape

    # Per-step "local progress fraction" within its OWN session block, so
    # the dropout ramp and drift accumulation restart at every new site.
    local_progress = np.empty(T, dtype=np.float32)
    block_start = 0
    for t in range(1, T + 1):
        if t == T or groups_ordered[t] != groups_ordered[block_start]:
            block_len = t - block_start
            denom = max(block_len - 1, 1)
            local_progress[block_start:t] = np.arange(block_len) / denom
            block_start = t
    dropout_prob = dropout_start + (dropout_end - dropout_start) * local_progress

    rng = np.random.default_rng(seed)
    drift_offset = np.zeros(D, dtype=np.float32)
    frac_visible = np.empty(T, dtype=np.float32)

    for t in range(T):
        if t == 0 or groups_ordered[t] != groups_ordered[t - 1]:
            drift_offset[:] = 0.0  # new session/site: deployment starts fresh

        drift_offset += rng.normal(0.0, walk_sigma, size=D).astype(np.float32)

        row = raw100[t]
        visible = row != NO_SIGNAL_RAW

        # 1) accumulated random-walk attenuation drift on visible APs
        row[visible] = np.clip(row[visible] + drift_offset[visible], RSSI_FLOOR, RSSI_CEIL)

        # 2) increasing random dropout of currently-visible APs
        flip = visible & (rng.random(D) < dropout_prob[t])
        row[flip] = NO_SIGNAL_RAW

        frac_visible[t] = (row != NO_SIGNAL_RAW).mean()
        raw100[t] = row

    X_scaled = scale_rssi(raw100, split.rssi_min, split.rssi_max)

    return DriftedStream(
        X_raw100=raw100,
        X_scaled=X_scaled,
        y=y,
        order=order,
        dropout_prob=dropout_prob,
        frac_visible=frac_visible,
        groups=groups_ordered,
    )


def random_drift_snapshot(
    raw100_buf: np.ndarray,
    rng: np.random.Generator,
    walk_sigma_range=(0.5, 20.0),
    dropout_range=(0.0, 0.6),
):
    """Perturb a buffer with ONE fixed (offset, dropout_prob) pair sampled
    at random, i.e. a snapshot of "where in a possible deployment drift
    timeline are we right now" rather than a ramp within the buffer itself
    (the buffer is short relative to the full deployment timeline, so
    treating drift as roughly constant within it is a reasonable
    simplification). Used to generate diverse meta-training episodes on
    TRAIN data only. Returns (drifted_raw100, severity, dropout_prob)."""
    B, D = raw100_buf.shape
    buf = raw100_buf.astype(np.float32).copy()
    severity = rng.uniform(*walk_sigma_range)
    dropout_prob = rng.uniform(*dropout_range)

    offset = (rng.normal(0.0, 1.0, size=D) * severity).astype(np.float32)
    visible = buf != NO_SIGNAL_RAW
    offset_b = np.broadcast_to(offset, buf.shape)
    buf[visible] = np.clip(buf[visible] + offset_b[visible], RSSI_FLOOR, RSSI_CEIL)

    flip = visible & (rng.random(buf.shape) < dropout_prob)
    buf[flip] = NO_SIGNAL_RAW
    return buf, severity, dropout_prob


if __name__ == "__main__":
    from data_loader import load_split

    split = load_split()
    stream = simulate_drift(split)

    print(f"Deployment timeline length: {len(stream.y)} steps")
    print(f"Session blocks in order: {list(dict.fromkeys(stream.groups))}")
    print(f"Dropout probability: start={stream.dropout_prob[0]:.2f} -> end={stream.dropout_prob[-1]:.2f}")
    print(f"Mean fraction of APs visible: t=0 -> {stream.frac_visible[0]:.3f}, "
          f"t=mid -> {stream.frac_visible[len(stream.y)//2]:.3f}, "
          f"t=end -> {stream.frac_visible[-1]:.3f}")

    # Sanity check: drift should accumulate, i.e. later fingerprints should
    # differ from their un-drifted originals more than earlier ones.
    orig_raw100 = split.X_test_raw100[stream.order]
    both_visible = (orig_raw100 != NO_SIGNAL_RAW) & (stream.X_raw100 != NO_SIGNAL_RAW)
    diffs = np.abs(stream.X_raw100 - orig_raw100)
    n_chunks = 5
    chunk_size = len(stream.y) // n_chunks
    for c in range(n_chunks):
        sl = slice(c * chunk_size, (c + 1) * chunk_size)
        mask = both_visible[sl]
        mean_abs_drift = diffs[sl][mask].mean() if mask.any() else float("nan")
        print(f"chunk {c}: mean |RSSI drift| on still-visible APs = {mean_abs_drift:.2f} dBm, "
              f"mean frac visible = {stream.frac_visible[sl].mean():.3f}")
