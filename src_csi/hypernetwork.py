"""
Hypernetwork: reads rolling-buffer statistics of unlabeled CSI amplitude
readings and predicts a weight DELTA for the frozen base network's final
linear layer, to compensate for drift.

Differences from the RSSI hypernetwork (src/, src_sod/):
  - No detection-rate feature: CSI amplitude is a continuous value that's
    always present (no "AP not detected" sentinel/concept for a single
    always-on link), so the stat vector is just mean and variance
    DIFFERENCES between the current buffer and a reference buffer, per
    feature (2 x 90 = 180-dim), not 3x with a detection-rate term.
  - The REFERENCE buffer resets at each new REFERENCE-POINT DWELL, not at
    session boundaries -- there are no different "sessions" here (single
    fixed Tx-Rx link), but the deployment simulation moves through
    different physical positions over time, and comparing against a
    different location would reintroduce the exact location/drift
    confound the RSSI reference-buffer design was built to eliminate
    (see hypernetwork.py's docstring in the RSSI pipelines for the full
    diagnostic history). A per-point-dwell reference is also the
    physically honest choice for an unlabeled streaming deployment: a
    system can only compare against readings it has ACTUALLY collected
    from roughly the same vicinity, not a fixed global anchor.

Output: (delta_weight, delta_bias) for the head, added on top of the
FROZEN base weights. Output layers are zero-initialized so an
untrained/unconfident hypernetwork starts as a no-op.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_buffer_stats(cur: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """cur, ref: (B, D) amplitude arrays. Returns (2*D,) stat vector:
    mean difference and variance difference, current vs. reference."""
    mean_diff = cur.mean(axis=0) - ref.mean(axis=0)
    var_diff = cur.var(axis=0) - ref.var(axis=0)
    return np.concatenate([mean_diff, var_diff]).astype(np.float32)


class HyperNetwork(nn.Module):
    def __init__(self, num_features: int, head_in: int, head_out: int = 2, hidden: int = 64):
        super().__init__()
        stat_dim = 2 * num_features
        self.head_in = head_in
        self.head_out = head_out
        self.trunk = nn.Sequential(
            nn.Linear(stat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out_weight = nn.Linear(hidden, head_out * head_in)
        self.out_bias = nn.Linear(hidden, head_out)

        nn.init.zeros_(self.out_weight.weight)
        nn.init.zeros_(self.out_weight.bias)
        nn.init.zeros_(self.out_bias.weight)
        nn.init.zeros_(self.out_bias.bias)

    def forward(self, stats: torch.Tensor):
        h = self.trunk(stats)
        dW = self.out_weight(h).view(*h.shape[:-1], self.head_out, self.head_in)
        db = self.out_bias(h)
        return dW, db
