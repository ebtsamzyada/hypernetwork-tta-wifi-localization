"""
Hypernetwork: reads rolling-buffer statistics of unlabeled RSSI fingerprints
and predicts a weight DELTA for the frozen base network's final linear
layer (the head), to compensate for environmental drift.

Design history (kept because it explains a non-obvious choice): the first
version fed the hypernetwork ABSOLUTE per-AP statistics (mean/var/detect
-rate) of the current buffer. Leave-one-session-out cross-validation
showed this gives ~0 generalizable benefit (mean improvement across folds
was statistically indistinguishable from noise). Diagnosis: only ~3-4% of
520 APs are visible at any one physical location, different locations see
almost disjoint AP subsets, and the "correct" head correction for a buffer
turned out to be dominated by WHICH LOCATION it's from (optimal bias
varied by ~15m across sessions) rather than by drift severity (correlation
~0.3). A hypernetwork can't recover a location-specific baseline it has
never seen.

Fix: compare the CURRENT buffer against a REFERENCE buffer captured
earlier in the SAME deployment session (in Step 5, the first rolling
window collected right after deployment starts, before drift has had time
to accumulate). Both buffers are from the same physical site, so their
DIFFERENCE cancels out location-specific AP visibility/signal-strength
entirely and isolates the drift itself. A quick oracle check (ridge
regression, leave-one-session-out) confirmed this: predictive R^2 for the
drift-induced prediction shift went from ~0.02 (absolute stats) to ~0.25
(reference-relative stats), with the dominant signal being how much the
AP DETECTION pattern has changed relative to that session's own baseline.

Input statistics per AP dimension, comparing current vs. reference buffer:
  - detection-rate difference (current - reference): the strongest signal,
    captures AP dropout relative to this site's own normal condition.
  - mean scaled-RSSI difference (current - reference): captures
    attenuation drift relative to this site's own normal levels.
  - current buffer's absolute detection rate: conveys how sparse/reliable
    the current reading is overall (not comparative, but still useful).

Output: (delta_weight, delta_bias) for the head, added on top of the
FROZEN base weights. Output layers are zero-initialized so an
untrained/unconfident hypernetwork starts as a no-op.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from data_loader import NO_SIGNAL_RAW


def compute_buffer_stats(cur_raw100: np.ndarray, cur_scaled: np.ndarray,
                          ref_raw100: np.ndarray, ref_scaled: np.ndarray) -> np.ndarray:
    """All arrays (B, D). `ref_*` is an earlier, less-drifted buffer from
    the SAME deployment session/site. Returns (3*D,) stat vector."""
    cur_detect_rate = (cur_raw100 != NO_SIGNAL_RAW).mean(axis=0)
    ref_detect_rate = (ref_raw100 != NO_SIGNAL_RAW).mean(axis=0)
    detect_rate_diff = cur_detect_rate - ref_detect_rate

    mean_diff = cur_scaled.mean(axis=0) - ref_scaled.mean(axis=0)

    return np.concatenate([detect_rate_diff, mean_diff, cur_detect_rate]).astype(np.float32)


class HyperNetwork(nn.Module):
    def __init__(self, num_aps: int, head_in: int, head_out: int = 2, hidden: int = 128):
        super().__init__()
        stat_dim = 3 * num_aps
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

        # Zero-init the output heads: at t=0, delta = 0, adapted == frozen base.
        nn.init.zeros_(self.out_weight.weight)
        nn.init.zeros_(self.out_weight.bias)
        nn.init.zeros_(self.out_bias.weight)
        nn.init.zeros_(self.out_bias.bias)

    def forward(self, stats: torch.Tensor):
        """stats: (stat_dim,) or (batch, stat_dim). Returns (delta_W, delta_b)."""
        h = self.trunk(stats)
        dW = self.out_weight(h).view(*h.shape[:-1], self.head_out, self.head_in)
        db = self.out_bias(h)
        return dW, db
