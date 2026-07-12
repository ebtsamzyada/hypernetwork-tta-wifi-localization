"""
Step 5 (CSI variant): The deployment simulation (headline result).

Runs the TEST split through the continuous drift timeline (Step 3) and
compares two localization strategies at every step:
  - STATIC FROZEN: the base MLP's original head, never touched.
  - HYPERNETWORK-ADAPTED: the hypernetwork looks at a rolling window of
    the most recent UNLABELED readings, compares it against a REFERENCE
    window captured at the start of the CURRENT reference-point dwell
    (reset whenever the deployment tour arrives at a new physical point
    -- see hypernetwork.py's docstring for why point-dwell, not a global
    anchor, is the right reference scope for this single-link setup),
    and predicts a replacement (delta_W, delta_b) for the frozen head.

Strictly causal: at step i, only readings up to i are ever used.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_split, scale_features
from drift_simulator import simulate_drift
from hypernetwork import HyperNetwork, compute_buffer_stats
from train_hypernetwork import load_frozen_base, euclidean_error, OUT_DIR

BUFFER_SIZE = 30
CADENCE = 15
GATE_THRESHOLD = None
# Calibrated on a HELD-OUT TRAIN-POINT deployment simulation (60 random
# train points, never the real test set). Unlike the RSSI pipelines, no
# instability was found at any clip level tested (0.05-0.6) -- improvement
# saturates around 0.2-0.3 at a small, consistent +0.022m (~0.5% relative)
# and stays flat, never regressing. 0.3 chosen as a safety-margin default.
MAX_DELTA_FRAC = 0.3
N_DRIFT_SEEDS = 15


def load_hypernetwork(out_dir=OUT_DIR):
    ckpt = torch.load(f"{out_dir}/hypernetwork.pt", weights_only=False)
    hypernet = HyperNetwork(num_features=ckpt["num_features"], head_in=ckpt["head_in"],
                             head_out=ckpt["head_out"], hidden=ckpt["hidden"])
    hypernet.load_state_dict(ckpt["state_dict"])
    hypernet.eval()
    return hypernet


def run_deployment(split, stream, base_model, base_head_w, base_head_b, target_mean, hypernet,
                    buffer_size=BUFFER_SIZE, cadence=CADENCE, gate_threshold=None, max_delta_frac=None):
    T = len(stream.y)
    X_t = torch.tensor(stream.X_scaled, dtype=torch.float32)
    with torch.no_grad():
        features = base_model.trunk(X_t)

    frozen_pred = (F.linear(features, base_head_w, base_head_b)).numpy() + target_mean

    base_combined_norm = float(np.sqrt(base_head_w.norm().item() ** 2 + base_head_b.norm().item() ** 2))
    adapted_pred = np.empty_like(frozen_pred)
    cur_head_w, cur_head_b = base_head_w, base_head_b
    n_firings = 0

    dwell_start = 0
    ref_raw = ref_scaled = None

    for i in range(T):
        if i == 0 or stream.point_ids[i] != stream.point_ids[i - 1]:
            dwell_start = i
            ref_raw = ref_scaled = None
            cur_head_w, cur_head_b = base_head_w, base_head_b

        steps_into_dwell = i - dwell_start
        if steps_into_dwell == buffer_size:
            ref_raw = stream.X_raw[dwell_start:i]
            ref_scaled = scale_features(ref_raw, split.feat_mean, split.feat_std)

        if ref_raw is not None and steps_into_dwell >= buffer_size \
                and (steps_into_dwell - buffer_size) % cadence == 0:
            window = slice(i - buffer_size, i)
            cur_scaled = stream.X_scaled[window]
            stats = compute_buffer_stats(cur_scaled, ref_scaled)
            stats_t = torch.tensor(stats, dtype=torch.float32)
            with torch.no_grad():
                dW, db = hypernet(stats_t)

            if max_delta_frac is not None:
                combined_norm = float(np.sqrt(dW.norm().item() ** 2 + db.norm().item() ** 2))
                max_norm = max_delta_frac * base_combined_norm
                if combined_norm > max_norm:
                    scale = max_norm / combined_norm
                    dW, db = dW * scale, db * scale

            cur_head_w = base_head_w + dW
            cur_head_b = base_head_b + db
            n_firings += 1

        with torch.no_grad():
            adapted_pred[i] = (F.linear(features[i], cur_head_w, cur_head_b)).numpy() + target_mean

    print(f"Hypernetwork fired {n_firings} times over {T} deployment steps.")
    return frozen_pred, adapted_pred


def multi_seed_evaluation(split, base_model, base_head_w, base_head_b, target_mean, hypernet,
                           n_seeds=N_DRIFT_SEEDS):
    frozen_means, adapted_means = [], []
    for seed in range(n_seeds):
        stream = simulate_drift(split, seed=seed)
        frozen_pred, adapted_pred = run_deployment(split, stream, base_model, base_head_w, base_head_b,
                                                     target_mean, hypernet, gate_threshold=GATE_THRESHOLD,
                                                     max_delta_frac=MAX_DELTA_FRAC)
        frozen_means.append(euclidean_error(frozen_pred, stream.y).mean())
        adapted_means.append(euclidean_error(adapted_pred, stream.y).mean())
    frozen_means, adapted_means = np.array(frozen_means), np.array(adapted_means)
    improvements = frozen_means - adapted_means

    print(f"\n=== Robust headline result: {n_seeds} independent random drift realizations ===")
    print(f"Frozen : {frozen_means.mean():.3f}m +/- {frozen_means.std():.3f}m")
    print(f"Adapted: {adapted_means.mean():.3f}m +/- {adapted_means.std():.3f}m")
    print(f"Improvement: {improvements.mean():+.3f}m +/- {improvements.std():.3f}m "
          f"({improvements.mean() / frozen_means.mean() * 100:.1f}% relative)")
    print(f"Seeds where adapted beat frozen: {(improvements > 0).sum()}/{n_seeds}")


def main():
    split = load_split()
    stream = simulate_drift(split)
    base_model, base_head_w, base_head_b, target_mean = load_frozen_base()
    hypernet = load_hypernetwork()

    frozen_pred, adapted_pred = run_deployment(split, stream, base_model, base_head_w, base_head_b,
                                                target_mean, hypernet, gate_threshold=GATE_THRESHOLD,
                                                max_delta_frac=MAX_DELTA_FRAC)

    frozen_err = euclidean_error(frozen_pred, stream.y)
    adapted_err = euclidean_error(adapted_pred, stream.y)

    print(f"\n=== Step 5 result for ONE drift realization (seed=0, for the plot below) ===")
    print(f"Overall mean error -- frozen: {frozen_err.mean():.3f}m   adapted: {adapted_err.mean():.3f}m")

    window = 200
    kernel = np.ones(window) / window
    frozen_smooth = np.convolve(frozen_err, kernel, mode="valid")
    adapted_smooth = np.convolve(adapted_err, kernel, mode="valid")
    steps = np.arange(len(frozen_smooth))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, frozen_smooth, label="Static frozen base network", color="tab:red")
    plt.plot(steps, adapted_smooth, label="Hypernetwork-adapted network", color="tab:blue")
    plt.xlabel(f"Deployment step (t), {window}-step rolling mean")
    plt.ylabel("Localization error (m)")
    plt.title("CSI: localization error vs. deployment time under continuous drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/step5_deployment_result.png", dpi=150)
    print(f"\nSaved plot to {OUT_DIR}/step5_deployment_result.png")

    multi_seed_evaluation(split, base_model, base_head_w, base_head_b, target_mean, hypernet)


if __name__ == "__main__":
    main()
