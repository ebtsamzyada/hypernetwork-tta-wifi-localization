"""
Step 5: The deployment simulation (headline result).

Runs the TEST split through the continuous drift timeline (Step 3) and
compares two localization strategies at every step:
  - STATIC FROZEN: the base MLP's original head, never touched.
  - HYPERNETWORK-ADAPTED: on a fixed cadence, the hypernetwork looks at a
    rolling window of the most recent UNLABELED fingerprints, compares it
    against a REFERENCE window captured once at deployment start (before
    drift has had time to accumulate -- same design used in meta-training,
    see hypernetwork.py's docstring for why this specific comparison is
    what makes the correction generalize to unseen sites), and predicts a
    replacement (delta_W, delta_b) for the frozen head. That adapted head
    is used for every prediction until the next firing.

This is a strictly causal, online simulation: at deployment step i, only
fingerprints up to i are ever used (no peeking at future or at labels --
labels are only used afterward, to SCORE the two strategies).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_split, scale_rssi
from drift_simulator import simulate_drift
from hypernetwork import HyperNetwork, compute_buffer_stats
from train_hypernetwork import load_frozen_base, euclidean_error, OUT_DIR

BUFFER_SIZE = 50  # matches train_hypernetwork.py -- see its comment
CADENCE = 15  # test sessions are short (40-190 rows); a smaller cadence
# than the UJI variant's 50 gives short sessions a chance to fire more
# than once
# Calibrated on 4 HELD-OUT TRAIN-SESSION-PAIR deployment simulations
# (never the real test set), with buffer_size=50. Weighted mean improvement
# keeps climbing as the clip loosens (up to fully unclipped), but so does
# worst-case risk (one pair's regression grows from -0.08m at 0.4 to -2.52m
# unclipped) -- chose 0.5 for a good balance: captures most of the
# aggregate benefit (+1.61m weighted vs. unclipped's +2.97m) while keeping
# the worst case mild (-0.44m, not a real failure mode).
GATE_THRESHOLD = None
MAX_DELTA_FRAC = 0.5


def load_hypernetwork(out_dir=OUT_DIR):
    ckpt = torch.load(f"{out_dir}/hypernetwork.pt", weights_only=False)
    hypernet = HyperNetwork(num_aps=ckpt["num_aps"], head_in=ckpt["head_in"],
                             head_out=ckpt["head_out"], hidden=ckpt["hidden"])
    hypernet.load_state_dict(ckpt["state_dict"])
    hypernet.eval()
    return hypernet


def run_deployment(split, stream, base_model, base_head_w, base_head_b, target_mean, hypernet,
                    buffer_size=BUFFER_SIZE, cadence=CADENCE, gate_threshold=None, max_delta_frac=None):
    """Causal, online simulation. IMPORTANT: the concatenated timeline
    crosses session boundaries, and each session is a DIFFERENT physical
    site (confirmed: sessions see almost disjoint AP subsets). The
    reference buffer is only meaningful when it comes from the SAME site
    as the current window, so it must be re-captured at the start of
    every session, not just once at the start of the whole timeline --
    reusing one global reference across sessions was found to cause a
    catastrophic regression (verified on a held-out validation deployment
    run) because it silently compared unrelated physical locations."""
    T = len(stream.y)
    X_t = torch.tensor(stream.X_scaled, dtype=torch.float32)
    with torch.no_grad():
        features = base_model.trunk(X_t)  # (T, head_in), frozen trunk, computed once

    frozen_pred = (F.linear(features, base_head_w, base_head_b)).numpy() + target_mean

    num_aps = len(split.wap_cols)
    base_combined_norm = float(np.sqrt(base_head_w.norm().item() ** 2 + base_head_b.norm().item() ** 2))
    adapted_pred = np.empty_like(frozen_pred)
    cur_head_w, cur_head_b = base_head_w, base_head_b  # no correction until first firing
    n_firings = 0
    gates = []

    session_start = 0
    ref_raw100 = ref_scaled = None

    for i in range(T):
        if i == 0 or stream.groups[i] != stream.groups[i - 1]:
            # New session/site starts here -- recalibrate reference and
            # firing schedule relative to THIS session's own start, and
            # fall back to the frozen head until enough of a reference
            # window has been collected for the new site.
            session_start = i
            ref_raw100 = ref_scaled = None
            cur_head_w, cur_head_b = base_head_w, base_head_b

        steps_into_session = i - session_start
        if steps_into_session == buffer_size:
            ref_raw100 = stream.X_raw100[session_start:i]
            ref_scaled = scale_rssi(ref_raw100, split.rssi_min, split.rssi_max)

        if ref_raw100 is not None and steps_into_session >= buffer_size \
                and (steps_into_session - buffer_size) % cadence == 0:
            window = slice(i - buffer_size, i)
            cur_raw100 = stream.X_raw100[window]
            cur_scaled = stream.X_scaled[window]
            stats = compute_buffer_stats(cur_raw100, cur_scaled, ref_raw100, ref_scaled)
            stats_t = torch.tensor(stats, dtype=torch.float32)
            with torch.no_grad():
                dW, db = hypernet(stats_t)

            if max_delta_frac is not None:
                # Hard-bound the correction's magnitude relative to the
                # base head's OWN weight norm. Found necessary because a
                # loss-level regularization penalty couldn't reliably keep
                # corrections small (measured dW norms of ~5-9 vs. the base
                # head's own norm of 4.69 -- i.e. the "correction" was
                # comparable to or larger than the original learned
                # mapping, which is what caused rare catastrophic
                # per-session regressions when it pointed the wrong way).
                combined_norm = float(np.sqrt(dW.norm().item() ** 2 + db.norm().item() ** 2))
                max_norm = max_delta_frac * base_combined_norm
                if combined_norm > max_norm:
                    scale = max_norm / combined_norm
                    dW, db = dW * scale, db * scale

            gate = 1.0
            if gate_threshold is not None:
                # How much has this buffer actually deviated from the
                # reference (excludes the absolute detect-rate feature,
                # which isn't a deviation signal). Near-zero deviation ->
                # scale the correction toward zero, since it's likely just
                # sampling noise between two clean windows, not real drift.
                signal_magnitude = np.abs(stats[: 2 * num_aps]).mean()
                gate = float(np.clip(signal_magnitude / gate_threshold, 0.0, 1.0))
                dW, db = dW * gate, db * gate
            gates.append(gate)

            cur_head_w = base_head_w + dW
            cur_head_b = base_head_b + db
            n_firings += 1

        with torch.no_grad():
            adapted_pred[i] = (F.linear(features[i], cur_head_w, cur_head_b)).numpy() + target_mean

    print(f"Hypernetwork fired {n_firings} times over {T} deployment steps (every {cadence} steps, "
          f"reference reset at each of the session boundaries).")
    return frozen_pred, adapted_pred


N_DRIFT_SEEDS = 15  # for the robust multi-seed headline number


def multi_seed_evaluation(split, base_model, base_head_w, base_head_b, target_mean, hypernet,
                           n_seeds=N_DRIFT_SEEDS):
    """A single drift realization (one seed) can make the headline number
    look better or worse than it robustly is. Averaging over many
    independent random drift realizations (different seeds for the same
    severity schedule) gives a defensible mean +/- std instead of one
    draw's luck."""
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
    print(f"Frozen : {frozen_means.mean():.2f}m +/- {frozen_means.std():.2f}m")
    print(f"Adapted: {adapted_means.mean():.2f}m +/- {adapted_means.std():.2f}m")
    print(f"Improvement: {improvements.mean():+.3f}m +/- {improvements.std():.3f}m "
          f"({improvements.mean() / frozen_means.mean() * 100:.1f}% relative)")
    print(f"Seeds where adapted beat frozen: {(improvements > 0).sum()}/{n_seeds}")


def main():
    split = load_split()
    stream = simulate_drift(split)  # seed=0, used for the representative plot below
    base_model, base_head_w, base_head_b, target_mean = load_frozen_base()
    hypernet = load_hypernetwork()

    frozen_pred, adapted_pred = run_deployment(split, stream, base_model, base_head_w, base_head_b,
                                                target_mean, hypernet, gate_threshold=GATE_THRESHOLD,
                                                max_delta_frac=MAX_DELTA_FRAC)

    frozen_err = euclidean_error(frozen_pred, stream.y)
    adapted_err = euclidean_error(adapted_pred, stream.y)

    print(f"\n=== Step 5 result for ONE drift realization (seed=0, for the plot below) ===")
    print(f"Overall mean error -- frozen: {frozen_err.mean():.2f}m   adapted: {adapted_err.mean():.2f}m")
    print(f"Overall median error -- frozen: {np.median(frozen_err):.2f}m   adapted: {np.median(adapted_err):.2f}m")

    n_chunks = 5
    T = len(frozen_err)
    cs = T // n_chunks
    print("\nPer-chunk breakdown (deployment progresses left to right):")
    for c in range(n_chunks):
        sl = slice(c * cs, (c + 1) * cs if c < n_chunks - 1 else T)
        print(f"  chunk {c}: frozen={frozen_err[sl].mean():6.2f}m  "
              f"adapted={adapted_err[sl].mean():6.2f}m  "
              f"improvement={frozen_err[sl].mean() - adapted_err[sl].mean():+.2f}m")

    # Smoothed curves for the plot (raw per-step error is very noisy)
    window = 100
    kernel = np.ones(window) / window
    frozen_smooth = np.convolve(frozen_err, kernel, mode="valid")
    adapted_smooth = np.convolve(adapted_err, kernel, mode="valid")
    steps = np.arange(len(frozen_smooth))

    plt.figure(figsize=(10, 5))
    plt.plot(steps, frozen_smooth, label="Static frozen base network", color="tab:red")
    plt.plot(steps, adapted_smooth, label="Hypernetwork-adapted network", color="tab:blue")
    plt.xlabel(f"Deployment step (t), {window}-step rolling mean")
    plt.ylabel("Localization error (m)")
    plt.title("Localization error vs. deployment time under continuous drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/step5_deployment_result.png", dpi=150)
    print(f"\nSaved plot to {OUT_DIR}/step5_deployment_result.png")

    multi_seed_evaluation(split, base_model, base_head_w, base_head_b, target_mean, hypernet)


if __name__ == "__main__":
    main()
