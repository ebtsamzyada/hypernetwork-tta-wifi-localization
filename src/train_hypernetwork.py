"""
Step 4: Meta-train the hypernetwork on synthetic drift episodes, using
TRAINING data only (the base model's frozen trunk/head + the held-out
test set are never touched here).

Each meta-training episode, within ONE randomly chosen train session:
  1. Sample a REFERENCE buffer (kept clean/undrifted -- stands in for the
     first rolling window collected right after deployment starts at a
     site, before drift has accumulated) and a separate CURRENT buffer
     (with labels kept aside for the loss) from the same session.
  2. Perturb the CURRENT buffer with a randomly severity-sampled drift
     snapshot (random walk-sigma AND random dropout probability), so the
     hypernetwork meta-learns to handle the whole range of severities it
     may see across a real deployment timeline.
  3. Compute stats comparing CURRENT vs REFERENCE (mean/detect-rate
     DIFFERENCE, see hypernetwork.py) -> hypernetwork -> (delta_W,
     delta_b) for the frozen base head. Comparing against a same-session
     reference (instead of absolute stats) is what makes this generalize
     across held-out sessions -- see hypernetwork.py's docstring for the
     diagnostic that led to this design.
  4. Run the CURRENT buffer through the FROZEN trunk (no grad) to get
     features, then the ADAPTED head (base_head + delta) to get
     predictions.
  5. Supervised MSE loss against the CURRENT buffer's true (x, y) --
     backprop updates ONLY the hypernetwork's parameters.

`train_one_run` is the reusable training/eval loop -- cv_hypernetwork.py
calls it once per cross-validation fold (to get an honest, session
-variance-robust estimate of the adaptation benefit) and once more on ALL
train sessions (to produce the final checkpoint used in Step 5).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit

from base_model import BaseMLP, FloorAwareBaseModel
from data_loader import load_split, scale_rssi
from drift_simulator import random_drift_snapshot
from floor_model import FloorClassifier
from hypernetwork import HyperNetwork, compute_buffer_stats

SEED = 42
BUFFER_SIZE = 64
NUM_META_STEPS = 1500
EPISODES_PER_STEP = 8  # gradient-accumulate over K episodes/step to tame variance
EVAL_EVERY = 25
EVAL_EPISODES = 60
OUT_DIR = "../outputs"
DELTA_REG_LAMBDA = 50.0  # penalize large delta-weight/bias to avoid rare destabilizing corrections
# (raised from 2.0: measured dW norms of 7-9 vs. the base head's own weight
# norm of only 4.69 -- the "correction" was overwhelming the learned
# mapping rather than nudging it, which is what caused rare catastrophic
# per-session regressions)


def euclidean_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - true, axis=1)


class EpisodeSampler:
    def __init__(self, raw100: np.ndarray, y: np.ndarray, groups: np.ndarray, allowed_groups, rng):
        self.pools = {}
        for g in allowed_groups:
            mask = groups == g
            self.pools[g] = (raw100[mask], y[mask])
        self.group_list = list(self.pools.keys())
        self.rng = rng

    def sample_pair(self, buffer_size: int):
        """Reference buffer (clean) + current buffer (to be drifted),
        independently sampled from the SAME session."""
        g = self.group_list[self.rng.integers(len(self.group_list))]
        pool_raw100, pool_y = self.pools[g]
        n = len(pool_raw100)
        ref_idx = self.rng.choice(n, size=buffer_size, replace=n < buffer_size)
        cur_idx = self.rng.choice(n, size=buffer_size, replace=n < buffer_size)
        return pool_raw100[ref_idx], pool_raw100[cur_idx], pool_y[cur_idx]


def make_drifted_episode(ref_raw100, cur_raw100, buf_y, rng, rssi_min, rssi_max):
    """Sample ONE random drift severity and apply it to the CURRENT buffer
    only -- the (nondeterministic) part of building an episode."""
    drifted_raw100, severity, dropout_prob = random_drift_snapshot(cur_raw100, rng)
    ref_scaled = scale_rssi(ref_raw100, rssi_min, rssi_max)
    cur_scaled = scale_rssi(drifted_raw100, rssi_min, rssi_max)
    return ref_raw100, ref_scaled, drifted_raw100, cur_scaled, buf_y, severity, dropout_prob


def score_episode(hypernet, base_model, base_head_w, base_head_b, target_mean,
                   ref_raw100, ref_scaled, cur_raw100, cur_scaled, buf_y, train_hypernet: bool):
    """Deterministic given its inputs -- no RNG here, so calling this twice
    with the same inputs and the same hypernetwork weights gives identical
    results."""
    X_t = torch.tensor(cur_scaled, dtype=torch.float32)
    y_t = torch.tensor(buf_y - target_mean, dtype=torch.float32)

    with torch.no_grad():
        features = base_model.trunk(X_t)
        frozen_pred = F.linear(features, base_head_w, base_head_b)

    stats = compute_buffer_stats(cur_raw100, cur_scaled, ref_raw100, ref_scaled)
    stats_t = torch.tensor(stats, dtype=torch.float32)

    ctx = torch.enable_grad() if train_hypernet else torch.no_grad()
    with ctx:
        dW, db = hypernet(stats_t)
        adapted_pred = F.linear(features, base_head_w + dW, base_head_b + db)
        loss = F.mse_loss(adapted_pred, y_t)
        if train_hypernet:
            # Discourage large corrections unless the fit strongly justifies
            # them -- rare oversized deltas otherwise occasionally destabilize
            # predictions badly (observed as a severe regression in one CV fold).
            loss = loss + DELTA_REG_LAMBDA * (dW.pow(2).mean() + db.pow(2).mean())

    return loss, adapted_pred.detach(), frozen_pred


def load_frozen_base(out_dir=OUT_DIR):
    """Loads the floor classifier + localization MLP as one frozen
    FloorAwareBaseModel. Everything downstream (meta-training, deployment)
    keeps calling `.trunk(x_scaled_rssi)` exactly as before -- the RSSI ++
    predicted-floor-probs augmentation happens transparently inside the
    wrapper. `base_head_w/b` are still just the localization MLP's head,
    which is the only thing the hypernetwork ever adapts."""
    ckpt = torch.load(f"{out_dir}/base_model.pt", weights_only=False)
    floor_clf = FloorClassifier(input_dim=ckpt["clf_input_dim"])
    floor_clf.load_state_dict(ckpt["clf_state_dict"])
    loc_mlp = BaseMLP(input_dim=ckpt["input_dim"])
    loc_mlp.load_state_dict(ckpt["state_dict"])
    base_model = FloorAwareBaseModel(floor_clf, loc_mlp)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    target_mean = ckpt["target_mean"]
    base_head_w, base_head_b = base_model.get_head_params()
    return base_model, base_head_w, base_head_b, target_mean


def train_one_run(split, base_model, base_head_w, base_head_b, target_mean,
                   train_groups, val_groups, num_steps=NUM_META_STEPS, seed=SEED,
                   eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES, verbose=True):
    """Meta-train a hypernetwork using episodes drawn only from
    `train_groups`. If `val_groups` is non-empty, tracks a fixed set of
    validation episodes for checkpoint selection and returns curves; the
    val sessions are assumed disjoint from train_groups by the caller.
    If `val_groups` is empty, just trains for num_steps and returns the
    final weights (used for the no-validation-available production run).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    train_sampler = EpisodeSampler(split.X_train_raw100, split.y_train, split.train_groups,
                                    train_groups, rng)

    fixed_val_episodes = []
    if val_groups:
        val_sampler = EpisodeSampler(split.X_train_raw100, split.y_train, split.train_groups,
                                      val_groups, np.random.default_rng(seed + 1))
        val_drift_rng = np.random.default_rng(seed + 2)
        fixed_val_episodes = [
            make_drifted_episode(*val_sampler.sample_pair(BUFFER_SIZE), val_drift_rng, split.rssi_min, split.rssi_max)
            for _ in range(eval_episodes)
        ]

    hypernet = HyperNetwork(num_aps=len(split.wap_cols), head_in=base_head_w.shape[1],
                             head_out=base_head_w.shape[0], hidden=128)
    opt = torch.optim.Adam(hypernet.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_err, best_frozen_err, best_state = float("inf"), None, None
    history = []

    for step in range(1, num_steps + 1):
        hypernet.train()
        opt.zero_grad()
        for _ in range(EPISODES_PER_STEP):
            ref_raw100, cur_raw100, buf_y = train_sampler.sample_pair(BUFFER_SIZE)
            ref_raw100, ref_scaled, drifted_raw100, cur_scaled, buf_y, _, _ = make_drifted_episode(
                ref_raw100, cur_raw100, buf_y, rng, split.rssi_min, split.rssi_max
            )
            loss, _, _ = score_episode(
                hypernet, base_model, base_head_w, base_head_b, target_mean,
                ref_raw100, ref_scaled, drifted_raw100, cur_scaled, buf_y, train_hypernet=True,
            )
            (loss / EPISODES_PER_STEP).backward()
        opt.step()

        if fixed_val_episodes and step % eval_every == 0:
            hypernet.eval()
            adapted_errs, frozen_errs = [], []
            with torch.no_grad():
                for ref_raw100, ref_scaled, drifted_raw100, cur_scaled, buf_y, _, _ in fixed_val_episodes:
                    _, vadapted, vfrozen = score_episode(
                        hypernet, base_model, base_head_w, base_head_b, target_mean,
                        ref_raw100, ref_scaled, drifted_raw100, cur_scaled, buf_y, train_hypernet=False,
                    )
                    adapted_errs.append(euclidean_error(vadapted.numpy() + target_mean, buf_y))
                    frozen_errs.append(euclidean_error(vfrozen.numpy() + target_mean, buf_y))
            adapted_mean = np.concatenate(adapted_errs).mean()
            frozen_mean = np.concatenate(frozen_errs).mean()
            history.append((step, adapted_mean, frozen_mean))
            if verbose:
                print(f"  step {step:5d}  meta-val adapted={adapted_mean:6.2f}m  frozen={frozen_mean:6.2f}m")
            if adapted_mean < best_val_err:
                best_val_err, best_frozen_err = adapted_mean, frozen_mean
                best_state = {k: v.clone() for k, v in hypernet.state_dict().items()}

    if best_state is not None:
        hypernet.load_state_dict(best_state)
    else:
        best_val_err, best_frozen_err = None, None

    return {
        "hypernet": hypernet,
        "best_val_adapted": best_val_err,
        "best_val_frozen": best_frozen_err,
        "history": history,
    }


def main():
    split = load_split()
    base_model, base_head_w, base_head_b, target_mean = load_frozen_base()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_idx, val_idx = next(gss.split(split.X_train, split.y_train, split.train_groups))
    meta_train_groups = sorted(set(split.train_groups[tr_idx]))
    meta_val_groups = sorted(set(split.train_groups[val_idx]))
    assert set(meta_train_groups).isdisjoint(meta_val_groups)
    print(f"Meta-train sessions: {meta_train_groups}")
    print(f"Meta-val sessions  : {meta_val_groups}")

    result = train_one_run(split, base_model, base_head_w, base_head_b, target_mean,
                            meta_train_groups, meta_val_groups)
    print(f"\nBest meta-val adapted error: {result['best_val_adapted']:.2f} m "
          f"(frozen: {result['best_val_frozen']:.2f} m)")

    torch.save(
        {
            "state_dict": result["hypernet"].state_dict(),
            "num_aps": len(split.wap_cols),
            "head_in": base_head_w.shape[1],
            "head_out": base_head_w.shape[0],
            "hidden": 128,
        },
        f"{OUT_DIR}/hypernetwork.pt",
    )
    print(f"Saved hypernetwork to {OUT_DIR}/hypernetwork.pt")


if __name__ == "__main__":
    main()
