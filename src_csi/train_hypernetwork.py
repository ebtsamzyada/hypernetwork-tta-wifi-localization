"""
Step 4 (CSI variant): Meta-train the hypernetwork on synthetic drift
episodes, using TRAINING points only (the frozen base model + the
held-out test points are never touched here).

Each meta-training episode, within ONE randomly chosen train reference
POINT (the CSI analogue of a "session" -- see hypernetwork.py's
docstring for why point-dwell, not a global reference, is the right
scope here):
  1. Sample a REFERENCE buffer (kept clean/undrifted) and a separate
     CURRENT buffer (labels kept aside for the loss) from that point's
     own packets.
  2. Perturb the CURRENT buffer with a randomly severity-sampled drift
     snapshot.
  3. Compute stats comparing CURRENT vs REFERENCE -> hypernetwork ->
     (delta_W, delta_b) for the frozen base head.
  4. Run the CURRENT buffer through the FROZEN trunk (no grad) to get
     features, then the ADAPTED head (base_head + delta) to get
     predictions.
  5. Supervised MSE loss against the CURRENT buffer's true (x, y) --
     backprop updates ONLY the hypernetwork's parameters.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit

from base_model import BaseMLP
from data_loader import load_split, scale_features
from drift_simulator import random_drift_snapshot
from hypernetwork import HyperNetwork, compute_buffer_stats

SEED = 42
BUFFER_SIZE = 30
NUM_META_STEPS = 1500
EPISODES_PER_STEP = 8
EVAL_EVERY = 25
EVAL_EPISODES = 60
OUT_DIR = "../outputs_csi"
DELTA_REG_LAMBDA = 50.0  # matches the RSSI pipelines' fix: penalize large
# corrections directly (a loss-level penalty alone wasn't enough there --
# see deploy_simulation.py's inference-time norm clip for the real fix).


def euclidean_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - true, axis=1)


class EpisodeSampler:
    def __init__(self, X_raw: np.ndarray, y: np.ndarray, point_ids: np.ndarray, allowed_points, rng):
        self.pools = {}
        for p in allowed_points:
            mask = point_ids == p
            self.pools[p] = (X_raw[mask], y[mask])
        self.point_list = list(self.pools.keys())
        self.rng = rng

    def sample_pair(self, buffer_size: int):
        p = self.point_list[self.rng.integers(len(self.point_list))]
        pool_X, pool_y = self.pools[p]
        n = len(pool_X)
        ref_idx = self.rng.choice(n, size=buffer_size, replace=n < buffer_size)
        cur_idx = self.rng.choice(n, size=buffer_size, replace=n < buffer_size)
        return pool_X[ref_idx], pool_X[cur_idx], pool_y[cur_idx]


def make_drifted_episode(ref_raw, cur_raw, buf_y, rng, feat_mean, feat_std):
    drifted_raw, severity = random_drift_snapshot(cur_raw, rng)
    ref_scaled = scale_features(ref_raw, feat_mean, feat_std)
    cur_scaled = scale_features(drifted_raw, feat_mean, feat_std)
    return ref_raw, ref_scaled, drifted_raw, cur_scaled, buf_y, severity


def score_episode(hypernet, base_model, base_head_w, base_head_b, target_mean,
                   ref_raw, ref_scaled, cur_raw, cur_scaled, buf_y, train_hypernet: bool):
    X_t = torch.tensor(cur_scaled, dtype=torch.float32)
    y_t = torch.tensor(buf_y - target_mean, dtype=torch.float32)

    with torch.no_grad():
        features = base_model.trunk(X_t)
        frozen_pred = F.linear(features, base_head_w, base_head_b)

    stats = compute_buffer_stats(cur_scaled, ref_scaled)
    stats_t = torch.tensor(stats, dtype=torch.float32)

    ctx = torch.enable_grad() if train_hypernet else torch.no_grad()
    with ctx:
        dW, db = hypernet(stats_t)
        adapted_pred = F.linear(features, base_head_w + dW, base_head_b + db)
        loss = F.mse_loss(adapted_pred, y_t)
        if train_hypernet:
            loss = loss + DELTA_REG_LAMBDA * (dW.pow(2).mean() + db.pow(2).mean())

    return loss, adapted_pred.detach(), frozen_pred


def load_frozen_base(out_dir=OUT_DIR):
    ckpt = torch.load(f"{out_dir}/base_model.pt", weights_only=False)
    base_model = BaseMLP(input_dim=ckpt["input_dim"])
    base_model.load_state_dict(ckpt["state_dict"])
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    target_mean = ckpt["target_mean"]
    base_head_w, base_head_b = base_model.get_head_params()
    return base_model, base_head_w, base_head_b, target_mean


def train_one_run(split, base_model, base_head_w, base_head_b, target_mean,
                   train_points, val_points, num_steps=NUM_META_STEPS, seed=SEED,
                   eval_every=EVAL_EVERY, eval_episodes=EVAL_EPISODES, verbose=True):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    train_sampler = EpisodeSampler(split.X_train_raw, split.y_train, split.point_train,
                                    train_points, rng)

    fixed_val_episodes = []
    if val_points:
        val_sampler = EpisodeSampler(split.X_train_raw, split.y_train, split.point_train,
                                      val_points, np.random.default_rng(seed + 1))
        val_drift_rng = np.random.default_rng(seed + 2)
        fixed_val_episodes = [
            make_drifted_episode(*val_sampler.sample_pair(BUFFER_SIZE), val_drift_rng,
                                  split.feat_mean, split.feat_std)
            for _ in range(eval_episodes)
        ]

    hypernet = HyperNetwork(num_features=split.num_features, head_in=base_head_w.shape[1],
                             head_out=base_head_w.shape[0], hidden=64)
    opt = torch.optim.Adam(hypernet.parameters(), lr=1e-3, weight_decay=1e-5)

    best_val_err, best_frozen_err, best_state = float("inf"), None, None
    history = []

    for step in range(1, num_steps + 1):
        hypernet.train()
        opt.zero_grad()
        for _ in range(EPISODES_PER_STEP):
            ref_raw, cur_raw, buf_y = train_sampler.sample_pair(BUFFER_SIZE)
            ref_raw, ref_scaled, drifted_raw, cur_scaled, buf_y, _ = make_drifted_episode(
                ref_raw, cur_raw, buf_y, rng, split.feat_mean, split.feat_std
            )
            loss, _, _ = score_episode(
                hypernet, base_model, base_head_w, base_head_b, target_mean,
                ref_raw, ref_scaled, drifted_raw, cur_scaled, buf_y, train_hypernet=True,
            )
            (loss / EPISODES_PER_STEP).backward()
        opt.step()

        if fixed_val_episodes and step % eval_every == 0:
            hypernet.eval()
            adapted_errs, frozen_errs = [], []
            with torch.no_grad():
                for ref_raw, ref_scaled, drifted_raw, cur_scaled, buf_y, _ in fixed_val_episodes:
                    _, vadapted, vfrozen = score_episode(
                        hypernet, base_model, base_head_w, base_head_b, target_mean,
                        ref_raw, ref_scaled, drifted_raw, cur_scaled, buf_y, train_hypernet=False,
                    )
                    adapted_errs.append(euclidean_error(vadapted.numpy() + target_mean, buf_y))
                    frozen_errs.append(euclidean_error(vfrozen.numpy() + target_mean, buf_y))
            adapted_mean = np.concatenate(adapted_errs).mean()
            frozen_mean = np.concatenate(frozen_errs).mean()
            history.append((step, adapted_mean, frozen_mean))
            if verbose:
                print(f"  step {step:5d}  meta-val adapted={adapted_mean:6.3f}m  frozen={frozen_mean:6.3f}m")
            if adapted_mean < best_val_err:
                best_val_err, best_frozen_err = adapted_mean, frozen_mean
                best_state = {k: v.clone() for k, v in hypernet.state_dict().items()}

    if best_state is not None:
        hypernet.load_state_dict(best_state)
    else:
        best_val_err, best_frozen_err = None, None

    return {"hypernet": hypernet, "best_val_adapted": best_val_err,
            "best_val_frozen": best_frozen_err, "history": history}


def main():
    split = load_split()
    base_model, base_head_w, base_head_b, target_mean = load_frozen_base()

    all_points = sorted(set(split.point_train))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    tr_idx, val_idx = next(gss.split(split.X_train, split.y_train, split.point_train))
    meta_train_points = sorted(set(split.point_train[tr_idx]))
    meta_val_points = sorted(set(split.point_train[val_idx]))
    assert set(meta_train_points).isdisjoint(meta_val_points)
    print(f"Meta-train points: {len(meta_train_points)}  Meta-val points: {len(meta_val_points)}")

    result = train_one_run(split, base_model, base_head_w, base_head_b, target_mean,
                            meta_train_points, meta_val_points)
    print(f"\nBest meta-val adapted error: {result['best_val_adapted']:.3f} m "
          f"(frozen: {result['best_val_frozen']:.3f} m)")

    torch.save(
        {"state_dict": result["hypernet"].state_dict(), "num_features": split.num_features,
         "head_in": base_head_w.shape[1], "head_out": base_head_w.shape[0], "hidden": 64},
        f"{OUT_DIR}/hypernetwork.pt",
    )
    print(f"Saved hypernetwork to {OUT_DIR}/hypernetwork.pt")


if __name__ == "__main__":
    main()
