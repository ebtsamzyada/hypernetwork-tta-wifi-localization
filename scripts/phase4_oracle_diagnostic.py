"""Phase 4 prerequisite: oracle diagnostic. Must run BEFORE any hypernetwork
training code, per the non-negotiable discipline -- check whether a real,
non-confounded, transferable signal exists in buffer statistics before
building anything that learns from them.

Ported design from the prior project's `wifi_tta/src/hypernetwork.py` and
its CSI pipeline's diagnostic (`wifi_tta/CSI_PIVOT_PLAN.md`, "Diagnostic:
is the buffer-stats signal linear, nonlinear, or a location confound?"):

- Buffer statistics compare a CURRENT buffer against a REFERENCE buffer
  from the same physical point but an earlier/cleaner session (here:
  reference = Layout 1 samples at a point, current = Layout 2/3 samples at
  the same point -- see PIVOT_PLAN.md Item 4). Reference-RELATIVE stats,
  not absolute ones -- the prior project found absolute per-AP stats gave
  ~0 generalizable signal because they're dominated by which location a
  buffer is from, not by drift severity.
- Compare three predictors of a frozen model's per-buffer error:
    (a) leave-points-out RandomForest R^2 on buffer stats (the real check)
    (b) random (non-grouped) RandomForest R^2 (confound check: if much
        higher than (a), buffer stats are leaking point identity)
    (c) point-identity-ALONE oracle R^2 (deliberately unfair/leaky: predict
        every buffer's error as that point's own mean error). If this
        beats (a), point identity explains error better than drift-buffer
        statistics do -- a warning sign per the non-negotiable discipline,
        not a green light for hypernetwork training.
  A ridge (linear) leave-points-out R^2 is also reported, since the prior
  project found the CSI signal was nonlinear (~0 for ridge, ~0.22 for RF)
  -- worth checking here too before assuming either shape.

Uses ONLY the 116 physical points held out from Phase 2 model training
(val + test groups, 58+58) so the per-buffer error target reflects a
frozen model that has genuinely never seen these points -- using
train-set points here would contaminate the target with memorization.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loading import (  # noqa: E402
    RSSI_MAX, RSSI_MIN,
    attach_floor_class, build_floor_class_map, filter_locatable_rows, load_raw_layout_csv,
)
from dataset import HyperParams  # noqa: E402
from model import GlobLocCNN  # noqa: E402
from splits import grouped_split  # noqa: E402
from virtual_space import build_virtual_space, generate_image  # noqa: E402

torch.set_num_threads(1)  # see src/train.py -- multithread dispatch overhead dominates at this image scale

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "Hybrid-fingerprint Data with Layout Change (HDLC)"
PHASE2_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase2"
OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "phase4"

FLOOR_HEIGHT_M = 3.5
BUFFER_SIZE = 10          # matches HDLC's natural 10-sample test burst per point
N_BOOTSTRAP = 5           # buffer instances drawn per (point, layout) condition
N_FOLDS = 5
SEED = 42


def _point_id(row):
    return f"{row['floor']}_{row['x']}_{row['y']}"


def load_layout_pool(layout_dir, layout_label):
    train_raw, wap_pos, wap_cols = load_raw_layout_csv(layout_dir / f"{layout_label} - Raw - Training.csv")
    test_raw, _, _ = load_raw_layout_csv(layout_dir / f"{layout_label} - Raw - Testing.csv")
    return pd.concat([train_raw, test_raw], ignore_index=True), wap_pos, wap_cols


@torch.no_grad()
def per_row_combined_3d_err(model, df, wap_pos, wap_cols, hp):
    """Frozen-model combined 3D error for every row in df (no augmentation
    -- this is an eval-time error target, not a training signal)."""
    errs = np.zeros(len(df), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        x, y = float(row["x"]), float(row["y"])
        floor_class = int(row["floor_class"])
        rssi_row = row[wap_cols].to_numpy(dtype=np.float32)
        ap_info, (vx, vy), (x_min, y_min), img_hw = build_virtual_space(
            x, y, rssi_row, wap_cols, wap_pos, hp.k_strongest, hp.margin_m, hp.max_extent_m,
        )
        img = torch.from_numpy(generate_image(ap_info, img_hw)).unsqueeze(0)
        xy_pred, floor_logits = model(img)
        pred_xy = xy_pred.numpy()[0] * 0.5 + np.array([x_min, y_min])  # PIXEL_METERS=0.5
        xy_err = float(np.linalg.norm(pred_xy - np.array([x, y])))
        pred_floor_class = int(torch.argmax(floor_logits, dim=1).item())
        errs[i] = np.sqrt(xy_err ** 2 + (FLOOR_HEIGHT_M * abs(pred_floor_class - floor_class)) ** 2)
    return errs


def compute_buffer_stats(cur_raw: np.ndarray, ref_raw: np.ndarray) -> np.ndarray:
    """(n_samples, D) raw dBm arrays -> (3*D,) stat vector. Reference-
    relative, not absolute -- see module docstring."""
    cur_detect = (cur_raw > RSSI_MIN).mean(axis=0)
    ref_detect = (ref_raw > RSSI_MIN).mean(axis=0)
    detect_diff = cur_detect - ref_detect

    cur_scaled = np.clip((cur_raw - RSSI_MIN) / (RSSI_MAX - RSSI_MIN), 0, 1)
    ref_scaled = np.clip((ref_raw - RSSI_MIN) / (RSSI_MAX - RSSI_MIN), 0, 1)
    mean_diff = cur_scaled.mean(axis=0) - ref_scaled.mean(axis=0)

    return np.concatenate([detect_diff, mean_diff, cur_detect]).astype(np.float32)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    l1_pool, wap_pos, wap_cols = load_layout_pool(DATA_ROOT / "Layout 1", "Layout 1")
    floor_map = build_floor_class_map(l1_pool)
    l1_pool = attach_floor_class(l1_pool, floor_map)
    l1_pool = filter_locatable_rows(l1_pool, wap_pos, wap_cols, HyperParams().k_strongest, "Layout 1 pool")
    train_df, val_df, test_df = grouped_split(l1_pool, val_frac=0.15, test_frac=0.15, seed=SEED)
    held_out_df = pd.concat([val_df, test_df], ignore_index=True)
    held_out_point_ids = set(_point_id(r) for _, r in held_out_df.iterrows())
    print(f"Using {len(held_out_point_ids)} points held out from Phase 2 training (val+test).\n")

    # reference-buffer source rows, keyed by point id (Layout 1, never trained on for these points)
    ref_rows_by_point = {pid: g for pid, g in held_out_df.assign(_pid=held_out_df.apply(_point_id, axis=1)).groupby("_pid")}

    hp = HyperParams()
    model = GlobLocCNN(num_floors=len(floor_map))
    model.load_state_dict(torch.load(PHASE2_DIR / "model.pt", map_location="cpu"))
    model.eval()

    X, y_target, groups = [], [], []

    for layout_dirname in ["Layout 2", "Layout 3"]:
        pool, _, _ = load_layout_pool(DATA_ROOT / layout_dirname, layout_dirname)
        pool = attach_floor_class(pool, floor_map)
        pool = filter_locatable_rows(pool, wap_pos, wap_cols, hp.k_strongest, layout_dirname)
        pool["_pid"] = pool.apply(_point_id, axis=1)
        pool = pool[pool["_pid"].isin(held_out_point_ids)].reset_index(drop=True)
        print(f"[{layout_dirname}] scoring {len(pool)} rows with the frozen Phase 2 model...")
        pool["_err"] = per_row_combined_3d_err(model, pool, wap_pos, wap_cols, hp)

        for pid, cur_rows in pool.groupby("_pid"):
            ref_rows = ref_rows_by_point[pid]
            cur_raw_all = cur_rows[wap_cols].to_numpy(dtype=np.float32)
            ref_raw_all = ref_rows[wap_cols].to_numpy(dtype=np.float32)
            cur_err_all = cur_rows["_err"].to_numpy(dtype=np.float32)

            for _ in range(N_BOOTSTRAP):
                cur_idx = rng.integers(0, len(cur_rows), size=min(BUFFER_SIZE, len(cur_rows)) if len(cur_rows) < BUFFER_SIZE else BUFFER_SIZE)
                ref_idx = rng.integers(0, len(ref_rows), size=min(BUFFER_SIZE, len(ref_rows)) if len(ref_rows) < BUFFER_SIZE else BUFFER_SIZE)
                stats = compute_buffer_stats(cur_raw_all[cur_idx], ref_raw_all[ref_idx])
                target = float(np.median(cur_err_all[cur_idx]))
                X.append(stats)
                y_target.append(target)
                groups.append(pid)

    X = np.stack(X)
    y_target = np.array(y_target, dtype=np.float64)
    groups = np.array(groups)
    print(f"\nBuilt {len(X)} buffer instances ({X.shape[1]}-dim stats) across "
          f"{len(set(groups))} held-out points.\n")

    # (a) leave-points-out RandomForest
    gkf = GroupKFold(n_splits=N_FOLDS)
    rf_grouped_pred = np.zeros_like(y_target)
    for tr_idx, te_idx in gkf.split(X, y_target, groups=groups):
        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(X[tr_idx], y_target[tr_idx])
        rf_grouped_pred[te_idx] = rf.predict(X[te_idx])
    r2_rf_grouped = r2_score(y_target, rf_grouped_pred)

    # (b) random (non-grouped) RandomForest -- confound check
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rf_random_pred = np.zeros_like(y_target)
    for tr_idx, te_idx in kf.split(X):
        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(X[tr_idx], y_target[tr_idx])
        rf_random_pred[te_idx] = rf.predict(X[te_idx])
    r2_rf_random = r2_score(y_target, rf_random_pred)

    # (c) leave-points-out Ridge -- linear-signal check
    ridge_grouped_pred = np.zeros_like(y_target)
    for tr_idx, te_idx in gkf.split(X, y_target, groups=groups):
        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(X[tr_idx], y_target[tr_idx])
        ridge_grouped_pred[te_idx] = ridge.predict(X[te_idx])
    r2_ridge_grouped = r2_score(y_target, ridge_grouped_pred)

    # (d) point-identity ALONE, oracle (deliberately unfair -- global point-mean, no holdout)
    point_mean = pd.Series(y_target).groupby(groups).transform("mean").to_numpy()
    r2_identity = r2_score(y_target, point_mean)

    print("=" * 70)
    print("Oracle diagnostic results")
    print("=" * 70)
    print(f"Leave-points-out RandomForest R^2:      {r2_rf_grouped:.3f}")
    print(f"Random (non-grouped) RandomForest R^2:  {r2_rf_random:.3f}")
    print(f"Leave-points-out Ridge (linear) R^2:     {r2_ridge_grouped:.3f}")
    print(f"Point-identity ALONE R^2 (oracle):       {r2_identity:.3f}")
    print()

    if r2_identity > r2_rf_grouped:
        print("*** WARNING: point identity alone predicts more error variance than "
              "the drift-buffer statistics do. This is a confound warning sign per "
              "the non-negotiable discipline -- NOT a green light for hypernetwork "
              "training. ***")
    else:
        print("Point identity does not dominate the buffer-stats signal -- "
              "proceeding to hypernetwork training is defensible.")

    if abs(r2_rf_grouped - r2_rf_random) > 0.05:
        print(f"*** NOTE: grouped vs random-CV RF R^2 differ by "
              f"{abs(r2_rf_grouped - r2_rf_random):.3f} -- buffer stats may be "
              f"leaking point identity through the split. ***")

    results = {
        "n_instances": len(X), "n_points": len(set(groups)), "stat_dim": int(X.shape[1]),
        "r2_rf_leave_points_out": r2_rf_grouped,
        "r2_rf_random_cv": r2_rf_random,
        "r2_ridge_leave_points_out": r2_ridge_grouped,
        "r2_point_identity_oracle": r2_identity,
    }
    import json
    with open(OUTDIR / "oracle_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTDIR / 'oracle_diagnostic.json'}")


if __name__ == "__main__":
    main()
