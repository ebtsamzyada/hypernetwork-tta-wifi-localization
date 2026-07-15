"""Phase 6: multi-site-backbone oracle diagnostic, retried with SYNTHETIC,
spatially-generic drift instead of HDLC's real (but location-entangled)
Layout 1->2/3 obstruction drift -- see PIVOT_PLAN.md's pivot decision.

Drift mechanism ported directly from the prior project's
`wifi_tta/src/drift_simulator.py::random_drift_snapshot`: a buffer is
perturbed with ONE randomly sampled (walk_sigma, dropout_prob) pair --
i.e. a snapshot of "where in a possible drift timeline are we", not a
ramp within the buffer. Severity is drawn independently per buffer
instance, decoupled from which physical point it is -- directly testing
whether spatially-generic drift (unlike HDLC's real obstruction drift)
produces a signal buffer statistics can actually predict.

Same three-way comparison as Phase 4 (leave-points-out RF vs random-CV RF
vs leave-points-out Ridge vs point-identity-alone oracle), same
reference-relative buffer-stats design, run on the best available
checkpoint (Phase 5c v4 fine-tuned HDLC model) instead of the original
single-site HDLC pipeline.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import HyperParams  # noqa: E402
from multisite_model import MultiSiteGlobLocCNN  # noqa: E402
from sites import (  # noqa: E402
    RSSI_MAX, RSSI_MIN, PRETRAIN_SITE_IDS,
    attach_floor_class, build_floor_class_map, filter_locatable_rows, load_site,
)
from splits import stratified_grouped_split_by_floor  # noqa: E402
from virtual_space import build_virtual_space, generate_image  # noqa: E402

torch.set_num_threads(1)

OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "phase6"
CHECKPOINT = Path(__file__).resolve().parents[1] / "outputs" / "phase5c_v4" / "model_hdlc.pt"
SITE_ID = "hdlc"
FLOOR_HEIGHT_M = 3.5
BUFFER_SIZE = 10
N_BOOTSTRAP = 8          # drift instances per held-out point
WALK_SIGMA_RANGE = (0.5, 20.0)   # dBm, matches prior project's random_drift_snapshot
DROPOUT_RANGE = (0.0, 0.6)
N_FOLDS = 5
SEED = 42


def _point_id(row):
    return f"{row['floor_id']}_{row['x']}_{row['y']}"


def random_drift_snapshot(rssi_buf: np.ndarray, rng: np.random.Generator):
    """Ported from wifi_tta/src/drift_simulator.py. rssi_buf: (B, D) clean
    RSSI. Returns (drifted_buf, severity, dropout_prob)."""
    buf = rssi_buf.astype(np.float32).copy()
    B, D = buf.shape
    severity = rng.uniform(*WALK_SIGMA_RANGE)
    dropout_prob = rng.uniform(*DROPOUT_RANGE)

    offset = (rng.normal(0.0, 1.0, size=D) * severity).astype(np.float32)
    visible = buf > RSSI_MIN
    offset_b = np.broadcast_to(offset, buf.shape)
    buf[visible] = np.clip(buf[visible] + offset_b[visible], RSSI_MIN, RSSI_MAX)

    flip = visible & (rng.random(buf.shape) < dropout_prob)
    buf[flip] = RSSI_MIN
    return buf, severity, dropout_prob


def compute_buffer_stats(cur_raw: np.ndarray, ref_raw: np.ndarray) -> np.ndarray:
    """Reference-relative buffer stats -- same design as Phase 4's
    oracle diagnostic and the prior project's hypernetwork.py."""
    cur_detect = (cur_raw > RSSI_MIN).mean(axis=0)
    ref_detect = (ref_raw > RSSI_MIN).mean(axis=0)
    detect_diff = cur_detect - ref_detect

    cur_scaled = np.clip((cur_raw - RSSI_MIN) / (RSSI_MAX - RSSI_MIN), 0, 1)
    ref_scaled = np.clip((ref_raw - RSSI_MIN) / (RSSI_MAX - RSSI_MIN), 0, 1)
    mean_diff = cur_scaled.mean(axis=0) - ref_scaled.mean(axis=0)

    return np.concatenate([detect_diff, mean_diff, cur_detect]).astype(np.float32)


TOTAL_SIGNAL_LOSS_PENALTY_M = 30.0  # severe synthetic drift can occasionally zero out
                                     # every visible AP in a reading -- a genuine total
                                     # localization failure, not a bug to hide by
                                     # skipping the row (that would bias the oracle
                                     # diagnostic's target to look better than reality
                                     # under severe drift). Chosen larger than any real
                                     # observed error (~1-3m) but not an absurd outlier,
                                     # roughly matching this site's naive-baseline scale.


@torch.no_grad()
def per_row_combined_3d_err(model, rows_df, wap_pos, wap_cols, hp, rssi_override=None):
    """rssi_override: optional (n_rows, n_wap) array to use INSTEAD of the
    df's own columns (i.e. the drifted buffer), same row order as rows_df."""
    errs = np.zeros(len(rows_df), dtype=np.float32)
    for i, (_, row) in enumerate(rows_df.iterrows()):
        x, y = float(row["x"]), float(row["y"])
        floor_class = int(row["floor_class"])
        rssi_row = rssi_override[i] if rssi_override is not None else row[wap_cols].to_numpy(dtype=np.float32)
        try:
            ap_info, (vx, vy), (x_min, y_min), img_hw = build_virtual_space(
                x, y, rssi_row, wap_cols, wap_pos, hp.k_strongest, hp.margin_m, hp.max_extent_m,
                rssi_min=RSSI_MIN, rssi_max=RSSI_MAX,
            )
        except ValueError:
            errs[i] = TOTAL_SIGNAL_LOSS_PENALTY_M
            continue
        img = torch.from_numpy(generate_image(ap_info, img_hw)).unsqueeze(0)
        xy_pred, floor_pred = model(img, SITE_ID)
        pred_xy = xy_pred.numpy()[0] * 0.5 + np.array([x_min, y_min])
        xy_err = float(np.linalg.norm(pred_xy - np.array([x, y])))
        num_floors = model.site_num_floors[SITE_ID]
        pred_floor_class = int(round(min(max(float(floor_pred.item()), 0.0), num_floors - 1)))
        errs[i] = np.sqrt(xy_err ** 2 + (FLOOR_HEIGHT_M * abs(pred_floor_class - floor_class)) ** 2)
    return errs


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    hp = HyperParams()

    print("Loading HDLC + reconstructing the exact v4 stratified split...")
    df, wap_pos, wap_cols = load_site(SITE_ID)
    floor_map = build_floor_class_map(df)
    df = attach_floor_class(df, floor_map)
    df = filter_locatable_rows(df, wap_pos, wap_cols, hp.k_strongest, SITE_ID)
    train_df, val_df, test_df = stratified_grouped_split_by_floor(df, "_group", "floor_id",
                                                                    val_frac=0.15, test_frac=0.15, seed=SEED)
    held_out_df = pd.concat([val_df, test_df], ignore_index=True)
    print(f"Using {held_out_df['_group'].nunique()} points held out from training (val+test), "
          f"never memorized by the frozen model.\n")

    site_num_floors = {}
    for site_id in PRETRAIN_SITE_IDS:
        d, _, _ = load_site(site_id)
        site_num_floors[site_id] = len(build_floor_class_map(d))

    model = MultiSiteGlobLocCNN(site_num_floors=site_num_floors)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu"))
    model.eval()

    X, y_target, groups = [], [], []
    by_point = held_out_df.assign(_pid=held_out_df.apply(_point_id, axis=1)).groupby("_pid")

    for pid, rows in by_point:
        clean_raw = rows[wap_cols].to_numpy(dtype=np.float32)
        n = len(clean_raw)
        for _ in range(N_BOOTSTRAP):
            ref_idx = rng.integers(0, n, size=min(BUFFER_SIZE, n))
            cur_idx = rng.integers(0, n, size=min(BUFFER_SIZE, n))
            ref_buf = clean_raw[ref_idx]
            cur_buf_clean = clean_raw[cur_idx]
            cur_buf_drifted, severity, dropout_prob = random_drift_snapshot(cur_buf_clean, rng)

            stats = compute_buffer_stats(cur_buf_drifted, ref_buf)
            cur_rows = rows.iloc[cur_idx]
            errs = per_row_combined_3d_err(model, cur_rows, wap_pos, wap_cols, hp, rssi_override=cur_buf_drifted)
            target = float(np.median(errs))

            X.append(stats)
            y_target.append(target)
            groups.append(pid)

    X = np.stack(X)
    y_target = np.array(y_target, dtype=np.float64)
    groups = np.array(groups)
    print(f"Built {len(X)} synthetic-drift buffer instances ({X.shape[1]}-dim stats) "
          f"across {len(set(groups))} held-out points.\n")

    gkf = GroupKFold(n_splits=N_FOLDS)
    rf_grouped_pred = np.zeros_like(y_target)
    for tr_idx, te_idx in gkf.split(X, y_target, groups=groups):
        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(X[tr_idx], y_target[tr_idx])
        rf_grouped_pred[te_idx] = rf.predict(X[te_idx])
    r2_rf_grouped = r2_score(y_target, rf_grouped_pred)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rf_random_pred = np.zeros_like(y_target)
    for tr_idx, te_idx in kf.split(X):
        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
        rf.fit(X[tr_idx], y_target[tr_idx])
        rf_random_pred[te_idx] = rf.predict(X[te_idx])
    r2_rf_random = r2_score(y_target, rf_random_pred)

    ridge_grouped_pred = np.zeros_like(y_target)
    for tr_idx, te_idx in gkf.split(X, y_target, groups=groups):
        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(X[tr_idx], y_target[tr_idx])
        ridge_grouped_pred[te_idx] = ridge.predict(X[te_idx])
    r2_ridge_grouped = r2_score(y_target, ridge_grouped_pred)

    point_mean = pd.Series(y_target).groupby(groups).transform("mean").to_numpy()
    r2_identity = r2_score(y_target, point_mean)

    print("=" * 70)
    print("Phase 6 oracle diagnostic results (synthetic, spatially-generic drift)")
    print("=" * 70)
    print(f"Leave-points-out RandomForest R^2:      {r2_rf_grouped:.3f}")
    print(f"Random (non-grouped) RandomForest R^2:  {r2_rf_random:.3f}")
    print(f"Leave-points-out Ridge (linear) R^2:     {r2_ridge_grouped:.3f}")
    print(f"Point-identity ALONE R^2 (oracle):       {r2_identity:.3f}")
    print()

    if r2_identity > max(r2_rf_grouped, r2_ridge_grouped):
        print("*** WARNING: point identity alone predicts more error variance than "
              "the drift-buffer statistics do. Confound warning sign, NOT a green "
              "light for hypernetwork training. ***")
    elif max(r2_rf_grouped, r2_ridge_grouped) > 0.05:
        print("Real, non-confounded signal found: buffer statistics predict "
              "drift-induced error better than point identity alone, and "
              "leave-points-out R^2 is meaningfully positive. Proceeding to "
              "hypernetwork training is defensible.")
    else:
        print("Leave-points-out R^2 is low/near-zero for both RF and Ridge -- "
              "no clear signal either way; treat as inconclusive, not a green light.")

    if abs(r2_rf_grouped - r2_rf_random) > 0.05:
        print(f"*** NOTE: grouped vs random-CV RF R^2 differ by "
              f"{abs(r2_rf_grouped - r2_rf_random):.3f} -- possible point-identity "
              f"leak through the split. ***")

    import json
    results = {
        "drift_mechanism": "synthetic_random_walk_dropout",
        "n_instances": len(X), "n_points": len(set(groups)), "stat_dim": int(X.shape[1]),
        "r2_rf_leave_points_out": r2_rf_grouped,
        "r2_rf_random_cv": r2_rf_random,
        "r2_ridge_leave_points_out": r2_ridge_grouped,
        "r2_point_identity_oracle": r2_identity,
    }
    with open(OUTDIR / "oracle_diagnostic_synthetic_drift.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTDIR / 'oracle_diagnostic_synthetic_drift.json'}")


if __name__ == "__main__":
    main()
