"""Phase 6b: multi-site-POOLED oracle diagnostic. Same synthetic,
spatially-generic drift mechanism as Phase 6 (scripts/phase6_oracle_diagnostic.py),
but pools buffer instances across HDLC + SODIndoorLoc + UJIIndoorLoc instead
of HDLC alone, to test whether "point identity" dominating the single-site
result (Phase 6: R^2=0.333) was partly an artifact of HDLC only having 116
distinct points to memorize -- a genuinely site-agnostic drift signal should
stay informative when pooled across hundreds of points spanning
architecturally different buildings, while identity gets far less powerful.

Per-AP buffer stats (3*num_APs dims) can't be concatenated across sites --
HDLC has 17 APs, UJI has 520. Switched to FIXED-SIZE aggregate statistics
(mean/std/max of the per-AP detect-rate-diff and RSSI-diff, plus overall
current detect rate) so every site produces a directly comparable feature
vector. "Identity" here means (site_id, point), not just point, matching
the pooled setting -- this is the more informative version of the same
confound check, not a different question.
"""
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
from phase6_oracle_diagnostic import random_drift_snapshot, TOTAL_SIGNAL_LOSS_PENALTY_M  # noqa: E402

torch.set_num_threads(1)

OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "phase6b"
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase5c_v4"
FLOOR_HEIGHT_M = 3.5
BUFFER_SIZE = 10
N_BOOTSTRAP = 6
N_FOLDS = 5
SEED = 42
# Only sites with a floor head AND enough points to be worth pooling
# (uji_b1/uji_b2 have very few groups -- 12/16 -- but the per-buffer
# instances still add real diversity of AP layout/site character, so kept).
POOL_SITES = ["hdlc", "sod_cetc331", "sod_hcxy", "sod_syl", "uji_b1", "uji_b2"]


def _point_id(row):
    return f"{row['floor_id']}_{row['x']}_{row['y']}"


def compute_aggregate_buffer_stats(cur_raw: np.ndarray, ref_raw: np.ndarray) -> np.ndarray:
    """Fixed-size (9-dim), dimension-independent version of Phase 6's
    per-AP buffer stats -- summary statistics across whatever APs this
    site has, instead of one feature per AP, so sites with different AP
    counts produce directly comparable feature vectors."""
    cur_detect = (cur_raw > RSSI_MIN).mean(axis=0)
    ref_detect = (ref_raw > RSSI_MIN).mean(axis=0)
    detect_diff = cur_detect - ref_detect

    cur_scaled = np.clip((cur_raw - RSSI_MIN) / (RSSI_MAX - RSSI_MIN), 0, 1)
    ref_scaled = np.clip((ref_raw - RSSI_MIN) / (RSSI_MAX - RSSI_MIN), 0, 1)
    mean_diff = cur_scaled.mean(axis=0) - ref_scaled.mean(axis=0)

    return np.array([
        detect_diff.mean(), detect_diff.std(), np.abs(detect_diff).max(),
        mean_diff.mean(), mean_diff.std(), np.abs(mean_diff).max(),
        cur_detect.mean(), cur_detect.std(),
        float(cur_raw.shape[1]),  # site AP-count, a legitimate site-character feature
    ], dtype=np.float32)


@torch.no_grad()
def per_row_combined_3d_err(model, rows_df, wap_pos, wap_cols, hp, site_id, rssi_override):
    errs = np.zeros(len(rows_df), dtype=np.float32)
    for i, (_, row) in enumerate(rows_df.iterrows()):
        x, y = float(row["x"]), float(row["y"])
        floor_class = int(row["floor_class"])
        rssi_row = rssi_override[i]
        try:
            ap_info, (vx, vy), (x_min, y_min), img_hw = build_virtual_space(
                x, y, rssi_row, wap_cols, wap_pos, hp.k_strongest, hp.margin_m, hp.max_extent_m,
                rssi_min=RSSI_MIN, rssi_max=RSSI_MAX,
            )
        except ValueError:
            errs[i] = TOTAL_SIGNAL_LOSS_PENALTY_M
            continue
        img = torch.from_numpy(generate_image(ap_info, img_hw)).unsqueeze(0)
        xy_pred, floor_pred = model(img, site_id)
        pred_xy = xy_pred.numpy()[0] * 0.5 + np.array([x_min, y_min])
        xy_err = float(np.linalg.norm(pred_xy - np.array([x, y])))
        num_floors = model.site_num_floors[site_id]
        pred_floor_class = int(round(min(max(float(floor_pred.item()), 0.0), num_floors - 1)))
        errs[i] = np.sqrt(xy_err ** 2 + (FLOOR_HEIGHT_M * abs(pred_floor_class - floor_class)) ** 2)
    return errs


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    hp = HyperParams()

    site_num_floors = {}
    for site_id in PRETRAIN_SITE_IDS:
        d, _, _ = load_site(site_id)
        site_num_floors[site_id] = len(build_floor_class_map(d))

    X, y_target, groups, site_tags = [], [], [], []

    for site_id in POOL_SITES:
        print(f"\n--- {site_id} ---")
        df, wap_pos, wap_cols = load_site(site_id)
        floor_map = build_floor_class_map(df)
        df = attach_floor_class(df, floor_map)
        df = filter_locatable_rows(df, wap_pos, wap_cols, hp.k_strongest, site_id)
        _, val_df, test_df = stratified_grouped_split_by_floor(df, "_group", "floor_id",
                                                                 val_frac=0.15, test_frac=0.15, seed=SEED)
        held_out_df = pd.concat([val_df, test_df], ignore_index=True)
        print(f"[{site_id}] {held_out_df['_group'].nunique()} points held out from training.")

        model = MultiSiteGlobLocCNN(site_num_floors=site_num_floors)
        model.load_state_dict(torch.load(CHECKPOINT_DIR / f"model_{site_id}.pt", map_location="cpu"))
        model.eval()

        by_point = held_out_df.assign(_pid=held_out_df.apply(_point_id, axis=1)).groupby("_pid")
        for pid, rows in by_point:
            clean_raw = rows[wap_cols].to_numpy(dtype=np.float32)
            n = len(clean_raw)
            for _ in range(N_BOOTSTRAP):
                ref_idx = rng.integers(0, n, size=min(BUFFER_SIZE, n))
                cur_idx = rng.integers(0, n, size=min(BUFFER_SIZE, n))
                ref_buf = clean_raw[ref_idx]
                cur_buf_drifted, severity, dropout_prob = random_drift_snapshot(clean_raw[cur_idx], rng)

                stats = compute_aggregate_buffer_stats(cur_buf_drifted, ref_buf)
                cur_rows = rows.iloc[cur_idx]
                errs = per_row_combined_3d_err(model, cur_rows, wap_pos, wap_cols, hp, site_id, cur_buf_drifted)
                target = float(np.median(errs))

                X.append(stats)
                y_target.append(target)
                groups.append(f"{site_id}__{pid}")
                site_tags.append(site_id)

    X = np.stack(X)
    y_target = np.array(y_target, dtype=np.float64)
    groups = np.array(groups)
    site_tags = np.array(site_tags)
    print(f"\nBuilt {len(X)} pooled buffer instances ({X.shape[1]}-dim aggregate stats) "
          f"across {len(set(groups))} (site, point) groups spanning {len(POOL_SITES)} sites.\n")

    gkf = GroupKFold(n_splits=N_FOLDS)
    rf_grouped_pred = np.zeros_like(y_target)
    for tr_idx, te_idx in gkf.split(X, y_target, groups=groups):
        rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        rf.fit(X[tr_idx], y_target[tr_idx])
        rf_grouped_pred[te_idx] = rf.predict(X[te_idx])
    r2_rf_grouped = r2_score(y_target, rf_grouped_pred)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rf_random_pred = np.zeros_like(y_target)
    for tr_idx, te_idx in kf.split(X):
        rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
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

    # Also: leave-SITE-out (not just leave-point-out) -- the strongest
    # possible generalization test, since it can't lean on ANY of a site's
    # own points at all, only patterns learned from OTHER sites entirely.
    gkf_site = GroupKFold(n_splits=len(POOL_SITES))
    rf_site_pred = np.zeros_like(y_target)
    for tr_idx, te_idx in gkf_site.split(X, y_target, groups=site_tags):
        rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
        rf.fit(X[tr_idx], y_target[tr_idx])
        rf_site_pred[te_idx] = rf.predict(X[te_idx])
    r2_rf_leave_site_out = r2_score(y_target, rf_site_pred)

    print("=" * 70)
    print("Phase 6b pooled oracle diagnostic results")
    print("=" * 70)
    print(f"Leave-points-out RandomForest R^2:      {r2_rf_grouped:.3f}")
    print(f"Random (non-grouped) RandomForest R^2:  {r2_rf_random:.3f}")
    print(f"Leave-points-out Ridge (linear) R^2:     {r2_ridge_grouped:.3f}")
    print(f"Point-identity ALONE R^2 (oracle):       {r2_identity:.3f}")
    print(f"Leave-SITE-out RandomForest R^2:         {r2_rf_leave_site_out:.3f}  "
          f"(strongest generalization test -- predicts entirely unseen sites)")
    print()

    if r2_identity > max(r2_rf_grouped, r2_ridge_grouped):
        print("*** WARNING: point identity alone predicts more error variance than "
              "the drift-buffer statistics do. Confound warning sign, NOT a green "
              "light for hypernetwork training. ***")
    elif max(r2_rf_grouped, r2_ridge_grouped) > 0.1 and r2_rf_leave_site_out > 0.0:
        print("Real, non-confounded, cross-site signal found. Proceeding to "
              "hypernetwork training is defensible.")
    else:
        print("Signal is weak/inconclusive; treat as a negative or marginal "
              "result, not a green light.")

    import json
    results = {
        "n_instances": len(X), "n_groups": len(set(groups)), "n_sites": len(POOL_SITES),
        "stat_dim": int(X.shape[1]),
        "r2_rf_leave_points_out": r2_rf_grouped,
        "r2_rf_random_cv": r2_rf_random,
        "r2_ridge_leave_points_out": r2_ridge_grouped,
        "r2_point_identity_oracle": r2_identity,
        "r2_rf_leave_site_out": r2_rf_leave_site_out,
    }
    with open(OUTDIR / "oracle_diagnostic_pooled.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUTDIR / 'oracle_diagnostic_pooled.json'}")


if __name__ == "__main__":
    main()
