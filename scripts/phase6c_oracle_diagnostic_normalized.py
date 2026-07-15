"""Phase 6c: last trial. Two principled (not cosmetic) fixes on top of
Phase 6b's pooled diagnostic:
  1. Drop the AP-count feature -- a direct, literal site-identity leak
     (each site has a fixed, distinct AP count).
  2. Normalize the target PER SITE (subtract that site's own median
     per-buffer error) instead of using raw absolute error. This is the
     conceptually correct framing for what a hypernetwork TTA mechanism
     actually needs: predicting drift-induced DEGRADATION RELATIVE TO a
     site's own normal condition, not absolute difficulty (which is
     inherently confounded with which site a buffer is from). It also
     makes "predict the mean" a harder baseline, since between-site
     variance is removed before any comparison happens.

Reuses Phase 6b's buffer-generation loop verbatim; only the feature
vector and target are changed. Treated as the final trial regardless of
outcome -- see PIVOT_PLAN.md's discipline against open-ended tuning.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import HyperParams  # noqa: E402
from multisite_model import MultiSiteGlobLocCNN  # noqa: E402
from sites import (  # noqa: E402
    PRETRAIN_SITE_IDS, attach_floor_class, build_floor_class_map,
    filter_locatable_rows, load_site,
)
from splits import stratified_grouped_split_by_floor  # noqa: E402
from phase6_oracle_diagnostic import random_drift_snapshot  # noqa: E402
from phase6b_oracle_diagnostic_pooled import per_row_combined_3d_err, POOL_SITES  # noqa: E402
from sites import RSSI_MIN, RSSI_MAX  # noqa: E402

torch.set_num_threads(1)

OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "phase6c"
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase5c_v4"
BUFFER_SIZE = 10
N_BOOTSTRAP = 6
N_FOLDS = 5
SEED = 42


def _point_id(row):
    return f"{row['floor_id']}_{row['x']}_{row['y']}"


def compute_stats_no_apcount(cur_raw: np.ndarray, ref_raw: np.ndarray) -> np.ndarray:
    """Same as Phase 6b's compute_aggregate_buffer_stats, MINUS the
    AP-count feature (the direct site-identity leak)."""
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
    ], dtype=np.float32)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    hp = HyperParams()

    site_num_floors = {}
    for site_id in PRETRAIN_SITE_IDS:
        d, _, _ = load_site(site_id)
        site_num_floors[site_id] = len(build_floor_class_map(d))

    X, y_raw, groups, site_tags = [], [], [], []

    for site_id in POOL_SITES:
        print(f"\n--- {site_id} ---")
        df, wap_pos, wap_cols = load_site(site_id)
        floor_map = build_floor_class_map(df)
        df = attach_floor_class(df, floor_map)
        df = filter_locatable_rows(df, wap_pos, wap_cols, hp.k_strongest, site_id)
        _, val_df, test_df = stratified_grouped_split_by_floor(df, "_group", "floor_id",
                                                                 val_frac=0.15, test_frac=0.15, seed=SEED)
        held_out_df = pd.concat([val_df, test_df], ignore_index=True)

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

                stats = compute_stats_no_apcount(cur_buf_drifted, ref_buf)
                cur_rows = rows.iloc[cur_idx]
                errs = per_row_combined_3d_err(model, cur_rows, wap_pos, wap_cols, hp, site_id, cur_buf_drifted)
                target = float(np.median(errs))

                X.append(stats)
                y_raw.append(target)
                groups.append(f"{site_id}__{pid}")
                site_tags.append(site_id)

    X = np.stack(X)
    y_raw = np.array(y_raw, dtype=np.float64)
    groups = np.array(groups)
    site_tags = np.array(site_tags)

    # Per-site target normalization: subtract each site's own median error.
    y_norm = y_raw.copy()
    site_medians = {}
    for s in set(site_tags):
        mask = site_tags == s
        med = np.median(y_raw[mask])
        site_medians[s] = med
        y_norm[mask] = y_raw[mask] - med
    print(f"\nPer-site median raw target (subtracted before fitting): {site_medians}")
    print(f"Built {len(X)} instances, {X.shape[1]}-dim stats (AP-count dropped), "
          f"{len(set(groups))} (site,point) groups across {len(POOL_SITES)} sites.\n")

    def run_all(y, label):
        gkf = GroupKFold(n_splits=N_FOLDS)
        rf_grouped_pred = np.zeros_like(y)
        for tr_idx, te_idx in gkf.split(X, y, groups=groups):
            rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
            rf.fit(X[tr_idx], y[tr_idx])
            rf_grouped_pred[te_idx] = rf.predict(X[te_idx])
        r2_rf_grouped = r2_score(y, rf_grouped_pred)

        ridge_grouped_pred = np.zeros_like(y)
        for tr_idx, te_idx in gkf.split(X, y, groups=groups):
            ridge = Ridge(alpha=1.0, random_state=SEED)
            ridge.fit(X[tr_idx], y[tr_idx])
            ridge_grouped_pred[te_idx] = ridge.predict(X[te_idx])
        r2_ridge_grouped = r2_score(y, ridge_grouped_pred)

        point_mean = pd.Series(y).groupby(groups).transform("mean").to_numpy()
        r2_identity = r2_score(y, point_mean)

        gkf_site = GroupKFold(n_splits=len(POOL_SITES))
        rf_site_pred = np.zeros_like(y)
        for tr_idx, te_idx in gkf_site.split(X, y, groups=site_tags):
            rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
            rf.fit(X[tr_idx], y[tr_idx])
            rf_site_pred[te_idx] = rf.predict(X[te_idx])
        r2_leave_site_out = r2_score(y, rf_site_pred)

        print(f"--- {label} ---")
        print(f"  Leave-points-out RandomForest R^2: {r2_rf_grouped:.3f}")
        print(f"  Leave-points-out Ridge R^2:        {r2_ridge_grouped:.3f}")
        print(f"  Point-identity ALONE R^2:           {r2_identity:.3f}")
        print(f"  Leave-SITE-out RandomForest R^2:    {r2_leave_site_out:.3f}")
        return dict(r2_rf_leave_points_out=r2_rf_grouped, r2_ridge_leave_points_out=r2_ridge_grouped,
                    r2_point_identity_oracle=r2_identity, r2_rf_leave_site_out=r2_leave_site_out)

    print("=" * 70)
    print("Phase 6c: AP-count dropped + per-site target normalization")
    print("=" * 70)
    results_norm = run_all(y_norm, "normalized target (site median subtracted)")
    print()
    results_raw = run_all(y_raw, "raw target, AP-count dropped only (isolates that one fix)")

    print()
    best_signal = max(results_norm["r2_rf_leave_points_out"], results_norm["r2_ridge_leave_points_out"])
    if results_norm["r2_point_identity_oracle"] > best_signal or results_norm["r2_rf_leave_site_out"] <= 0:
        print("*** Still a confound / no cross-site signal after both fixes. Final verdict: negative. ***")
    else:
        print("Real, non-confounded, cross-site signal found after normalization. "
              "Proceeding to hypernetwork training is defensible.")

    import json
    with open(OUTDIR / "oracle_diagnostic_normalized.json", "w") as f:
        json.dump({"normalized": results_norm, "raw_apcount_dropped_only": results_raw,
                    "site_medians": site_medians}, f, indent=2)
    print(f"\nSaved to {OUTDIR / 'oracle_diagnostic_normalized.json'}")


if __name__ == "__main__":
    main()
