"""Phase 6d: genuinely new signal source, not a variant of 6/6b/6c.

Phases 6, 6b, 6c all used hand-crafted RSSI summary statistics (detect-rate
diffs, mean-signal diffs) as the buffer signature fed to the oracle
regressor. All three failed to show a real, non-confounded, cross-site
signal.

This trial replaces the feature source entirely: instead of statistics over
raw RSSI, use the FROZEN BACKBONE'S OWN LEARNED 128-d BOTTLENECK EMBEDDING
(MultiSiteGlobLocCNN.features()) for every buffer instance, and summarize
drift as embedding-space displacement between the reference and current
buffer's mean embedding. Rationale: the backbone already learned what's
relevant to localization from the RSSI pattern; feature-space drift is a
more direct proxy for "how is this affecting the thing we're trying to
correct" than generic RSSI statistics, and it is dimension-independent by
construction (always 128-d, regardless of a site's raw AP count), so it
can't leak site identity through AP-count the way Phase 6b's raw stats did.

Reuses Phase 6b's buffer-generation loop and drift injector verbatim; only
the feature vector changes. Treated as the final trial per the standing
one-more-bounded-trial agreement (see PIVOT_PLAN.md) -- reported regardless
of outcome.
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
    RSSI_MAX, RSSI_MIN, PRETRAIN_SITE_IDS,
    attach_floor_class, build_floor_class_map, filter_locatable_rows, load_site,
)
from splits import stratified_grouped_split_by_floor  # noqa: E402
from virtual_space import build_virtual_space, generate_image  # noqa: E402
from phase6_oracle_diagnostic import random_drift_snapshot, TOTAL_SIGNAL_LOSS_PENALTY_M  # noqa: E402
from phase6b_oracle_diagnostic_pooled import per_row_combined_3d_err, POOL_SITES  # noqa: E402

torch.set_num_threads(1)

OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "phase6d"
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase5c_v4"
BUFFER_SIZE = 10
N_BOOTSTRAP = 6
N_FOLDS = 5
SEED = 42
EMBED_DIM = 128


def _point_id(row):
    return f"{row['floor_id']}_{row['x']}_{row['y']}"


@torch.no_grad()
def buffer_embeddings(model, rows_df, wap_pos, wap_cols, hp, site_id, rssi_matrix):
    """Per-instance 128-d bottleneck embeddings for one buffer. Rows whose
    RSSI has no positioned AP at all (build_virtual_space raises, same
    total-signal-loss edge case as Phase 6/6b's error computation) get a
    zero embedding rather than being dropped -- dropping would silently
    shrink the buffer and bias the mean toward "easy" instances."""
    embeds = np.zeros((len(rows_df), EMBED_DIM), dtype=np.float32)
    for i, (_, row) in enumerate(rows_df.iterrows()):
        x, y = float(row["x"]), float(row["y"])
        rssi_row = rssi_matrix[i]
        try:
            ap_info, (vx, vy), (x_min, y_min), img_hw = build_virtual_space(
                x, y, rssi_row, wap_cols, wap_pos, hp.k_strongest, hp.margin_m, hp.max_extent_m,
                rssi_min=RSSI_MIN, rssi_max=RSSI_MAX,
            )
        except ValueError:
            continue
        img = torch.from_numpy(generate_image(ap_info, img_hw)).unsqueeze(0)
        embeds[i] = model.features(img).numpy()[0]
    return embeds


def compute_embedding_drift_stats(cur_embeds: np.ndarray, ref_embeds: np.ndarray) -> np.ndarray:
    """Fixed-size (8-dim) summary of embedding-space displacement between a
    current (possibly drifted) buffer and its reference buffer -- mirrors
    Phase 6c's 8-dim RSSI-stats vector in shape, so RF/Ridge capacity is
    matched across trials, but every input is a learned-feature quantity,
    never a raw RSSI or AP-count statistic."""
    cur_mean = cur_embeds.mean(axis=0)
    ref_mean = ref_embeds.mean(axis=0)
    diff = cur_mean - ref_mean

    l2_diff = float(np.linalg.norm(diff))
    cur_norm = float(np.linalg.norm(cur_mean))
    ref_norm = float(np.linalg.norm(ref_mean))
    denom = (cur_norm * ref_norm) if cur_norm > 1e-8 and ref_norm > 1e-8 else 1.0
    cosine_dist = 1.0 - float(np.dot(cur_mean, ref_mean) / denom)

    return np.array([
        l2_diff,
        np.abs(diff).mean(), np.abs(diff).max(), diff.std(),
        cur_embeds.std(axis=0).mean(),  # within-buffer instability, current
        ref_embeds.std(axis=0).mean(),  # within-buffer instability, reference (baseline)
        cosine_dist,
        cur_norm,
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
                ref_rows = rows.iloc[ref_idx]
                cur_rows = rows.iloc[cur_idx]
                ref_buf = clean_raw[ref_idx]
                cur_buf_drifted, severity, dropout_prob = random_drift_snapshot(clean_raw[cur_idx], rng)

                ref_embeds = buffer_embeddings(model, ref_rows, wap_pos, wap_cols, hp, site_id, ref_buf)
                cur_embeds = buffer_embeddings(model, cur_rows, wap_pos, wap_cols, hp, site_id, cur_buf_drifted)
                stats = compute_embedding_drift_stats(cur_embeds, ref_embeds)

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

    # Per-site target normalization (Phase 6c's fix, kept): removes
    # between-site absolute-difficulty variance so "predict the mean" isn't
    # an easy baseline and the regressor must use the actual drift signal.
    y_norm = y_raw.copy()
    site_medians = {}
    for s in set(site_tags):
        mask = site_tags == s
        med = np.median(y_raw[mask])
        site_medians[s] = med
        y_norm[mask] = y_raw[mask] - med
    print(f"\nPer-site median raw target (subtracted before fitting): {site_medians}")
    print(f"Built {len(X)} instances, {X.shape[1]}-dim embedding-drift stats, "
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
                    r2_point_identity_oracle=r2_identity, r2_rf_leave_site_out=r2_leave_site_out,
                    rf_leave_site_out_pred=rf_site_pred, rf_leave_points_out_pred=rf_grouped_pred)

    print("=" * 70)
    print("Phase 6d: embedding-space drift signal (frozen backbone features)")
    print("=" * 70)
    results_norm = run_all(y_norm, "normalized target (site median subtracted)")
    print()
    results_raw = run_all(y_raw, "raw target (unnormalized)")

    print()
    best_signal = max(results_norm["r2_rf_leave_points_out"], results_norm["r2_ridge_leave_points_out"])
    if results_norm["r2_point_identity_oracle"] > best_signal or results_norm["r2_rf_leave_site_out"] <= 0:
        print("*** Still a confound / no cross-site signal from embedding-space drift either. "
              "Final verdict: negative. ***")
    else:
        print("Real, non-confounded, cross-site signal found from embedding-space drift. "
              "Proceeding to hypernetwork training is defensible.")

    np.savez(OUTDIR / "raw_arrays.npz", X=X, y_raw=y_raw, y_norm=y_norm, groups=groups,
              site_tags=site_tags, rf_leave_site_out_pred_norm=results_norm["rf_leave_site_out_pred"],
              rf_leave_points_out_pred_norm=results_norm["rf_leave_points_out_pred"])
    print(f"Saved raw arrays to {OUTDIR / 'raw_arrays.npz'}")

    import json
    results_norm_json = {k: v for k, v in results_norm.items() if not k.startswith("rf_")}
    results_raw_json = {k: v for k, v in results_raw.items() if not k.startswith("rf_")}
    with open(OUTDIR / "oracle_diagnostic_embedding.json", "w") as f:
        json.dump({"normalized": results_norm_json, "raw": results_raw_json,
                    "site_medians": site_medians}, f, indent=2)
    print(f"Saved to {OUTDIR / 'oracle_diagnostic_embedding.json'}")


if __name__ == "__main__":
    main()
