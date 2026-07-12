"""
Robust meta-validation for the CSI hypernetwork: rotates all training
POINTS (the CSI analogue of "sessions") through K-fold validation, each
point held out exactly once, pooling results for an honest,
point-variance-robust estimate -- mirrors the RSSI pipelines'
cv_hypernetwork.py exactly in spirit.

Then trains ONE production hypernetwork on ALL training points (no
holdout) for use in Step 5.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import KFold

from data_loader import load_split
from train_hypernetwork import SEED, NUM_META_STEPS, OUT_DIR, load_frozen_base, train_one_run

N_FOLDS = 5


def main():
    split = load_split()
    base_model, base_head_w, base_head_b, target_mean = load_frozen_base()

    all_points = np.array(sorted(set(split.point_train)))
    print(f"All {len(all_points)} train points")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    pooled_adapted, pooled_frozen = [], []
    fold_summaries = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(all_points)):
        fold_train_points = list(all_points[tr_idx])
        fold_val_points = list(all_points[val_idx])
        print(f"\n=== Fold {fold}: {len(fold_val_points)} val points ===")

        result = train_one_run(
            split, base_model, base_head_w, base_head_b, target_mean,
            fold_train_points, fold_val_points,
            num_steps=800, seed=SEED + fold, eval_every=40, eval_episodes=60, verbose=False,
        )
        print(f"  best: adapted={result['best_val_adapted']:.3f}m  frozen={result['best_val_frozen']:.3f}m")
        fold_summaries.append((len(fold_val_points), result["best_val_adapted"], result["best_val_frozen"]))
        pooled_adapted.append(result["best_val_adapted"])
        pooled_frozen.append(result["best_val_frozen"])

    print("\n=== Per-fold summary ===")
    for n, a, f in fold_summaries:
        print(f"  n_val={n}: adapted={a:.3f}m  frozen={f:.3f}m  improvement={f - a:+.3f}m")

    pooled_adapted = np.array(pooled_adapted)
    pooled_frozen = np.array(pooled_frozen)
    print(f"\nMean across folds -- adapted: {pooled_adapted.mean():.3f}m  "
          f"frozen: {pooled_frozen.mean():.3f}m  "
          f"mean improvement: {(pooled_frozen - pooled_adapted).mean():+.3f}m")
    print(f"Folds where adapted beat frozen: {(pooled_adapted < pooled_frozen).sum()}/{N_FOLDS}")

    print(f"\n=== Training production hypernetwork on all {len(all_points)} points ===")
    final = train_one_run(
        split, base_model, base_head_w, base_head_b, target_mean,
        list(all_points), val_points=[], num_steps=NUM_META_STEPS, seed=SEED, verbose=False,
    )
    hypernet = final["hypernet"]

    torch.save(
        {"state_dict": hypernet.state_dict(), "num_features": split.num_features,
         "head_in": base_head_w.shape[1], "head_out": base_head_w.shape[0], "hidden": 64},
        f"{OUT_DIR}/hypernetwork.pt",
    )
    print(f"Saved production hypernetwork to {OUT_DIR}/hypernetwork.pt")


if __name__ == "__main__":
    main()
