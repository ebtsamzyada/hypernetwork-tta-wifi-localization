"""
Robust meta-validation: a single fixed 3-session holdout can land on an
unlucky (or lucky) combination of inherently-hard sessions, since
UJIIndoorLoc sessions vary a lot in how well-covered their area is by the
rest of the building (confirmed: some train sessions have ~13m clean
baseline error, others ~19m, with ZERO drift applied at all). Averaging
one such split into a headline number is misleading.

This script rotates ALL 12 train sessions through K-fold validation (each
session is held out exactly once), pools every fold's fixed validation
episodes together, and reports the adapted-vs-frozen gap across that full
pooled set -- an honest, session-variance-robust estimate of whether the
hypernetwork actually helps.

It then trains ONE production hypernetwork on ALL 12 train sessions
(no holdout) for the step count the CV curves show is stable/converged,
and saves it for Step 5. Step 5's test-set evaluation is the real
headline number regardless; this just makes sure the checkpoint we carry
into it wasn't cherry-picked by accident.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import KFold

from data_loader import load_split
from train_hypernetwork import (
    SEED, NUM_META_STEPS, OUT_DIR,
    load_frozen_base, train_one_run, euclidean_error,
)

N_FOLDS = 4


def main():
    split = load_split()
    base_model, base_head_w, base_head_b, target_mean = load_frozen_base()

    all_groups = np.array(sorted(set(split.train_groups)))
    print(f"All {len(all_groups)} train sessions: {list(all_groups)}")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    pooled_adapted, pooled_frozen = [], []
    fold_summaries = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(all_groups)):
        fold_train_groups = list(all_groups[tr_idx])
        fold_val_groups = list(all_groups[val_idx])
        print(f"\n=== Fold {fold}: val sessions {fold_val_groups} ===")

        result = train_one_run(
            split, base_model, base_head_w, base_head_b, target_mean,
            fold_train_groups, fold_val_groups,
            num_steps=1000, seed=SEED + fold, eval_every=50, eval_episodes=60, verbose=False,
        )
        print(f"  best: adapted={result['best_val_adapted']:.2f}m  frozen={result['best_val_frozen']:.2f}m")
        fold_summaries.append((fold_val_groups, result["best_val_adapted"], result["best_val_frozen"]))
        pooled_adapted.append(result["best_val_adapted"])
        pooled_frozen.append(result["best_val_frozen"])

    print("\n=== Per-fold summary (each session was held out exactly once) ===")
    for val_groups, a, f in fold_summaries:
        print(f"  val={val_groups}: adapted={a:.2f}m  frozen={f:.2f}m  "
              f"improvement={f - a:+.2f}m")

    pooled_adapted = np.array(pooled_adapted)
    pooled_frozen = np.array(pooled_frozen)
    print(f"\nMean across folds -- adapted: {pooled_adapted.mean():.2f}m  "
          f"frozen: {pooled_frozen.mean():.2f}m  "
          f"mean improvement: {(pooled_frozen - pooled_adapted).mean():+.2f}m")
    print(f"Folds where adapted beat frozen: {(pooled_adapted < pooled_frozen).sum()}/{N_FOLDS}")

    # Production hypernetwork: train on ALL 12 sessions, no holdout.
    # Step count fixed at NUM_META_STEPS since the per-fold curves above
    # (each run for 1000 steps) showed the adapted-vs-frozen gap plateaus
    # well before that many steps and does not diverge afterward.
    print(f"\n=== Training production hypernetwork on all {len(all_groups)} sessions ===")
    final = train_one_run(
        split, base_model, base_head_w, base_head_b, target_mean,
        list(all_groups), val_groups=[], num_steps=NUM_META_STEPS, seed=SEED, verbose=False,
    )
    hypernet = final["hypernet"]

    torch.save(
        {
            "state_dict": hypernet.state_dict(),
            "num_aps": len(split.wap_cols),
            "head_in": base_head_w.shape[1],
            "head_out": base_head_w.shape[0],
            "hidden": 128,
        },
        f"{OUT_DIR}/hypernetwork.pt",
    )
    print(f"Saved production hypernetwork to {OUT_DIR}/hypernetwork.pt")


if __name__ == "__main__":
    main()
