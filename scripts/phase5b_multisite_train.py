"""Phase 5b: multi-site foundation backbone. Shared trunk + xy_head, per-
site floor_heads (see src/multisite_model.py docstring for why). Trained
jointly across the 6-site pretraining pool, evaluated per-site against
each site's own naive baseline (Phase 5a), plus a genuine zero-shot xy-only
eval on uji_b0 -- never touched during training, and floor accuracy isn't
reportable there at all since there's no floor head for an unseen site
(reported as a placeholder for a future linear-probe experiment, not
silently skipped).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import HyperParams  # noqa: E402
from multisite_dataset import MultiSiteDataset, collate_single  # noqa: E402
from multisite_model import MultiSiteGlobLocCNN  # noqa: E402
from multisite_train import run_epoch_multisite, train_model_multisite  # noqa: E402
from naive_baseline import naive_constant_baseline  # noqa: E402
from sites import (  # noqa: E402
    HELDOUT_SITE_ID, PRETRAIN_SITE_IDS,
    attach_floor_class, build_floor_class_map, filter_locatable_rows, load_site,
)
from splits import grouped_split_by_column  # noqa: E402

OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "phase5b_v2"
EPOCHS = 8  # v1 was 5 epochs, plain concatenation+shuffle (outputs/phase5b,
            # kept for comparison). v2 adds site-balanced oversampling
            # (train_model_multisite's balanced=True default -- every site
            # oversampled up to the largest site's row count, ~2.2x more
            # rows/epoch) to investigate the HDLC floor-accuracy regression
            # and sod_cetc331/uji under-representation -- see PIVOT_PLAN.md.
SEED = 42


def naive_baseline_generic(train_df, eval_df):
    tr = train_df.rename(columns={"floor_id": "floor"})
    ev = eval_df.rename(columns={"floor_id": "floor"})
    return naive_constant_baseline(tr, ev)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    hp = HyperParams()

    site_train, site_val, site_test = {}, {}, {}
    site_meta = {}
    site_num_floors = {}
    naive_per_site = {}

    print("=" * 78)
    print("Loading + splitting all 6 pretraining-pool sites")
    print("=" * 78)
    for site_id in PRETRAIN_SITE_IDS:
        df, wap_pos, wap_cols = load_site(site_id)
        floor_map = build_floor_class_map(df)
        df = attach_floor_class(df, floor_map)
        df = filter_locatable_rows(df, wap_pos, wap_cols, hp.k_strongest, site_id)
        train_df, val_df, test_df = grouped_split_by_column(df, "_group", val_frac=0.15, test_frac=0.15, seed=SEED)

        site_train[site_id] = train_df
        site_val[site_id] = val_df
        site_test[site_id] = test_df
        site_meta[site_id] = (wap_pos, wap_cols)
        site_num_floors[site_id] = len(floor_map)
        naive_per_site[site_id] = naive_baseline_generic(train_df, test_df)
        print()

    train_ds = MultiSiteDataset(site_train, site_meta, hp, augment=True, seed=SEED)
    val_ds = MultiSiteDataset(site_val, site_meta, hp, augment=False, seed=SEED)
    print(f"Combined training pool: {len(train_ds)} rows across {len(site_train)} sites")
    print(f"Combined validation pool: {len(val_ds)} rows\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = MultiSiteGlobLocCNN(site_num_floors=site_num_floors).to(device)

    t0 = time.time()
    model, history, best_val = train_model_multisite(model, train_ds, val_ds, hp, epochs=EPOCHS, device=device)
    print(f"\nTraining took {time.time() - t0:.1f}s for {EPOCHS} epochs")

    print("\n" + "=" * 78)
    print("Final per-site test results vs. naive baseline")
    print("=" * 78)
    final_results = {}
    for site_id in PRETRAIN_SITE_IDS:
        test_ds = MultiSiteDataset({site_id: site_test[site_id]}, {site_id: site_meta[site_id]}, hp, augment=False, seed=SEED)
        _, per_site = run_epoch_multisite(model, test_ds, hp, optimizer=None, device=device)
        m = per_site[site_id]
        naive = naive_per_site[site_id]
        print(f"[{site_id:12s}] model xy={m['median_xy_err_m']:7.2f}m  floor_acc={m['floor_accuracy']:.3f}  "
              f"3d_err={m['median_3d_err_m']:7.2f}m   ||   naive xy={naive['median_xy_err_m']:7.2f}m  "
              f"floor_acc={naive['floor_accuracy']:.3f}")
        beat_xy = m["median_xy_err_m"] < naive["median_xy_err_m"]
        beat_floor = (np.isnan(m["floor_accuracy"]) or m["floor_accuracy"] >= naive["floor_accuracy"])
        if not beat_xy:
            print(f"    *** WARNING: [{site_id}] model does NOT beat naive xy baseline ***")
        if not beat_floor:
            print(f"    *** WARNING: [{site_id}] model does NOT beat naive floor baseline ***")
        final_results[site_id] = {"model": m, "naive": naive, "beat_xy": bool(beat_xy), "beat_floor": bool(beat_floor)}

    print("\n" + "=" * 78)
    print("Zero-shot evaluation: uji_b0 (NEVER trained on -- no floor head exists for it)")
    print("=" * 78)
    ho_df, ho_wap_pos, ho_wap_cols = load_site(HELDOUT_SITE_ID)
    ho_floor_map = build_floor_class_map(ho_df)  # only for the naive-baseline reference, not used by the model
    ho_df = attach_floor_class(ho_df, ho_floor_map)
    ho_df = filter_locatable_rows(ho_df, ho_wap_pos, ho_wap_cols, hp.k_strongest, HELDOUT_SITE_ID)
    ho_naive = naive_baseline_generic(ho_df, ho_df)  # in-site reference only, not a model comparison split

    ho_ds = MultiSiteDataset({HELDOUT_SITE_ID: ho_df}, {HELDOUT_SITE_ID: (ho_wap_pos, ho_wap_cols)}, hp, augment=False, seed=SEED)
    _, ho_per_site = run_epoch_multisite(model, ho_ds, hp, optimizer=None, device=device)
    ho_metrics = ho_per_site[HELDOUT_SITE_ID]
    print(f"[{HELDOUT_SITE_ID}] zero-shot model xy={ho_metrics['median_xy_err_m']:.2f}m "
          f"(floor: N/A, no head for an unseen site)   ||   in-site naive xy={ho_naive['median_xy_err_m']:.2f}m")
    zeroshot_beats_naive = ho_metrics["median_xy_err_m"] < ho_naive["median_xy_err_m"]
    print(f"Zero-shot backbone beats in-site naive baseline: {zeroshot_beats_naive}")
    if not zeroshot_beats_naive:
        print("*** WARNING: zero-shot xy error does NOT beat the naive baseline on the held-out site. "
              "This is the actual foundation-model test -- do not claim transfer works if this fails. ***")

    with open(OUTDIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(OUTDIR / "results.json", "w") as f:
        json.dump({
            "per_site_test": final_results,
            "zero_shot_uji_b0": {"model_xy_err_m": ho_metrics["median_xy_err_m"],
                                  "naive_xy_err_m": ho_naive["median_xy_err_m"],
                                  "beats_naive": bool(zeroshot_beats_naive)},
        }, f, indent=2)
    torch.save(model.state_dict(), OUTDIR / "model.pt")
    print(f"\nSaved outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
