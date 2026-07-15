"""Generic zero-shot evaluation: load a joint checkpoint, run it (frozen,
no floor head available) on a site it was NEVER trained on. Reused for
both held-out sites: uji_b0 (same dataset family as uji_b1/uji_b2, in
training) and tampere (a genuinely different dataset/country/methodology,
held out specifically to test generalization beyond one dataset family --
see PIVOT_PLAN.md "second zero-shot site").
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import HyperParams  # noqa: E402
from multisite_dataset import MultiSiteDataset  # noqa: E402
from multisite_model import MultiSiteGlobLocCNN  # noqa: E402
from multisite_train import run_epoch_multisite  # noqa: E402
from naive_baseline import naive_constant_baseline  # noqa: E402
from sites import (  # noqa: E402
    PRETRAIN_SITE_IDS, attach_floor_class, build_floor_class_map,
    filter_locatable_rows, load_site,
)

HP_OVERRIDES = dict(margin_m=2.0, max_extent_m=10.0)  # must match the checkpoint's training hp
SEED = 42


def naive_baseline_generic(train_df, eval_df):
    tr = train_df.rename(columns={"floor_id": "floor"})
    ev = eval_df.rename(columns={"floor_id": "floor"})
    return naive_constant_baseline(tr, ev)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="e.g. outputs/phase5b_v4")
    parser.add_argument("--site_id", required=True, help="held-out site to zero-shot eval, e.g. tampere")
    args = parser.parse_args()

    checkpoint_dir = Path(__file__).resolve().parents[1] / args.checkpoint_dir
    hp = HyperParams(**HP_OVERRIDES)
    device = "cpu"

    # site_num_floors is needed to reconstruct the model skeleton (its
    # floor_heads ModuleDict), even though the held-out site's floor won't
    # be evaluated -- must match exactly what the checkpoint was trained with.
    site_num_floors = {}
    for site_id in PRETRAIN_SITE_IDS:
        df, wap_pos, wap_cols = load_site(site_id)
        floor_map = build_floor_class_map(df)
        site_num_floors[site_id] = len(floor_map)

    model = MultiSiteGlobLocCNN(site_num_floors=site_num_floors)
    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))

    print(f"\n{'=' * 78}\nZero-shot evaluation: {args.site_id} (NEVER trained on)\n{'=' * 78}")
    ho_df, ho_wap_pos, ho_wap_cols = load_site(args.site_id)
    ho_floor_map = build_floor_class_map(ho_df)
    ho_df = attach_floor_class(ho_df, ho_floor_map)
    ho_df = filter_locatable_rows(ho_df, ho_wap_pos, ho_wap_cols, hp.k_strongest, args.site_id)
    ho_naive = naive_baseline_generic(ho_df, ho_df)  # in-site reference only

    ho_ds = MultiSiteDataset({args.site_id: ho_df}, {args.site_id: (ho_wap_pos, ho_wap_cols)}, hp, augment=False, seed=SEED)
    _, ho_per_site = run_epoch_multisite(model, ho_ds, hp, optimizer=None, device=device)
    ho_metrics = ho_per_site[args.site_id]
    print(f"[{args.site_id}] zero-shot model xy={ho_metrics['median_xy_err_m']:.2f}m "
          f"(floor: N/A, no head for an unseen site)   ||   in-site naive xy={ho_naive['median_xy_err_m']:.2f}m")
    beats_naive = ho_metrics["median_xy_err_m"] < ho_naive["median_xy_err_m"]
    print(f"Zero-shot backbone beats in-site naive baseline: {beats_naive}")
    if not beats_naive:
        print("*** WARNING: zero-shot xy error does NOT beat the naive baseline. ***")

    outpath = checkpoint_dir / f"zeroshot_{args.site_id}.json"
    with open(outpath, "w") as f:
        json.dump({"model_xy_err_m": ho_metrics["median_xy_err_m"],
                   "naive_xy_err_m": ho_naive["median_xy_err_m"],
                   "beats_naive": bool(beats_naive), "n": ho_metrics["n"]}, f, indent=2)
    print(f"Saved to {outpath}")


if __name__ == "__main__":
    main()
