"""Continue fine-tuning uji_b1/uji_b2 specifically -- Phase 5c v2 showed
floor accuracy still climbing at epoch 10 for both, not yet converged
(uji_b2: ...44.9, 48.5, 48.8 -- still rising at the last epoch). Other 4
sites are excluded here: sod_hcxy/sod_syl are trivial (single floor,
already at 100%), sod_cetc331 was flat across all 10 epochs (94.1-94.9%),
and hdlc was oscillating around 77-82% rather than climbing -- none of
those show the same clear still-improving signal UJI does.

Starts from the ALREADY-FINE-TUNED phase5c_v2 checkpoints (not the v2
joint backbone), reusing finetune_one_site unmodified -- it doesn't care
whether the starting state dict is the joint backbone or a previously
fine-tuned one, it just continues training from wherever it's given.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import HyperParams  # noqa: E402
from multisite_model import MultiSiteGlobLocCNN  # noqa: E402
from naive_baseline import naive_constant_baseline  # noqa: E402
from phase5c_finetune import finetune_one_site  # noqa: E402
from sites import (  # noqa: E402
    PRETRAIN_SITE_IDS, attach_floor_class, build_floor_class_map,
    filter_locatable_rows, load_site,
)
from splits import grouped_split_by_column  # noqa: E402

PHASE5C_V2_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase5c_v2"
OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "phase5c_uji_continue"
CONTINUE_SITES = ["uji_b1", "uji_b2"]
MORE_EPOCHS = 15
SEED = 42

import phase5c_finetune as ft  # noqa: E402
ft.FINETUNE_EPOCHS = MORE_EPOCHS


def naive_baseline_generic(train_df, eval_df):
    tr = train_df.rename(columns={"floor_id": "floor"})
    ev = eval_df.rename(columns={"floor_id": "floor"})
    return naive_constant_baseline(tr, ev)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hp = HyperParams()

    print("Re-deriving splits/site_num_floors for all 6 pretraining sites...")
    site_data = {}
    site_num_floors = {}
    for site_id in PRETRAIN_SITE_IDS:
        df, wap_pos, wap_cols = load_site(site_id)
        floor_map = build_floor_class_map(df)
        df = attach_floor_class(df, floor_map)
        df = filter_locatable_rows(df, wap_pos, wap_cols, hp.k_strongest, site_id)
        train_df, val_df, test_df = grouped_split_by_column(df, "_group", val_frac=0.15, test_frac=0.15, seed=SEED)
        site_data[site_id] = (train_df, val_df, test_df, wap_pos, wap_cols)
        site_num_floors[site_id] = len(floor_map)

    with open(PHASE5C_V2_DIR / "results.json") as f:
        v2_finetune_results = json.load(f)

    summary = {}
    for site_id in CONTINUE_SITES:
        print(f"\n{'=' * 78}\nContinuing fine-tune: {site_id} (+{MORE_EPOCHS} more epochs)\n{'=' * 78}")
        train_df, val_df, test_df, wap_pos, wap_cols = site_data[site_id]
        naive = naive_baseline_generic(train_df, test_df)
        prior = v2_finetune_results[site_id]["v3_finetuned"]
        print(f"[{site_id}] starting from phase5c_v2 checkpoint: xy={prior['median_xy_err_m']:.2f}m "
              f"floor_acc={prior['floor_accuracy']:.3f}")

        starting_state = torch.load(PHASE5C_V2_DIR / f"model_{site_id}.pt", map_location=device)

        t0 = time.time()
        model, test_m = finetune_one_site(starting_state, site_num_floors, site_id, train_df, val_df, test_df,
                                           wap_pos, wap_cols, hp, device=device)
        dt = time.time() - t0

        print(f"[{site_id}] after +{MORE_EPOCHS} epochs: xy={test_m['median_xy_err_m']:.2f}m "
              f"floor_acc={test_m['floor_accuracy']:.3f}  ({dt:.0f}s)")
        print(f"[{site_id}] naive: xy={naive['median_xy_err_m']:.2f}m floor_acc={naive['floor_accuracy']:.3f}")

        summary[site_id] = {"phase5c_v2": prior, "continued": test_m, "naive": naive}
        torch.save(model.state_dict(), OUTDIR / f"model_{site_id}.pt")

    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    for site_id, s in summary.items():
        print(f"{site_id:10s} | phase5c_v2 floor_acc={s['phase5c_v2']['floor_accuracy']:.3f} -> "
              f"continued floor_acc={s['continued']['floor_accuracy']:.3f}")

    with open(OUTDIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
