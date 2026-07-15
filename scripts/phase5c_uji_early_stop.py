"""Continue uji_b1/uji_b2 with early stopping instead of a fixed epoch
count, starting from the phase5c_uji_continue checkpoints (which were
already +15 epochs past phase5c_v2):
  - uji_b2: same LR (3e-5) -- its val curve was climbing cleanly, no
    reason to slow it down.
  - uji_b1: LOWER LR (1e-5) -- its val curve got noisy rather than
    converging under the standard LR, so backing off should let it settle
    into genuine improvement instead of oscillating.
Early stopping: patience=6 epochs, min_delta=0.005m on the combined 3D
validation metric, max_epochs=40 safety cap.
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import HyperParams  # noqa: E402
from naive_baseline import naive_constant_baseline  # noqa: E402
from phase5c_finetune import finetune_one_site_early_stop  # noqa: E402
from sites import (  # noqa: E402
    PRETRAIN_SITE_IDS, attach_floor_class, build_floor_class_map,
    filter_locatable_rows, load_site,
)
from splits import grouped_split_by_column  # noqa: E402

PRIOR_DIR = Path(__file__).resolve().parents[1] / "outputs" / "phase5c_uji_continue"
OUTDIR = Path(__file__).resolve().parents[1] / "outputs" / "phase5c_uji_early_stop"
SEED = 42

SITE_CONFIG = {
    "uji_b1": {"lr": 1e-5},
    "uji_b2": {"lr": 3e-5},
}


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

    with open(PRIOR_DIR / "results.json") as f:
        prior_results = json.load(f)

    summary = {}
    for site_id, cfg in SITE_CONFIG.items():
        print(f"\n{'=' * 78}\nEarly-stopping fine-tune: {site_id} (lr={cfg['lr']:.0e})\n{'=' * 78}")
        train_df, val_df, test_df, wap_pos, wap_cols = site_data[site_id]
        naive = naive_baseline_generic(train_df, test_df)
        prior = prior_results[site_id]["continued"]
        print(f"[{site_id}] starting from phase5c_uji_continue checkpoint: xy={prior['median_xy_err_m']:.2f}m "
              f"floor_acc={prior['floor_accuracy']:.3f}")

        starting_state = torch.load(PRIOR_DIR / f"model_{site_id}.pt", map_location=device)

        t0 = time.time()
        model, test_m, history, best_epoch = finetune_one_site_early_stop(
            starting_state, site_num_floors, site_id, train_df, val_df, test_df,
            wap_pos, wap_cols, hp, device=device, lr=cfg["lr"],
        )
        dt = time.time() - t0

        print(f"[{site_id}] converged at epoch {best_epoch}/{len(history)}: "
              f"xy={test_m['median_xy_err_m']:.2f}m floor_acc={test_m['floor_accuracy']:.3f}  ({dt:.0f}s)")
        print(f"[{site_id}] naive: xy={naive['median_xy_err_m']:.2f}m floor_acc={naive['floor_accuracy']:.3f}")

        summary[site_id] = {"prior": prior, "final": test_m, "naive": naive,
                             "best_epoch": best_epoch, "epochs_run": len(history), "history": history}
        torch.save(model.state_dict(), OUTDIR / f"model_{site_id}.pt")

    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    for site_id, s in summary.items():
        print(f"{site_id:10s} | prior floor_acc={s['prior']['floor_accuracy']:.3f} -> "
              f"final floor_acc={s['final']['floor_accuracy']:.3f} "
              f"(converged epoch {s['best_epoch']}/{s['epochs_run']})")

    with open(OUTDIR / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
