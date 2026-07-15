from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from multisite_dataset import collate_single
from virtual_space import PIXEL_METERS

torch.set_num_threads(1)

FLOOR_HEIGHT_M = 3.5  # HDLC-validated; used as a generic per-floor-class-gap
                       # penalty for the combined 3D metric on every site --
                       # not independently verified per site, treat cross-
                       # site 3D-error comparisons as approximate.


def build_site_balanced_sampler(dataset):
    """Weight each sample inversely to its site's row count, with
    num_samples = n_sites * max_site_count, so EVERY site gets oversampled
    up to the size of the largest site -- unlike naive per-epoch balancing
    (target = mean site size), this never REDUCES a large site's exposure
    relative to plain concatenation, it only boosts small sites (e.g.
    sod_cetc331, 1795 rows vs sod_hcxy's 12230) that were comparatively
    starved for shared-trunk gradient signal before. See PIVOT_PLAN.md
    "Phase 5b regression investigation"."""
    site_counts = Counter(site_id for site_id, _ in dataset.index)
    max_count = max(site_counts.values())
    weights = [1.0 / site_counts[site_id] for site_id, _ in dataset.index]
    num_samples = len(site_counts) * max_count
    print(f"Balanced sampler: site sizes {dict(site_counts)}, "
          f"oversampling every site up to {max_count} draws/epoch "
          f"({num_samples} total draws/epoch vs {len(dataset)} rows unweighted).")
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def run_epoch_multisite(model, dataset, hp, optimizer=None, device="cpu", sampler=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    if sampler is not None:
        loader = DataLoader(dataset, batch_size=1, sampler=sampler,
                             collate_fn=collate_single, num_workers=0)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=train_mode,
                             collate_fn=collate_single, num_workers=0)
    mse = nn.MSELoss()
    # Ordinal floor loss: SmoothL1 (Huber) regression on the floor index,
    # not cross-entropy -- penalizes being off by 2 floors more than off by
    # 1, matching the adjacent-floor confusion pattern found in Phase 5d.
    # See multisite_model.py's docstring.
    floor_loss_fn = nn.SmoothL1Loss()

    records = []  # per-sample dicts, tagged with site_id
    step = 0
    if train_mode:
        optimizer.zero_grad()

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            sample = batch[0]
            site_id = sample["site_id"]
            img = sample["image"].unsqueeze(0).to(device)
            target_xy = sample["target_xy"].unsqueeze(0).to(device)
            floor_class = sample["target_floor_class"]

            xy_pred, floor_pred = model(img, site_id)
            loss_xy = mse(xy_pred, target_xy)

            if floor_pred is not None:
                target_floor_t = torch.tensor([[float(floor_class)]], device=device)
                loss_floor = floor_loss_fn(floor_pred, target_floor_t)
                loss = loss_xy + hp.lambda_floor * loss_floor
            else:
                loss = loss_xy

            if train_mode:
                (loss / hp.pseudo_batch).backward()
                step += 1
                if step % hp.pseudo_batch == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            origin = sample["origin"].numpy()
            true_xy = sample["physical_xy"].numpy()
            pred_xy = xy_pred.detach().cpu().numpy()[0] * PIXEL_METERS + origin
            xy_err = float(np.linalg.norm(pred_xy - true_xy))

            if floor_pred is not None:
                num_floors = model.site_num_floors[site_id]
                raw_floor = float(floor_pred.detach().cpu().item())
                pred_floor_class = int(round(min(max(raw_floor, 0.0), num_floors - 1)))
                floor_ok = pred_floor_class == floor_class
            else:
                pred_floor_class, floor_ok = None, None

            records.append({
                "site_id": site_id,
                "xy_err_m": xy_err,
                "true_floor_class": floor_class,
                "pred_floor_class": pred_floor_class,
                "floor_correct": floor_ok,
            })

        if train_mode and step % hp.pseudo_batch != 0:
            optimizer.step()
            optimizer.zero_grad()

    return summarize_records(records)


def summarize_records(records):
    """Returns (pooled_metrics, per_site_metrics). Combined 3D error is
    only computed for records that HAVE a floor prediction (i.e. not
    zero-shot no-floor-head records)."""
    by_site = {}
    for r in records:
        by_site.setdefault(r["site_id"], []).append(r)

    per_site = {}
    for site_id, recs in by_site.items():
        xy_errs = [r["xy_err_m"] for r in recs]
        floor_recs = [r for r in recs if r["floor_correct"] is not None]
        floor_acc = (float(np.mean([r["floor_correct"] for r in floor_recs]))
                     if floor_recs else float("nan"))
        if floor_recs:
            combined_3d = [
                float(np.sqrt(r["xy_err_m"] ** 2 +
                              (FLOOR_HEIGHT_M * abs(r["pred_floor_class"] - r["true_floor_class"])) ** 2))
                for r in floor_recs
            ]
            median_3d = float(np.median(combined_3d))
        else:
            median_3d = float("nan")
        floor_mae = (float(np.mean([abs(r["pred_floor_class"] - r["true_floor_class"]) for r in floor_recs]))
                     if floor_recs else float("nan"))
        per_site[site_id] = {
            "n": len(recs),
            "median_xy_err_m": float(np.median(xy_errs)),
            "floor_accuracy": floor_acc,
            "floor_mae": floor_mae,
            "median_3d_err_m": median_3d,
        }

    all_xy = [r["xy_err_m"] for r in records]
    floor_recs_all = [r for r in records if r["floor_correct"] is not None]
    if floor_recs_all:
        combined_3d_all = [
            float(np.sqrt(r["xy_err_m"] ** 2 +
                          (FLOOR_HEIGHT_M * abs(r["pred_floor_class"] - r["true_floor_class"])) ** 2))
            for r in floor_recs_all
        ]
        pooled_median_3d = float(np.median(combined_3d_all))
    else:
        pooled_median_3d = float("nan")
    pooled = {
        "n": len(records),
        "median_xy_err_m": float(np.median(all_xy)) if all_xy else float("nan"),
        "median_3d_err_m": pooled_median_3d,
    }
    return pooled, per_site


def train_model_multisite(model, train_ds, val_ds, hp, epochs, device="cpu", verbose=True, balanced=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    history = []
    best_val = float("inf")
    best_state = None
    train_sampler = build_site_balanced_sampler(train_ds) if balanced else None

    for ep in range(1, epochs + 1):
        train_pooled, train_per_site = run_epoch_multisite(model, train_ds, hp, optimizer, device, sampler=train_sampler)
        val_pooled, val_per_site = run_epoch_multisite(model, val_ds, hp, optimizer=None, device=device)

        history.append({"epoch": ep, "train_pooled": train_pooled, "train_per_site": train_per_site,
                         "val_pooled": val_pooled, "val_per_site": val_per_site})

        if verbose:
            print(f"  epoch {ep:2d}/{epochs} | train_median_xy {train_pooled['median_xy_err_m']:.2f}m | "
                  f"val_median_xy {val_pooled['median_xy_err_m']:.2f}m val_median_3d {val_pooled['median_3d_err_m']:.2f}m")
            for site_id, m in val_per_site.items():
                print(f"      val[{site_id:12s}] xy={m['median_xy_err_m']:7.2f}m  "
                      f"floor_acc={m['floor_accuracy']:.3f}  3d_err={m['median_3d_err_m']:7.2f}m")

        # Selection criterion: pooled median COMBINED 3D error, not xy alone
        # -- xy-only selection was found to silently pick floor-worse
        # checkpoints (HDLC's fine-tune floor accuracy peaked at epoch 2 but
        # a later, xy-better/floor-worse epoch got selected instead). See
        # PIVOT_PLAN.md "model-selection bugfix".
        val_selection_metric = val_pooled["median_3d_err_m"]
        if not np.isnan(val_selection_metric) and val_selection_metric < best_val:
            best_val = val_selection_metric
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val
