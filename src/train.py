import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import HyperParams, collate_single
from virtual_space import PIXEL_METERS

# Single-threaded is faster here: these are tiny (~1x20x40px) images, and
# torch's multi-threaded dispatch overhead per op dominates over the actual
# compute at this scale. Measured ~3.5x speedup on this workload.
torch.set_num_threads(1)

FLOOR_HEIGHT_M = 3.5  # validated against the source paper -- see PIVOT_PLAN.md


def run_epoch(model, dataset, hp: HyperParams, optimizer=None, device="cpu"):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=train_mode,
                         collate_fn=collate_single, num_workers=0)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    total_loss, n_seen = 0.0, 0
    correct_floor = 0
    records = []
    step = 0
    if train_mode:
        optimizer.zero_grad()

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            sample = batch[0]
            img = sample["image"].unsqueeze(0).to(device)
            target_xy = sample["target_xy"].unsqueeze(0).to(device)
            floor_class = sample["target_floor_class"]

            xy_pred, floor_logits = model(img)
            loss_xy = mse(xy_pred, target_xy)
            target_floor_t = torch.tensor([floor_class], dtype=torch.long, device=device)
            loss_floor = ce(floor_logits, target_floor_t)
            loss = loss_xy + hp.lambda_floor * loss_floor

            if train_mode:
                (loss / hp.pseudo_batch).backward()
                step += 1
                if step % hp.pseudo_batch == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()
            n_seen += 1

            origin = sample["origin"].numpy()
            true_xy = sample["physical_xy"].numpy()
            pred_xy = xy_pred.detach().cpu().numpy()[0] * PIXEL_METERS + origin
            xy_err = float(np.linalg.norm(pred_xy - true_xy))

            pred_floor_class = int(torch.argmax(floor_logits, dim=1).item())
            floor_ok = pred_floor_class == floor_class
            correct_floor += int(floor_ok)

            records.append({
                "xy_err_m": xy_err,
                "true_floor_class": floor_class,
                "pred_floor_class": pred_floor_class,
                "floor_correct": floor_ok,
            })

        if train_mode and step % hp.pseudo_batch != 0:
            optimizer.step()
            optimizer.zero_grad()

    avg_loss = total_loss / max(1, n_seen)
    floor_acc = correct_floor / max(1, n_seen)
    all_xy = [r["xy_err_m"] for r in records]
    combined_3d = [
        float(np.sqrt(r["xy_err_m"] ** 2 +
                       (FLOOR_HEIGHT_M * abs(r["pred_floor_class"] - r["true_floor_class"])) ** 2))
        for r in records
    ]

    metrics = {
        "avg_loss": avg_loss,
        "floor_accuracy": floor_acc,
        "median_xy_err_m": float(np.median(all_xy)) if all_xy else float("nan"),
        "median_3d_err_m": float(np.median(combined_3d)) if combined_3d else float("nan"),
    }
    return metrics, records


def train_model(model, train_ds, val_ds, hp: HyperParams, epochs, device="cpu", verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate)
    history = []
    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        train_metrics, _ = run_epoch(model, train_ds, hp, optimizer, device)
        val_metrics, _ = run_epoch(model, val_ds, hp, optimizer=None, device=device)
        history.append({"epoch": ep, "train": train_metrics, "val": val_metrics})

        if verbose:
            print(
                f"  epoch {ep:3d}/{epochs} | train_loss {train_metrics['avg_loss']:.4f} "
                f"train_floor_acc {train_metrics['floor_accuracy']:.3f} | "
                f"val_loss {val_metrics['avg_loss']:.4f} "
                f"val_floor_acc {val_metrics['floor_accuracy']:.3f} "
                f"val_3d_err {val_metrics['median_3d_err_m']:.2f}m"
            )

        if val_metrics["median_3d_err_m"] < best_val:
            best_val = val_metrics["median_3d_err_m"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val
