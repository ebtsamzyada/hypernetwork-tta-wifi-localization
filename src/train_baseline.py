"""
Step 2: Train the frozen base network and report an honest baseline MAE.

The base network is now two frozen pieces (see base_model.FloorAwareBaseModel):
  1. A floor classifier (floor_model.py). Building 2's floors share almost
     the identical (x,y) footprint (median distance from a point to the
     nearest point on a DIFFERENT floor is ~0m), so a floor-blind model
     faces real, quantified ambiguity -- a floor-oracle check showed this
     costs ~2.5m of avoidable mean error (10.80m -> 8.25m). Floor turns
     out to be ~90% recoverable from RSSI alone, so its PREDICTED
     probabilities (not the unavailable true label) are fed to the
     localization MLP as extra input features.
  2. The localization MLP, trained on RSSI ++ predicted-floor-probs.

Evaluation metric: mean Euclidean localization error in metres, i.e.
mean(|| (x_pred, y_pred) - (x_true, y_true) ||_2), the standard metric in
indoor-localization papers. Per-axis MAE is reported too for completeness.

To keep epoch/hyperparameter selection from indirectly leaking test
information, we carve a validation set OUT OF THE TRAINING GROUPS (again
via GroupShuffleSplit, so no session leaks) and use it purely for early
stopping. The held-out test split is touched exactly once, at the end.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

from base_model import BaseMLP, FloorAwareBaseModel
from data_loader import load_split
from floor_model import train_floor_classifier

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = "../outputs"


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def euclidean_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - true, axis=1)


def main():
    set_seed()
    split = load_split()

    print("=== Training floor classifier (frozen preprocessing step) ===")
    floor_clf = train_floor_classifier(split.X_train, split.train_floor, split.train_groups, seed=SEED)
    floor_clf.eval()
    for p in floor_clf.parameters():
        p.requires_grad_(False)

    def augment(X):
        with torch.no_grad():
            probs = floor_clf.predict_proba(torch.tensor(X, dtype=torch.float32)).numpy()
        return np.concatenate([X, probs], axis=1).astype(np.float32)

    X_train_aug = augment(split.X_train)
    X_test_aug = augment(split.X_test)

    # Carve a validation set out of TRAIN groups only (early stopping),
    # test split stays untouched until final evaluation.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, val_idx = next(gss.split(X_train_aug, split.y_train, split.train_groups))
    assert set(split.train_groups[tr_idx]).isdisjoint(split.train_groups[val_idx])

    X_tr, y_tr = X_train_aug[tr_idx], split.y_train[tr_idx]
    X_val, y_val = X_train_aug[val_idx], split.y_train[val_idx]
    X_test, y_test = X_test_aug, split.y_test

    # NOTE: plain float32 .mean(axis=0) on these large-magnitude UTM
    # coordinates loses real precision (verified: can be off by >100m on
    # the Y component due to float32 accumulation). Harmless here since
    # target_mean is used consistently to center/un-center in both
    # directions (the network's bias absorbs whatever constant is chosen),
    # but computing it correctly is free and removes a latent footgun.
    target_mean = y_tr.astype(np.float64).mean(axis=0).astype(np.float32)

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=DEVICE)

    X_tr_t, y_tr_t = to_tensor(X_tr), to_tensor(y_tr - target_mean)
    X_val_t = to_tensor(X_val)
    X_test_t = to_tensor(X_test)

    print("\n=== Training localization MLP (RSSI ++ predicted floor probs) ===")
    model = BaseMLP(input_dim=X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    batch_size = 64
    n = X_tr_t.shape[0]
    best_val_err = float("inf")
    best_state = None
    patience, bad_epochs = 20, 0
    max_epochs = 300

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_tr_t[idx], y_tr_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).cpu().numpy() + target_mean
        val_err = euclidean_error(val_pred, y_val).mean()

        if val_err < best_val_err - 1e-4:
            best_val_err = val_err
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 20 == 0 or bad_epochs == 0:
            print(f"epoch {epoch:3d}  train_mse={epoch_loss:.2f}  val_err={val_err:.2f}m")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}, best val_err={best_val_err:.2f}m")
            break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).cpu().numpy() + target_mean
    test_err = euclidean_error(test_pred, y_test)
    per_axis_mae = np.abs(test_pred - y_test).mean(axis=0)

    print("\n=== Honest baseline (grouped, held-out test, floor-aware) ===")
    print(f"Test sessions held out: {sorted(set(split.test_groups))}")
    print(f"Mean Euclidean error : {test_err.mean():.2f} m")
    print(f"Median Euclidean error: {np.median(test_err):.2f} m")
    print(f"90th pct error        : {np.percentile(test_err, 90):.2f} m")
    print(f"Per-axis MAE (x, y)   : {per_axis_mae[0]:.2f} m, {per_axis_mae[1]:.2f} m")

    torch.save(
        {
            "clf_state_dict": floor_clf.state_dict(),
            "clf_input_dim": split.X_train.shape[1],
            "state_dict": model.state_dict(),
            "target_mean": target_mean,
            "input_dim": X_tr.shape[1],
            "rssi_dim": split.X_train.shape[1],
        },
        f"{OUT_DIR}/base_model.pt",
    )
    print(f"\nSaved base model (floor classifier + localization MLP) to {OUT_DIR}/base_model.pt")


if __name__ == "__main__":
    main()
