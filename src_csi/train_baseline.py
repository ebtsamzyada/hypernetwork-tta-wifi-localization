"""
Step 2 (CSI variant): Train the frozen base MLP and report an honest
baseline MAE, on the qiang5love1314 Lab Dataset (single Tx-Rx link CSI).

Evaluation metric: mean Euclidean localization error in metres.

To keep epoch selection from indirectly leaking test information, a
validation set is carved OUT OF THE TRAINING POINTS (GroupShuffleSplit on
reference-point ID, same discipline as every other split in this
project) for early stopping; the held-out test points are touched once.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit

from base_model import BaseMLP
from data_loader import load_split

SEED = 42
OUT_DIR = "../outputs_csi"


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def euclidean_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - true, axis=1)


def main():
    set_seed()
    split = load_split()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, val_idx = next(gss.split(split.X_train, split.y_train, split.point_train))
    assert set(split.point_train[tr_idx]).isdisjoint(split.point_train[val_idx])

    X_tr, y_tr = split.X_train[tr_idx], split.y_train[tr_idx]
    X_val, y_val = split.X_train[val_idx], split.y_train[val_idx]
    X_test, y_test = split.X_test, split.y_test

    target_mean = y_tr.astype(np.float64).mean(axis=0).astype(np.float32)

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32)

    X_tr_t, y_tr_t = to_tensor(X_tr), to_tensor(y_tr - target_mean)
    X_val_t = to_tensor(X_val)
    X_test_t = to_tensor(X_test)

    model = BaseMLP(input_dim=split.num_features)
    # weight_decay raised 1e-5 -> 1e-3: only ~250 distinct training
    # locations here (however many redundant packets each), and the first
    # attempt (weaker regularization) overfit badly -- see base_model.py.
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    batch_size = 512  # larger than the RSSI pipelines': ~10x more training rows here
    n = X_tr_t.shape[0]
    best_val_err = float("inf")
    best_state = None
    patience, bad_epochs = 10, 0
    max_epochs = 60

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
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
            val_pred = model(X_val_t).numpy() + target_mean
        val_err = euclidean_error(val_pred, y_val).mean()

        if val_err < best_val_err - 1e-4:
            best_val_err = val_err
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f"epoch {epoch:3d}  train_mse={epoch_loss:.3f}  val_err={val_err:.3f}m")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}, best val_err={best_val_err:.3f}m")
            break

    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).numpy() + target_mean
    test_err = euclidean_error(test_pred, y_test)
    per_axis_mae = np.abs(test_pred - y_test).mean(axis=0)

    # Naive constant-prediction baseline, reported EVERY run from now on:
    # a first attempt silently lost to this baseline (5.27m model vs 4.90m
    # naive) due to overfitting that wasn't caught until asked to double
    # check -- this comparison is now a standing sanity check, not an
    # afterthought.
    naive_pred = np.tile(target_mean, (len(y_test), 1))
    naive_err = euclidean_error(naive_pred, y_test)

    print("\n=== Honest baseline (CSI, Lab Dataset, point-disjoint test split) ===")
    print(f"Mean Euclidean error : {test_err.mean():.3f} m   (naive constant-prediction baseline: {naive_err.mean():.3f} m)")
    print(f"Median Euclidean error: {np.median(test_err):.3f} m")
    print(f"90th pct error        : {np.percentile(test_err, 90):.3f} m")
    print(f"Per-axis MAE (x, y)   : {per_axis_mae[0]:.3f} m, {per_axis_mae[1]:.3f} m")
    if test_err.mean() >= naive_err.mean():
        print("*** WARNING: model does not beat the naive baseline. Do not trust this checkpoint. ***")

    torch.save(
        {"state_dict": model.state_dict(), "target_mean": target_mean, "input_dim": split.num_features,
         "feat_mean": split.feat_mean, "feat_std": split.feat_std},
        f"{OUT_DIR}/base_model.pt",
    )
    print(f"\nSaved base model to {OUT_DIR}/base_model.pt")


if __name__ == "__main__":
    main()
