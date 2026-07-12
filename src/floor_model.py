"""
Floor classifier: a small MLP mapping scaled RSSI fingerprints to a floor
probability distribution over building 2's 5 floors.

Why this exists: UJIIndoorLoc floors within one building share almost the
identical (x,y) footprint (verified: median distance from any point to the
nearest point on a DIFFERENT floor is ~0m, and 96-100% of points have
another floor's point within 5m). The base MLP's target explicitly
discards floor, so without floor information the model must resolve
genuine cross-floor RSSI ambiguity using only within-floor signal
differences -- a floor-oracle check showed this costs ~2.5m of avoidable
mean error (10.80m -> 8.25m). Floor turns out to be highly (though not
perfectly) recoverable from RSSI alone (~90% accuracy held-out), so this
classifier's predicted floor probabilities are used as extra input
features to the localization MLP (see base_model.py's augmented input),
rather than needing the true (unavailable at deployment) floor label.

This classifier is trained once, frozen, and treated as a fixed
preprocessing step (like scale_rssi) for the rest of the pipeline -- it is
NOT adapted by the hypernetwork. It runs on drifted input just like the
localization network, so its accuracy degrading under drift is a
realistic, expected part of the deployment simulation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit

N_FLOORS = 5


class FloorClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, N_FLOORS),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.net(x), dim=-1)


def train_floor_classifier(X_train, floor_train, groups_train, seed=42, verbose=True):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(gss.split(X_train, floor_train, groups_train))

    X_tr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
    y_tr = torch.tensor(floor_train[tr_idx], dtype=torch.long)
    X_val = torch.tensor(X_train[val_idx], dtype=torch.float32)
    y_val = torch.tensor(floor_train[val_idx], dtype=torch.long)

    torch.manual_seed(seed)
    clf = FloorClassifier(input_dim=X_train.shape[1])
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-5)

    best_acc, best_state, bad = 0.0, None, 0
    for epoch in range(300):
        clf.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 64):
            idx = perm[i : i + 64]
            opt.zero_grad()
            loss = F.cross_entropy(clf(X_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()

        clf.eval()
        with torch.no_grad():
            val_acc = (clf(X_val).argmax(dim=-1) == y_val).float().mean().item()
        if val_acc > best_acc + 1e-4:
            best_acc, best_state, bad = val_acc, {k: v.clone() for k, v in clf.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 20:
            break

    clf.load_state_dict(best_state)
    if verbose:
        print(f"Floor classifier: best held-out val accuracy = {best_acc:.4f}")
    return clf
