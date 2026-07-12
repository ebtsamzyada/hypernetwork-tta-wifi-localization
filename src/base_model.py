"""
Base network: a plain MLP mapping RSSI fingerprints to (x, y).

Kept deliberately simple (trunk + single linear head) because the head is
exactly the layer the hypernetwork (Step 4/5) will later re-weight at
deployment time. get_head_params/set_head_params expose that layer as flat
tensors so the hypernetwork's output can be dropped straight in.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden=(256, 128)):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden[1], 2)  # predicts centered (x, y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Trunk output, i.e. the input the head/hypernetwork acts on."""
        return self.trunk(x)

    def get_head_params(self):
        return self.head.weight.data.clone(), self.head.bias.data.clone()

    def set_head_params(self, weight: torch.Tensor, bias: torch.Tensor):
        with torch.no_grad():
            self.head.weight.copy_(weight)
            self.head.bias.copy_(bias)


class FloorAwareBaseModel(nn.Module):
    """Wraps a frozen floor classifier + the localization BaseMLP so the
    rest of the pipeline (hypernetwork meta-training, deployment sim) can
    keep calling `.trunk(x_scaled_rssi)` / `.get_head_params()` exactly as
    before, with RAW SCALED RSSI as input -- the floor-probability
    augmentation happens internally and transparently.

    Why a wrapper instead of baking floor into BaseMLP directly: the
    hypernetwork only ever adapts `loc_mlp.head`, never the classifier, so
    keeping them as separate frozen submodules makes that boundary
    explicit and keeps BaseMLP itself unchanged/reusable.
    """

    def __init__(self, floor_clf: nn.Module, loc_mlp: BaseMLP):
        super().__init__()
        self.floor_clf = floor_clf
        self.loc_mlp = loc_mlp
        self.head = loc_mlp.head

    def augment(self, x_scaled_rssi: torch.Tensor) -> torch.Tensor:
        floor_probs = self.floor_clf.predict_proba(x_scaled_rssi)
        return torch.cat([x_scaled_rssi, floor_probs], dim=-1)

    def trunk(self, x_scaled_rssi: torch.Tensor) -> torch.Tensor:
        return self.loc_mlp.trunk(self.augment(x_scaled_rssi))

    def forward(self, x_scaled_rssi: torch.Tensor) -> torch.Tensor:
        return self.loc_mlp(self.augment(x_scaled_rssi))

    def get_head_params(self):
        return self.loc_mlp.get_head_params()

    def set_head_params(self, weight: torch.Tensor, bias: torch.Tensor):
        self.loc_mlp.set_head_params(weight, bias)
