"""
Base network: a plain MLP mapping CSI amplitude features to (x, y).

Kept deliberately simple (trunk + single linear head), matching the
"don't over-engineer the base network" discipline from the original
scope -- and matching the RSSI pipelines' BaseMLP exactly in structure,
so the hypernetwork mechanism (Step 4/5) transfers unchanged: it only
ever re-weights `head`, never `trunk`.

Regularized more than the RSSI BaseMLP (dropout + smaller hidden layers):
a first attempt without dropout and with hidden=(128,64) overfit badly on
this dataset -- 0.64m train error vs 5.14m validation error (an 8x gap),
and lost to a trivial constant-prediction baseline on the held-out test
points. Root cause: only ~250 distinct training LOCATIONS (however many
packets each), which is little spatial diversity for a network of that
capacity to generalize from rather than memorize.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BaseMLP(nn.Module):
    def __init__(self, input_dim: int, hidden=(64, 32), dropout: float = 0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden[1], 2)  # predicts centered (x, y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def get_head_params(self):
        return self.head.weight.data.clone(), self.head.bias.data.clone()

    def set_head_params(self, weight: torch.Tensor, bias: torch.Tensor):
        with torch.no_grad():
            self.head.weight.copy_(weight)
            self.head.bias.copy_(bias)


class SubcarrierGraphEncoder(nn.Module):
    """Graph over the 30 subcarrier groups (nodes), for this dataset's
    single Tx-Rx link. Each node's raw feature is the CSI amplitude across
    the 3 receive antennas at that subcarrier. Edges connect each
    subcarrier to its nearby neighbors IN FREQUENCY (a banded adjacency,
    not full attention) -- unlike the RSSI graph encoder (which had no
    real physical edge structure to exploit and fell back to a fully-
    connected attention set), frequency adjacency between subcarriers is
    a genuine physical relationship (nearby subcarriers experience
    correlated fading), so this is a more principled graph than the RSSI
    one: a fixed structural mask shared across the whole batch, not a
    per-sample visibility mask.

    Input layout: the flat (batch, 90) feature vector is
    antenna-major -- reshape to (batch, 3, 30) then permute to
    (batch, 30, 3) to get (subcarrier, antenna) node tokens, matching
    data_loader.py's `amplitude.reshape(3*30, packets).T` flattening.
    """

    def __init__(self, num_antennas: int = 3, num_subcarriers: int = 30,
                 model_dim: int = 64, n_heads: int = 4, n_layers: int = 2,
                 out_dim: int = 128, freq_window: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.input_proj = nn.Linear(num_antennas, model_dim)
        self.pos_embedding = nn.Embedding(num_subcarriers, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim * 2,
            batch_first=True, dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.feat_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Sequential(nn.Linear(model_dim, out_dim), nn.ReLU())
        self.register_buffer("subcarrier_ids", torch.arange(num_subcarriers))

        idx = torch.arange(num_subcarriers)
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
        freq_mask = dist > freq_window  # True = blocked (too far in frequency)
        self.register_buffer("freq_mask", freq_mask)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        squeeze = x_flat.dim() == 1
        if squeeze:
            x_flat = x_flat.unsqueeze(0)
        batch = x_flat.shape[0]

        x = x_flat.view(batch, self.num_antennas, self.num_subcarriers).permute(0, 2, 1)
        tokens = self.input_proj(x) + self.pos_embedding(self.subcarrier_ids).unsqueeze(0)

        encoded = self.transformer(tokens, mask=self.freq_mask)
        pooled = self.feat_dropout(encoded.mean(dim=1))  # every subcarrier always present, plain mean is fine
        out = self.out_proj(pooled)
        return out.squeeze(0) if squeeze else out


class GraphBaseModel(nn.Module):
    def __init__(self, num_antennas: int = 3, num_subcarriers: int = 30, out_dim: int = 128):
        super().__init__()
        self.trunk = SubcarrierGraphEncoder(num_antennas, num_subcarriers, out_dim=out_dim)
        self.head = nn.Linear(out_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    def get_head_params(self):
        return self.head.weight.data.clone(), self.head.bias.data.clone()

    def set_head_params(self, weight: torch.Tensor, bias: torch.Tensor):
        with torch.no_grad():
            self.head.weight.copy_(weight)
            self.head.bias.copy_(bias)
