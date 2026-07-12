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


VISIBLE_EPS = 1e-6


class APGraphEncoder(nn.Module):
    """Graph/set-attention encoder over AP tokens, replacing BaseMLP's
    fully-connected trunk. See module-level rationale below.

    Why: the whole debugging journey on the plain-MLP pipeline traced back
    to one root cause -- a fixed-length RSSI vector where most entries are
    "not detected" confounds location identity with drift signal, because
    a plain MLP has no notion of "AP identity" as a first-class object; it
    just has one weight per (input position, hidden unit) pair. This
    encoder treats each AP as its own token with a learned identity
    embedding, explicitly modeling AP-AP relationships via self-attention
    and dynamically down-weighting absent APs via an attention mask,
    rather than forcing a fixed-position vector through dense layers.
    This is the graph-encoder idea from the original proposal, adapted to
    RSSI (no real CSI/antenna-pair data here) as a "set transformer over
    AP tokens" -- self-attention over a fully connected graph IS a graph
    attention formulation, and it avoids variable-size-graph batching
    complexity (every AP index is always a token; visibility is an
    attention mask, not a change in graph size).

    "Visible" is derived directly from the scaled RSSI input (a detected
    reading scales to > 0; the "not detected" sentinel scales to ~0 by
    construction -- see data_loader.py's scale_rssi), so the public
    signature stays exactly `trunk(x_scaled)`, identical to BaseMLP.
    """

    def __init__(self, num_aps: int, embed_dim: int = 16, model_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, out_dim: int = 128):
        super().__init__()
        self.num_aps = num_aps
        self.ap_embedding = nn.Embedding(num_aps, embed_dim)
        self.input_proj = nn.Linear(embed_dim + 1, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim * 2,
            batch_first=True, dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Sequential(nn.Linear(model_dim, out_dim), nn.ReLU())
        self.register_buffer("ap_ids", torch.arange(num_aps))

    def forward(self, x_scaled: torch.Tensor) -> torch.Tensor:
        """x_scaled: (batch, num_aps) or (num_aps,). Returns (batch, out_dim)."""
        squeeze = x_scaled.dim() == 1
        if squeeze:
            x_scaled = x_scaled.unsqueeze(0)
        batch = x_scaled.shape[0]

        visible = x_scaled > VISIBLE_EPS  # (batch, num_aps)

        ap_emb = self.ap_embedding(self.ap_ids).unsqueeze(0).expand(batch, -1, -1)
        node_in = torch.cat([ap_emb, x_scaled.unsqueeze(-1)], dim=-1)
        tokens = self.input_proj(node_in)  # (batch, num_aps, model_dim)

        key_padding_mask = ~visible  # True = ignore this AP in attention
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            # Defensive: a fully-empty buffer row would otherwise NaN the
            # softmax. Doesn't happen on real data but guard anyway.
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        encoded = self.transformer(tokens, src_key_padding_mask=key_padding_mask)

        mask_f = visible.float().unsqueeze(-1)
        pooled = (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        out = self.out_proj(pooled)
        return out.squeeze(0) if squeeze else out


class GraphBaseModel(nn.Module):
    def __init__(self, num_aps: int, out_dim: int = 128):
        super().__init__()
        self.trunk = APGraphEncoder(num_aps, out_dim=out_dim)
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
