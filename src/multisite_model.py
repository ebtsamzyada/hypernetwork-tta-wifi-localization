import torch
import torch.nn as nn

from model import ConvBlock, SpatialPyramidPool


class MultiSiteGlobLocCNN(nn.Module):
    """Shared 5-block conv trunk + SPP + xy_head (all site-agnostic, since
    the regression target is already local pixel coordinates within each
    sample's own generated image -- see PIVOT_PLAN.md "Multi-site audit").
    Floor classification is NOT shared: floor label spaces aren't
    comparable across sites (site A's "floor 2" has no relation to site
    B's "floor 2"), so each site gets its own small linear floor head off
    the same 128-d bottleneck."""

    def __init__(self, site_num_floors: dict, spp_levels=(4, 2, 1)):
        super().__init__()
        channels = [1, 16, 32, 64, 128, 128]
        self.blocks = nn.ModuleList([ConvBlock(channels[i], channels[i + 1]) for i in range(5)])
        self.spp = SpatialPyramidPool(spp_levels)
        spp_out = channels[-1] * sum(lvl * lvl for lvl in spp_levels)

        self.trunk = nn.Sequential(
            nn.Linear(spp_out, 512), nn.GroupNorm(8, 512), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(512, 128), nn.GroupNorm(8, 128), nn.LeakyReLU(0.1, inplace=True),
        )
        self.xy_head = nn.Sequential(
            nn.Linear(128, 64), nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 2),
        )
        self.floor_heads = nn.ModuleDict({
            site_id: nn.Linear(128, n) for site_id, n in site_num_floors.items()
        })

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def features(self, x):
        for block in self.blocks:
            x = block(x)
            if x.shape[-1] == 0 or x.shape[-2] == 0:
                break
        return self.trunk(self.spp(x))

    def forward(self, x, site_id: str):
        """site_id must be a key in self.floor_heads, EXCEPT for zero-shot
        sites not seen during training -- pass site_id=None to get only
        (xy_pred, None), skipping floor classification entirely (there is
        no head to use)."""
        feat = self.features(x)
        xy_pred = self.xy_head(feat)
        floor_logits = self.floor_heads[site_id](feat) if site_id in self.floor_heads else None
        return xy_pred, floor_logits
