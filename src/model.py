import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        g = min(groups, out_ch)
        while out_ch % g != 0:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g, out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, ceil_mode=True),
        )

    def forward(self, x):
        return self.net(x)


class SpatialPyramidPool(nn.Module):
    def __init__(self, levels=(4, 2, 1)):
        super().__init__()
        self.levels = levels

    def forward(self, x):
        pooled = [
            nn.functional.adaptive_max_pool2d(x, output_size=(lvl, lvl)).reshape(1, -1)
            for lvl in self.levels
        ]
        return torch.cat(pooled, dim=1)


class GlobLocCNN(nn.Module):
    """Shared 5-block conv trunk + SPP, two heads off a 128-d bottleneck:
    xy_head (regression) and floor_head (classification, num_floors
    classes). WiFi-only single-channel input."""

    def __init__(self, num_floors, spp_levels=(4, 2, 1)):
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
        self.floor_head = nn.Linear(128, num_floors)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
            if x.shape[-1] == 0 or x.shape[-2] == 0:
                break
        feat = self.trunk(self.spp(x))
        return self.xy_head(feat), self.floor_head(feat)
