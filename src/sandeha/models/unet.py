"""A U-Net with residual blocks, GroupNorm and SiLU."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _Block(nn.Module):
    def __init__(self, channels_in: int, channels: int, groups: int):
        super().__init__()
        self.norm_in = nn.GroupNorm(min(groups, channels_in), channels_in)
        self.conv_in = nn.Conv2d(channels_in, channels, 3, padding=1)
        self.norm_out = nn.GroupNorm(min(groups, channels), channels)
        self.conv_out = nn.Conv2d(channels, channels, 3, padding=1)
        self.skip = nn.Conv2d(channels_in, channels, 1) if channels_in != channels else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv_in(F.silu(self.norm_in(x)))
        y = self.conv_out(F.silu(self.norm_out(y)))
        return y + self.skip(x)


class UNet(nn.Module):
    """``levels`` resolutions, doubling the width at each coarser one."""

    def __init__(self, base_width: int = 32, levels: int = 4, groups: int = 8, inputs: int = 6, outputs: int = 3):
        super().__init__()
        widths = [base_width * 2**level for level in range(levels)]
        self.first = nn.Conv2d(inputs, base_width, 3, padding=1)
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        previous = base_width
        for width in widths:
            self.down.append(_Block(previous, width, groups))
            previous = width
        for width in widths[:-1]:
            self.pool.append(nn.Conv2d(width, width, 3, stride=2, padding=1))
        self.up = nn.ModuleList(
            _Block(widths[level + 1] + widths[level], widths[level], groups)
            for level in reversed(range(levels - 1))
        )
        self.last_norm = nn.GroupNorm(min(groups, base_width), base_width)
        self.last = nn.Conv2d(base_width, outputs, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        y = self.first(x)
        skips = []
        for level, block in enumerate(self.down):
            y = block(y)
            if level < len(self.pool):
                skips.append(y)
                y = self.pool[level](y)
        for block in self.up:
            skip = skips.pop()
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))
        return self.last(F.silu(self.last_norm(y)))
