"""A Fourier neural operator, with coordinate channels because the box isn't periodic."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _Spectral(nn.Module):
    """A learned product on the lowest modes, along both signs of the row frequency."""

    def __init__(self, width: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1 / (width * width)
        # Real and imaginary parts stored apart, as gradient scaling handles only real tensors.
        self.low = nn.Parameter(scale * torch.randn(width, width, modes, modes, 2))
        self.high = nn.Parameter(scale * torch.randn(width, width, modes, modes, 2))

    def forward(self, x: Tensor) -> Tensor:
        # FFTs have no half-precision form, so this part always runs in full precision.
        with torch.autocast(x.device.type, enabled=False):
            height, width = x.shape[-2:]
            spectrum = torch.fft.rfft2(x.float(), norm="ortho")
            rows = min(self.modes, height // 2)
            columns = min(self.modes, width // 2 + 1)
            out = torch.zeros_like(spectrum)
            for weights, row_slice in ((self.low, slice(0, rows)), (self.high, slice(-rows, None))):
                product = torch.view_as_complex(weights[:, :, :rows, :columns].float())
                out[:, :, row_slice, :columns] = torch.einsum(
                    "bixy,ioxy->boxy", spectrum[:, :, row_slice, :columns], product
                )
            return torch.fft.irfft2(out, s=(height, width), norm="ortho")


class FNO2d(nn.Module):
    def __init__(self, width: int = 48, modes: int = 24, layers: int = 4, inputs: int = 6, outputs: int = 3):
        super().__init__()
        self.lift = nn.Conv2d(inputs + 2, width, 1)
        self.spectral = nn.ModuleList(_Spectral(width, modes) for _ in range(layers))
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(nn.Conv2d(width, 2 * width, 1), nn.SiLU(), nn.Conv2d(2 * width, outputs, 1))

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        rows = torch.linspace(0, 1, height, device=x.device, dtype=x.dtype)
        columns = torch.linspace(0, 1, width, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(rows, columns, indexing="ij")
        coordinates = torch.stack([grid_x, grid_y]).expand(batch, -1, -1, -1)
        y = self.lift(torch.cat([x, coordinates], dim=1))
        for spectral, local in zip(self.spectral, self.local):
            y = F.silu(spectral(y) + local(y))
        return self.project(y)
