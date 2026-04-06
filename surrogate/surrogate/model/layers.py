"""Core layers for the 3D Fourier Neural Operator."""

import torch
import torch.nn as nn


class SpectralConv3d(nn.Module):
    """3D Fourier layer: FFT -> truncate modes -> learned complex multiply -> iFFT.

    Operates on real-valued inputs via rfftn/irfftn.  Retains only the first
    ``(modes_z, modes_y, modes_x)`` frequency coefficients and applies a
    learned complex weight tensor to mix channels in the spectral domain.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_z: int,
        modes_y: int,
        modes_x: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_z = modes_z
        self.modes_y = modes_y
        self.modes_x = modes_x

        scale = 1.0 / (in_channels * out_channels)

        # Weights are stored as real float32 tensors with a trailing dim of 2
        # (real part, imaginary part).  view_as_complex() in forward gives a
        # zero-copy complex view.  This avoids storing cfloat Parameters, which
        # GradScaler cannot handle (no CUDA kernel for ComplexFloat unscaling).
        shape = (in_channels, out_channels, modes_z, modes_y, modes_x, 2)
        self.weights1 = nn.Parameter(scale * torch.randn(shape))
        self.weights2 = nn.Parameter(scale * torch.randn(shape))
        self.weights3 = nn.Parameter(scale * torch.randn(shape))
        self.weights4 = nn.Parameter(scale * torch.randn(shape))

    @staticmethod
    def _compl_mul3d(
        input_tensor: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Complex multiplication + channel mixing via einsum.

        input_tensor: (B, C_in, D, H, W) complex
        weights:      (C_in, C_out, D, H, W) complex
        output:       (B, C_out, D, H, W) complex
        """
        return torch.einsum("bidhw,iodhw->bodhw", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Real tensor of shape (B, C, D, H, W).

        Returns:
            Real tensor of shape (B, C_out, D, H, W).
        """
        B, C, D, H, W = x.shape
        orig_dtype = x.dtype

        # FFT and complex weight multiply must run in float32:
        # ComplexHalf is not supported by the einsum/baddbmm CUDA kernels, so
        # we explicitly cast out of any AMP float16 context for this block.
        with torch.amp.autocast("cuda", enabled=False):
            x32 = x.float()

            # Reconstruct complex views from the real-valued parameter storage
            w1 = torch.view_as_complex(self.weights1.float())
            w2 = torch.view_as_complex(self.weights2.float())
            w3 = torch.view_as_complex(self.weights3.float())
            w4 = torch.view_as_complex(self.weights4.float())

            # 3D real FFT (compact along last dim)
            x_ft = torch.fft.rfftn(x32, dim=[-3, -2, -1])
            # x_ft shape: (B, C, D, H, W//2+1)

            mz, my, mx = self.modes_z, self.modes_y, self.modes_x

            # Allocate output spectrum
            out_ft = torch.zeros(
                B, self.out_channels, D, H, W // 2 + 1,
                dtype=torch.cfloat, device=x.device,
            )

            # Fill the 4 octants (positive/negative Z & Y, positive X only due to rfft)
            # Octant 1: +Z, +Y, +X
            out_ft[:, :, :mz, :my, :mx] = self._compl_mul3d(
                x_ft[:, :, :mz, :my, :mx], w1
            )
            # Octant 2: -Z, +Y, +X
            out_ft[:, :, -mz:, :my, :mx] = self._compl_mul3d(
                x_ft[:, :, -mz:, :my, :mx], w2
            )
            # Octant 3: +Z, -Y, +X
            out_ft[:, :, :mz, -my:, :mx] = self._compl_mul3d(
                x_ft[:, :, :mz, -my:, :mx], w3
            )
            # Octant 4: -Z, -Y, +X
            out_ft[:, :, -mz:, -my:, :mx] = self._compl_mul3d(
                x_ft[:, :, -mz:, -my:, :mx], w4
            )

            # Inverse FFT back to spatial domain
            x_out = torch.fft.irfftn(out_ft, s=(D, H, W), dim=[-3, -2, -1])

        # Cast output back to the input dtype so the bypass conv addition is type-consistent
        return x_out.to(orig_dtype)


class FourierLayer(nn.Module):
    """Single Fourier layer: spectral conv + bypass conv + GELU.

    Follows the standard FNO pattern:
        out = GELU(SpectralConv3d(x) + Conv3d_1x1x1(x))
    """

    def __init__(
        self,
        width: int,
        modes_z: int,
        modes_y: int,
        modes_x: int,
    ):
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes_z, modes_y, modes_x)
        self.bypass = nn.Conv3d(width, width, kernel_size=1)
        self.norm = nn.InstanceNorm3d(width)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.spectral(x) + self.bypass(x)))
