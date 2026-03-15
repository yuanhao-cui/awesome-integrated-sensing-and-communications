"""
CNN-based near-field beam training model.

Implements a UNet-like convolutional neural network that maps estimated CSI
to phase-only beamforming vectors for XL-MIMO systems.

Architecture (from paper Section IV-A):
    Input: Real and imaginary parts of estimated CSI (batch, 1, 2, Nt)
    Output: Phase values in [-1, 1] mapped to unit-norm beamforming vector (batch, Nt)

The encoder progressively downsamples along the antenna dimension while expanding
feature channels, then the decoder upsamples back to the original resolution.
A final linear layer + tanh produces the beamforming phases.

Reference:
    Eq. (15)-(16) in the paper describe the CNN input/output mapping.
"""

from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn


class BeamTrainingNet(nn.Module):
    """UNet-like CNN for near-field beam training in XL-MIMO.

    Takes estimated CSI (real + imaginary concatenated along dim=1) and outputs
    phase values for constructing a unit-norm analog beamforming vector.

    Args:
        in_channels: Number of input channels (default: 1 for complex CSI
            stored as real+imag stacked along spatial dim).
        out_channels: Number of output channels from the decoder (default: 1).
        init_features: Base number of feature maps, doubled at each encoder level.
        antenna_count: Number of antennas N_t (default: 256).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        init_features: int = 8,
        antenna_count: int = 256,
    ):
        super().__init__()
        self.antenna_count = antenna_count
        features = init_features

        # Encoder path
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.encoder3 = self._block(features * 2, features * 2, name="enc3")

        # Decoder path
        self.upconv2 = nn.ConvTranspose2d(
            features * 2, features * 2, kernel_size=(1, 2), stride=(1, 2)
        )
        self.decoder2 = self._block(features * 2, features, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features, features, kernel_size=(1, 2), stride=(1, 2)
        )
        self.decoder1 = self._block(features, out_channels, name="dec1")

        # Output head: flatten spatial features → phase values
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(antenna_count * 2, antenna_count)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 2, Nt) containing real and
                imaginary parts of estimated CSI.

        Returns:
            Phase values of shape (batch, Nt) in [-1, 1]. These are multiplied
            by pi in trans_vrf to obtain the actual phases for beamforming.
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        dec2 = self.upconv2(enc3)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(dec1)
        dec1 = self.flatten(dec1)
        dec1 = self.linear(dec1)
        dec1 = self.tanh(dec1)
        return dec1

    @staticmethod
    def _block(in_channels: int, features: int, name: str) -> nn.Sequential:
        """Create a convolutional block with two Conv2d + BN + ReLU layers.

        Args:
            in_channels: Number of input channels.
            features: Number of output feature maps.
            name: Prefix for layer names (for OrderedDict keys).

        Returns:
            Sequential block with conv-norm-relu × 2.
        """
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=2,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=2,
                            padding=0,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
