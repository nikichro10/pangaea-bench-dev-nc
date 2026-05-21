import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder

_ALIGN_CORNERS_MODES = {"bilinear", "bicubic"}


def _build_transposed_upsample(num_classes: int, scale: int) -> nn.Sequential:
    if scale <= 0 or (scale & (scale - 1)) != 0:
        raise ValueError(
            f"transposed_conv upsampling requires scale to be a power of 2, got {scale}. "
            f"Use bilinear or bicubic for non-power-of-2 patch sizes."
        )
    n_steps = int(math.log2(scale))
    return nn.Sequential(*[
        nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1)
        for _ in range(n_steps)
    ])


class MLPFusionDecoder(Decoder):
    """Multi-scale MLP fusion decoder for segmentation.

    Projects each encoder output layer to `channels` with a 1x1 conv,
    concatenates them, fuses with a 1x1 + 3x3 conv pair, classifies with a
    final 1x1 conv, and upsamples to the target resolution. Encoder is frozen.

    Args:
        encoder: frozen backbone; supplies output_layers / output_dim / patch_size.
        num_classes: number of segmentation classes.
        finetune: passed through to Decoder base (must be False for frozen encoder).
        channels: projection width; concat gives N*channels input to fusion (default 128).
        upsample_mode: "bilinear" (default), "bicubic", "nearest", or "transposed_conv".
    """

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        channels: int = 128,
        upsample_mode: str = "bilinear",
    ):
        super().__init__(encoder=encoder, num_classes=num_classes, finetune=finetune)

        self.model_name = "MLPFusionDecoder"
        self.upsample_mode = upsample_mode

        for param in self.encoder.parameters():
            param.requires_grad = False

        n_layers = len(encoder.output_layers)

        self.projections = nn.ModuleList([
            nn.Conv2d(dim, channels, kernel_size=1)
            for dim in encoder.output_dim
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(n_layers * channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Conv2d(channels, num_classes, kernel_size=1)

        if upsample_mode == "transposed_conv":
            self.upsample_layer = _build_transposed_upsample(num_classes, encoder.patch_size)
        else:
            self.upsample_layer = None

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        with torch.no_grad():
            feats = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})

        projected = [F.relu(proj(f), inplace=True) for proj, f in zip(self.projections, feats)]
        x = torch.cat(projected, dim=1)

        x = self.fusion(x)
        x = self.cls_head(x)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        if self.upsample_layer is not None:
            x = self.upsample_layer(x)
            if tuple(x.shape[-2:]) != tuple(output_shape):
                x = F.interpolate(x, size=output_shape, mode="bilinear", align_corners=False)
        else:
            align_corners = False if self.upsample_mode in _ALIGN_CORNERS_MODES else None
            x = F.interpolate(x, size=output_shape, mode=self.upsample_mode,
                               align_corners=align_corners)

        return x
