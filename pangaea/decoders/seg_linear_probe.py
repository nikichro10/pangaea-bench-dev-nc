import torch
import torch.nn as nn
import torch.nn.functional as F

from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder

_ALIGN_CORNERS_MODES = {"bilinear", "bicubic"}


class LinearProbeDecoder(Decoder):
    """Single linear layer probe for segmentation.

    Takes only the last encoder output layer, applies a single 1x1 conv
    (no nonlinearity) to map directly to num_classes, and upsamples to the
    target resolution. Encoder is frozen throughout.

    For multi-temporal inputs (multi_temporal > 1) the encoder is called once
    per timestep and the last-layer features are mean-pooled across time before
    the classification head, preserving the strictly linear probe property.

    Args:
        encoder: frozen backbone; last element of output_dim gives the input channels.
        num_classes: number of segmentation classes.
        finetune: passed through to Decoder base (must be False).
        multi_temporal: number of input timesteps. 0/False/1 → single-temporal.
        upsample_mode: interpolation mode for upsampling (default "bilinear").
    """

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        multi_temporal: int = 1,
        upsample_mode: str = "bilinear",
    ):
        super().__init__(encoder=encoder, num_classes=num_classes, finetune=finetune)

        self.model_name = "LinearProbeDecoder"
        self.upsample_mode = upsample_mode
        self.multi_temporal = int(multi_temporal) if multi_temporal else 1

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.cls_head = nn.Conv2d(encoder.output_dim[-1], num_classes, kernel_size=1)

    def _encode_single(self, img: dict[str, torch.Tensor], t: int) -> torch.Tensor:
        with torch.no_grad():
            feats = self.encoder({k: v[:, :, t, :, :] for k, v in img.items()})
        return feats[-1]

    def _upsample(self, x: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
        align_corners = False if self.upsample_mode in _ALIGN_CORNERS_MODES else None
        return F.interpolate(x, size=output_shape, mode=self.upsample_mode,
                             align_corners=align_corners)

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        if self.multi_temporal > 1:
            # Encode each timestep, mean-pool across T: (B, C, H, W)
            frames = [self._encode_single(img, t) for t in range(self.multi_temporal)]
            x = torch.stack(frames, dim=0).mean(dim=0)
        else:
            x = self._encode_single(img, 0)

        x = self.cls_head(x)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        return self._upsample(x, output_shape)


class SiamLinearProbeDecoder(Decoder):
    """Siamese linear probe for change detection.

    Encodes T=0 ("before") and T=1 ("after") independently with the frozen
    encoder, concatenates the last-layer features, applies a single 1x1 conv
    (no nonlinearity) to map to num_classes, and upsamples to the target
    resolution.

    Args:
        encoder: frozen backbone; last element of output_dim gives the input channels.
        num_classes: number of output classes.
        finetune: passed through to Decoder base (must be False).
        upsample_mode: interpolation mode for upsampling (default "bilinear").
    """

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
        upsample_mode: str = "bilinear",
    ):
        super().__init__(encoder=encoder, num_classes=num_classes, finetune=finetune)

        self.model_name = "SiamLinearProbeDecoder"
        self.upsample_mode = upsample_mode

        for param in self.encoder.parameters():
            param.requires_grad = False

        # Concat doubles the channel count before the classification head
        self.cls_head = nn.Conv2d(2 * encoder.output_dim[-1], num_classes, kernel_size=1)

    def _encode_single(self, img: dict[str, torch.Tensor], t: int) -> torch.Tensor:
        with torch.no_grad():
            feats = self.encoder({k: v[:, :, t, :, :] for k, v in img.items()})
        return feats[-1]

    def _upsample(self, x: torch.Tensor, output_shape: torch.Size) -> torch.Tensor:
        align_corners = False if self.upsample_mode in _ALIGN_CORNERS_MODES else None
        return F.interpolate(x, size=output_shape, mode=self.upsample_mode,
                             align_corners=align_corners)

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        feat1 = self._encode_single(img, 0)
        feat2 = self._encode_single(img, 1)

        x = self.cls_head(torch.cat((feat1, feat2), dim=1))

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        return self._upsample(x, output_shape)
