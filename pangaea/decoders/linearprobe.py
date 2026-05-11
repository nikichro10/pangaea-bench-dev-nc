import torch
import torch.nn as nn
import torch.nn.functional as F

from pangaea.decoders.base import Decoder
from pangaea.encoders.base import Encoder


class SegLinearProbe(Decoder):
    """Strict linear probe for segmentation.

    Takes only the final encoder layer features (B, D, H/p, W/p),
    applies a single 1x1 conv to project to num_classes, then
    bilinearly upsamples to the target resolution.

    The only trainable layer is the 1x1 conv — the encoder is always frozen.
    """

    def __init__(
        self,
        encoder: Encoder,
        num_classes: int,
        finetune: bool,
    ):
        super().__init__(encoder=encoder, num_classes=num_classes, finetune=finetune)

        self.model_name = "SegLinearProbe"

        for param in self.encoder.parameters():
            param.requires_grad = False

        in_dim = self.encoder.output_dim[-1]
        self.conv = nn.Conv2d(in_dim, num_classes, kernel_size=1)

    def forward(
        self, img: dict[str, torch.Tensor], output_shape: torch.Size | None = None
    ) -> torch.Tensor:
        with torch.no_grad():
            feat = self.encoder({k: v[:, :, 0, :, :] for k, v in img.items()})

        x = feat[-1]  # (B, D, H/p, W/p)
        x = self.conv(x)  # (B, num_classes, H/p, W/p)

        if output_shape is None:
            output_shape = img[list(img.keys())[0]].shape[-2:]

        return F.interpolate(x, size=output_shape, mode="bilinear", align_corners=False)
