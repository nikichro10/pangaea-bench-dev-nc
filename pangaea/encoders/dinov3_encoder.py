from logging import Logger
from pathlib import Path

import timm
import torch
from safetensors.torch import load_file

from pangaea.encoders.base import Encoder


class DINOv3_Encoder(Encoder):
    """
    DINOv3 encoder for the PANGAEA benchmark (RGB satellite baseline).

    Paper: "DINOv3: Self-supervised learning for vision at unprecedented scale" (2025)
    https://arxiv.org/abs/2508.10104

    Wraps timm's Eva-based DINOv3 backbone. Supports the satellite-pretrained
    variants (sat493m) distilled on 493M Maxar RGB ortho-rectified images.

    Uses forward_intermediates() to extract multi-scale spatial features.
    No wavelength conditioning — expects 3-channel RGB input only.

    Available variants:
      vit_large_patch16_dinov3.sat493m  — embed_dim=1024, depth=24
      vit_7b_patch16_dinov3.sat493m     — embed_dim=4096, depth=40
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        timm_model: str,
        embed_dim: int,
        output_layers: list[int],
        output_dim: int | list[int],
        download_url: str,
    ):
        super().__init__(
            model_name=timm_model,
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        self.model = timm.create_model(
            timm_model,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        imgs = x["optical"]  # (B, 3, H, W)
        return self.model.forward_intermediates(
            imgs,
            indices=list(self.output_layers),
            norm=True,
            output_fmt="NCHW",
            intermediates_only=True,
        )

    def load_encoder_weights(self, logger: Logger) -> None:
        state_dict = load_file(self.encoder_weights)

        missing: dict[str, torch.Size] = {}
        incompatible_shape: dict[str, tuple[torch.Size, torch.Size]] = {}
        to_load: dict[str, torch.Tensor] = {}

        for name, param in self.model.named_parameters():
            if name not in state_dict:
                missing[name] = param.shape
            elif state_dict[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, state_dict[name].shape)
            else:
                to_load[name] = state_dict[name]

        for name, buf in self.model.named_buffers():
            if name in state_dict and state_dict[name].shape == buf.shape:
                to_load[name] = state_dict[name]

        self.model.load_state_dict(to_load, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)
