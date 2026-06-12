from logging import Logger
from pathlib import Path

import timm
import torch

from pangaea.encoders.base import Encoder


class DINOv2_Encoder(Encoder):
    """
    DINOv2 ViT-B/14 encoder for the PANGAEA benchmark (pure-RGB baseline).

    Paper: "DINOv2: Learning Robust Visual Features without Supervision" (TMLR 2024)
    https://arxiv.org/abs/2304.07193

    Loads the standard ViT-B/14 backbone pretrained on LVD-142M via DINO v2.
    Uses timm's get_intermediate_layers to extract multi-scale spatial features.
    No wavelength conditioning — expects 3-channel RGB input only.

    Weights: https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        output_layers: list[int],
        output_dim: int | list[int],
        download_url: str,
    ):
        super().__init__(
            model_name="dinov2_vitb14",
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
            "vit_base_patch14_dinov2.lvd142m",
            pretrained=False,
            img_size=input_size,
            num_classes=0,
        )

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        imgs = x["optical"]  # (B, 3, H, W)
        return self.model.get_intermediate_layers(
            imgs,
            n=list(self.output_layers),
            reshape=True,
            norm=True,
        )

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained = torch.load(self.encoder_weights, map_location="cpu")

        # Unwrap common checkpoint wrappers
        for key in ("model", "teacher", "state_dict"):
            if isinstance(pretrained, dict) and key in pretrained and isinstance(pretrained[key], dict):
                pretrained = pretrained[key]
                break

        # Strip common prefixes (backbone., encoder., module.)
        state_dict: dict[str, torch.Tensor] = {}
        for k, v in pretrained.items():
            for prefix in ("backbone.", "encoder.", "module."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            state_dict[k] = v

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

        # Also load buffers (e.g. pos_embed stored as non-parameter)
        for name, buf in self.model.named_buffers():
            if name in state_dict and state_dict[name].shape == buf.shape:
                to_load[name] = state_dict[name]

        self.model.load_state_dict(to_load, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)
