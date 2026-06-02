import os
import sys
from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn

from pangaea.encoders.base import Encoder


def _get_panopticon_hub_dir() -> str:
    return os.path.join(torch.hub.get_dir(), "panopticon-FM_panopticon_main")


def _build_panopticon_vitb14(attn_dim: int = 2304) -> nn.Module:
    hub_dir = _get_panopticon_hub_dir()
    if hub_dir not in sys.path:
        sys.path.insert(0, hub_dir)

    from dinov2.models import vision_transformer as vits

    return vits.__dict__["vit_base"](
        img_size=518,
        patch_size=14,
        init_values=1.0e-05,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        num_register_tokens=0,
        embed_layer="PanopticonPE",
        pe_args=dict(
            attn_dim=attn_dim,
            chnfus_cfg=dict(
                layer_norm=False,
                attn_cfg=dict(num_heads=16),
            ),
        ),
    )


class PanopticonEncoder(Encoder):
    """Panopticon ViT-B/14 encoder.

    Paper: https://arxiv.org/abs/2503.10845
    Handles arbitrary band combinations via channel cross-attention (chn_ids).
    Use without BandPadding (no_pad preprocessing) so only real bands are passed.

    Args:
        encoder_weights (str | Path): Path to panopticon_vitb14_teacher.pth.
        input_bands (dict[str, list[str]]): Band names per modality.
        input_size (int): Spatial input size (224 recommended).
        embed_dim (int): ViT embedding dimension (768 for ViT-B).
        output_dim (int | list[int]): Output feature dimension(s) accepted by decoder.
        output_layers (list[int]): 0-indexed transformer block indices to return.
        wave_list (dict[str, dict[str, float]]): Maps band names to channel IDs
            (nm wavelengths for optical, negative integers for SAR).
        download_url (str): URL to download encoder_weights if missing.
        patch_size (int): ViT patch size (14).
        attn_dim (int): Internal attention dimension of PanopticonPE (2304).
        use_norm (bool): Whether to apply the encoder's final LayerNorm.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        output_dim: int | list[int],
        output_layers: list[int],
        wave_list: dict[str, dict[str, float]],
        download_url: str,
        patch_size: int = 14,
        attn_dim: int = 2304,
        use_norm: bool = False,
    ):
        super().__init__(
            model_name="panopticon",
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

        self.patch_size = patch_size
        self.use_norm = use_norm
        self.wave_list = wave_list

        self.encoder = _build_panopticon_vitb14(attn_dim=attn_dim)
        self.norm = self.encoder.norm

        # Build chn_ids from wave_list — shape (C,), registered as buffer so it
        # moves to the right device automatically.
        chn_ids = torch.tensor(
            [self.wave_list[m][bi] for m, b in self.input_bands.items() for bi in b],
            dtype=torch.float32,
        )
        self.register_buffer("chn_ids", chn_ids)

    @property
    def wv_list(self) -> list:
        return self.chn_ids.tolist()

    def set_wv_list_from_data_bands(self, data_bands: dict[str, list[str]]) -> None:
        """Restrict chn_ids to bands actually present in the dataset.

        Call this when BandPadding is NOT used so Panopticon receives only real
        bands with their true channel IDs rather than zero-padded phantom channels.
        """
        chn_ids = [
            self.wave_list[m][bi]
            for m, b in self.input_bands.items()
            for bi in b
            if m in data_bands and bi in data_bands[m]
        ]
        if not chn_ids:
            raise ValueError(
                "No bands remain after filtering to available data bands."
            )
        self.chn_ids = torch.tensor(chn_ids, dtype=torch.float32).to(
            self.chn_ids.device
        )

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        x = torch.cat(
            [image[m].squeeze(2) for m in self.input_bands.keys()], dim=1
        )  # (B, C, H, W)
        B = x.shape[0]

        # chn_ids: (B, C) — expand from buffer (C,)
        chn_ids = self.chn_ids.unsqueeze(0).expand(B, -1)
        x_dict = {"imgs": x, "chn_ids": chn_ids}

        if self.encoder.chunked_blocks:
            blocks = self.encoder._get_intermediate_layers_chunked(
                x_dict, self.output_layers
            )
        else:
            blocks = self.encoder._get_intermediate_layers_not_chunked(
                x_dict, self.output_layers
            )

        H = W = self.input_size // self.patch_size
        output = []
        for blk in blocks:
            if self.use_norm:
                blk = self.norm(blk)
            # drop CLS token, reshape (B, H*W, D) → (B, D, H, W)
            out = (
                blk[:, 1:]
                .permute(0, 2, 1)
                .view(B, self.embed_dim, H, W)
                .contiguous()
            )
            output.append(out)

        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained = torch.load(
            self.encoder_weights, map_location="cpu", weights_only=False
        )
        incompatible_shape = {}
        missing = {}
        matched = {}
        state = self.encoder.state_dict()
        for name, param in state.items():
            if name not in pretrained:
                missing[name] = param.shape
            elif pretrained[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained[name].shape)
            else:
                matched[name] = pretrained[name]

        self.encoder.load_state_dict(matched, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)
        logger.info(
            f"Panopticon: loaded {len(matched)}/{len(state)} parameters "
            f"from {self.encoder_weights}"
        )
