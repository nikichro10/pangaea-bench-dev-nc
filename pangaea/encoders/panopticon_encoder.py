import os
import sys
from logging import Logger
from pathlib import Path

import torch
import torch.nn as nn

from pangaea.encoders.base import Encoder


def _get_panopticon_hub_dir() -> str:
    return os.path.join(torch.hub.get_dir(), "panopticon-FM_panopticon_main")


def _ensure_hub_code() -> None:
    """Clone the Panopticon repo via torch.hub if not already present.

    Only runs on first use; subsequent calls are a no-op directory check.
    Note: hubconf.py unconditionally re-downloads the weights as a side effect
    of torch.hub.load, so this is only called when the directory is missing.
    """
    hub_dir = _get_panopticon_hub_dir()
    if not os.path.isdir(hub_dir):
        torch.hub.load(
            "panopticon-FM/panopticon",
            "panopticon_vitb14",
            trust_repo=True,
            skip_validation=True,
        )


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

    The hub repo is cloned once on first use; weights are downloaded once to
    encoder_weights by the base class and loaded directly from there — no
    re-download on subsequent runs or concurrent jobs.

    Args:
        encoder_weights (str | Path): Local path for the checkpoint.
        input_bands (dict[str, list[str]]): Band names per modality.
        input_size (int): Spatial input size (224 recommended).
        embed_dim (int): ViT embedding dimension (768 for ViT-B).
        output_dim (int | list[int]): Output feature dimension(s) accepted by decoder.
        output_layers (list[int]): 0-indexed transformer block indices to return.
        wave_list (dict[str, dict[str, float]]): Maps band names to channel IDs
            (nm wavelengths for optical, negative integers for SAR).
        download_url (str): URL for the base class to download encoder_weights.
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
        self._attn_dim = attn_dim

        # encoder and norm are built in load_encoder_weights once hub code is available
        self.encoder: nn.Module | None = None
        self.norm: nn.Module | None = None

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
        self.register_buffer(
            "chn_ids",
            torch.tensor(chn_ids, dtype=torch.float32),
        )

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        x = torch.cat(
            [image[m].squeeze(2) for m in self.input_bands.keys()], dim=1
        )  # (B, C, H, W)
        B, _, H, W = x.shape

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

        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        output = []
        for blk in blocks:
            if self.use_norm:
                blk = self.norm(blk)
            # drop CLS token, reshape (B, H*W, D) → (B, D, H, W)
            out = (
                blk[:, 1:]
                .permute(0, 2, 1)
                .view(B, self.embed_dim, patch_h, patch_w)
                .contiguous()
            )
            output.append(out)

        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        # Ensure hub code is cloned (no-op if already present, so safe for
        # concurrent jobs — they all skip the download).
        _ensure_hub_code()

        self.encoder = _build_panopticon_vitb14(attn_dim=self._attn_dim)
        self.norm = self.encoder.norm

        # Load from the local file downloaded by base class download_model().
        # The checkpoint is a flat state dict (confirmed from hubconf.py).
        pretrained = torch.load(
            self.encoder_weights, map_location="cpu", weights_only=False
        )
        incompatible = self.encoder.load_state_dict(pretrained, strict=False)
        del pretrained

        if incompatible.missing_keys:
            logger.warning(
                "Panopticon: missing keys:\n"
                + "\n".join(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.warning(
                "Panopticon: unexpected keys:\n"
                + "\n".join(incompatible.unexpected_keys)
            )
        logger.info(
            f"Panopticon: loaded weights from {self.encoder_weights}"
        )
