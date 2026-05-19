# Model components adapted from TorchGeo (MIT License) and
# https://github.com/Panopticon-FM/panopticon

import math
from logging import Logger
from pathlib import Path
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pangaea.encoders.base import Encoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    return x if isinstance(x, tuple) else (x, x)


def _resize_pos_embed(pos_embed: Tensor, new_grid: int, old_grid: int) -> Tensor:
    """Bicubic interpolation of absolute positional embeddings.

    Args:
        pos_embed: [1, 1 + old_grid², embed_dim]
        new_grid:  target grid size (e.g. 224//14 = 16)
        old_grid:  source grid size stored in the checkpoint (e.g. 518//14 = 37)
    """
    assert pos_embed.shape[1] == 1 + old_grid ** 2, (
        f"pos_embed has {pos_embed.shape[1]} tokens, expected 1+{old_grid}²={1 + old_grid**2}"
    )
    if new_grid == old_grid:
        return pos_embed
    cls_tok = pos_embed[:, :1]
    patch   = pos_embed[:, 1:]                                    # [1, old_grid², D]
    D       = patch.shape[-1]
    patch   = patch.reshape(1, old_grid, old_grid, D).permute(0, 3, 1, 2).float()
    patch   = F.interpolate(patch, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    patch   = patch.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, D)
    return torch.cat([cls_tok.to(patch.dtype), patch], dim=1)


def _sincos_pos_embed(embed_dim: int, pos: Tensor) -> Tensor:
    """Standard 1-D sincos positional embedding.

    Args:
        embed_dim: must be even.
        pos: [M] positions.

    Returns:
        [M, embed_dim]
    """
    assert embed_dim % 2 == 0
    half   = embed_dim // 2
    omega  = torch.arange(half, dtype=torch.float32, device=pos.device) / half
    omega  = 1.0 / (10000 ** omega)
    out    = torch.outer(pos.view(-1).float(), omega)   # [M, D/2]
    return torch.cat([out.sin(), out.cos()], dim=1)     # [M, D]


# ---------------------------------------------------------------------------
# Channel attention components (from TorchGeo / Panopticon)
# ---------------------------------------------------------------------------

class Conv3dWrapper(nn.Module):
    """Channel-wise patchification: [B, C, H, W] → [B, C, L, D]."""

    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        ps = _to_tuple(patch_size)
        self.conv3d = nn.Conv3d(1, embed_dim, kernel_size=(1, ps[0], ps[1]),
                                stride=(1, ps[0], ps[1]))

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.conv3d(x.unsqueeze(1)).squeeze(1)   # B D C Hp Wp
        hp, wp = x.shape[-2:]
        return x.flatten(-2).permute(0, 2, 3, 1), hp, wp  # B C L D


class CrossAttnNoQueryProj(nn.Module):
    """Cross-attention without query projection."""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5
        self.inproj_q  = nn.Identity()
        self.inproj_k  = nn.Linear(dim, dim, bias=qkv_bias)
        self.inproj_v  = nn.Linear(dim, dim, bias=qkv_bias)

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                key_padding_mask: Tensor | None = None) -> Tensor:
        B, Nq, D = q.shape
        H = self.num_heads
        dh = D // H
        q = self.inproj_q(q).reshape(B, Nq, H, dh).permute(0, 2, 1, 3) * self.scale
        k = self.inproj_k(k).reshape(B, k.shape[1], H, dh).permute(0, 2, 1, 3)
        v = self.inproj_v(v).reshape(B, v.shape[1], H, dh).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1)
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None], float("-inf"))
        attn = attn.softmax(dim=-1)
        return (attn @ v).transpose(1, 2).reshape(B, Nq, D)


class ChnEmb(nn.Module):
    """Channel embeddings from wavelength IDs (nm for optical, <0 for SAR).

    Optical: sincos embedding of the wavelength in nm.
    SAR: learned embeddings for up to 12 polarization/orbit combinations,
         addressed by negative integer IDs (-1 … -12).
         Mapping used here (verify against Panopticon sentinel1.yaml):
           -1 → VV, -2 → VH, -3 → ASC_VV, -4 → ASC_VH, -5 → DSC_VV, -6 → DSC_VH
    """

    def __init__(self, embed_dim: int, opt_coarsity: int = 1) -> None:
        super().__init__()
        self.embed_dim    = embed_dim
        self.opt_coarsity = opt_coarsity
        d1 = embed_dim // 3
        d2 = embed_dim - 2 * d1
        self.embed_transmit = nn.Parameter(torch.zeros(2, d1))
        self.embed_receive  = nn.Parameter(torch.zeros(2, d1))
        self.embed_orbit    = nn.Parameter(torch.zeros(2, d2))

    def forward(self, chn_ids: Tensor) -> Tensor:
        """
        Args:
            chn_ids: [B, C] — wavelengths in nm (optical) or negative int (SAR).
        Returns:
            [B, C, embed_dim]
        """
        mus         = chn_ids.clone().float()
        sar_mask    = mus < 0
        opt_mask    = ~sar_mask
        dtype       = self.embed_transmit.dtype
        embs        = torch.zeros(*mus.shape, self.embed_dim,
                                  device=mus.device, dtype=dtype)

        # Optical: sincos of coarsened wavelength
        if opt_mask.any():
            mus_opt = (mus[opt_mask] // self.opt_coarsity)
            embs[opt_mask] = _sincos_pos_embed(self.embed_dim,
                                               mus_opt.view(-1)).to(dtype)

        # SAR: learned embeddings indexed by -(id+1)
        if sar_mask.any():
            transmit = torch.cat([self.embed_transmit[0].repeat(2, 1),
                                   self.embed_transmit[1].repeat(2, 1)], dim=0).repeat(3, 1)
            receive  = torch.cat([self.embed_receive[0].unsqueeze(0),
                                   self.embed_receive[1].repeat(2, 1),
                                   self.embed_receive[0].unsqueeze(0)], dim=0).repeat(3, 1)
            orbit    = torch.stack([self.embed_orbit.mean(0),
                                     self.embed_orbit[0],
                                     self.embed_orbit[1]]).repeat_interleave(4, dim=0)
            sar_embs = torch.cat([transmit, receive, orbit], dim=1)  # [12, embed_dim]
            embs[sar_mask] = sar_embs[(-(mus[sar_mask] + 1)).long()]

        return embs


class ChnAttn(nn.Module):
    """Cross-attention over channels: [B, C, L, D] → [B, L, D]."""

    def __init__(self, dim: int,
                 chnemb_cfg: dict[str, Any] = {},
                 attn_cfg: dict[str, Any] = {},
                 layer_norm: bool = False) -> None:
        super().__init__()
        self.chnemb = ChnEmb(**chnemb_cfg, embed_dim=dim)
        self.query  = nn.Parameter(torch.randn(1, 1, dim))
        self.xattn  = CrossAttnNoQueryProj(dim=dim, **attn_cfg)
        if layer_norm:
            self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, chn_ids: Tensor,
                mask: Tensor | None = None) -> Tensor:
        B, _, L, D = x.shape
        x = x + self.chnemb(chn_ids).unsqueeze(2)         # add channel embeddings
        x = x.permute(0, 2, 1, 3).flatten(0, 1)           # [BL, C, D]
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, L, -1).flatten(0, 1)
        query = self.query.expand(x.shape[0], -1, -1)     # [BL, 1, D]
        x = self.xattn(query, x, x, key_padding_mask=mask)
        x = x.reshape(B, L, D)
        if hasattr(self, "layer_norm"):
            x = self.layer_norm(x)
        return x


class PanopticonPE(nn.Module):
    """Panopticon patch embedding: dict → [B, num_patches, embed_dim]."""

    def __init__(self, attn_dim: int, embed_dim: int, patch_size: int,
                 chnfus_cfg: dict[str, Any] = {}, img_size: int = 224) -> None:
        super().__init__()
        self.conv3d    = Conv3dWrapper(patch_size=patch_size, embed_dim=attn_dim)
        self.chnfus    = ChnAttn(**chnfus_cfg, dim=attn_dim)
        self.proj      = nn.Linear(attn_dim, embed_dim)
        self.patch_size, self.img_size, self.grid_size, self.num_patches = \
            self._init_sizes(patch_size, img_size)

    def _init_sizes(self, patch_size, img_size):
        ps = _to_tuple(patch_size)
        im = _to_tuple(img_size)
        grid = (im[0] // ps[0], im[1] // ps[1])
        return ps, im, grid, grid[0] * grid[1]

    def forward(self, x_dict: dict[str, Tensor]) -> Tensor:
        x       = x_dict["imgs"]
        chn_ids = x_dict["chn_ids"]
        mask    = x_dict.get("spec_masks", None)
        x, _, _ = self.conv3d(x)          # [B, C, L, D]
        x       = self.chnfus(x, chn_ids=chn_ids, mask=mask)  # [B, L, D]
        return self.proj(x)               # [B, L, embed_dim]


# ---------------------------------------------------------------------------
# Pangaea Encoder
# ---------------------------------------------------------------------------

class PanopticonEncoder(Encoder):
    """Panopticon ViT-Base Foundation Model encoder for Pangaea.

    Paper: https://arxiv.org/abs/2503.10845
    Weights: https://huggingface.co/lewaldm/panopticon

    Wavelength-aware via channel cross-attention (PanopticonPE).
    Optical channel IDs in nanometres; SAR uses negative integers per the
    Panopticon sentinel1.yaml convention.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        output_dim: int | list[int],
        output_layers: list[int],
        download_url: str,
        wave_list: dict[str, dict[str, float]],
        patch_size: int = 14,
        attn_dim: int = 2304,
        use_norm: bool = False,
    ) -> None:
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
        self.attn_dim   = attn_dim
        self.use_norm   = use_norm

        # wave_list is the config-provided wavelength dict (nm for optical, <0 for SAR).
        # It is the single source of truth for building and restricting wv_list.
        self._wave_list_dict: dict[str, dict[str, float]] = wave_list

        # wv_list: flat list of wavelengths (nm) for all active bands, in order.
        # Updated by set_wv_list_from_data_bands when BandPadding is not used.
        self.wv_list: list[float] = [
            wave_list[mod][band]
            for mod, bands in input_bands.items()
            for band in bands
        ]

        # Build timm ViT and replace patch_embed with PanopticonPE
        self.vit: nn.Module = timm.create_model(
            "vit_base_patch14_dinov2", pretrained=False
        )
        self.vit.patch_embed = PanopticonPE(
            attn_dim=attn_dim,
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=input_size,
            chnfus_cfg={"attn_cfg": {"num_heads": 16}},
        )
        # Resize pos_embed to match our img_size (checkpoint stores 518-px version)
        n_patches = (input_size // patch_size) ** 2
        self.vit.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + n_patches, embed_dim)
        )

    # ------------------------------------------------------------------
    def set_wv_list_from_data_bands(self, data_bands: dict[str, list[str]]) -> None:
        """Restrict wv_list to bands actually present in the dataset.

        Both wv_list and the forward pass read from the same source (_wave_list_dict
        filtered by input_bands ∩ data_bands), keeping channel count consistent.
        Call this when BandPadding is NOT used.
        """
        wv_list = [
            self._wave_list_dict[mod][band]
            for mod, bands in self.input_bands.items()
            for band in bands
            if mod in data_bands and band in data_bands[mod]
        ]
        if not wv_list:
            raise ValueError("No wavelengths remain after filtering to available data bands.")
        self.wv_list = wv_list

    # ------------------------------------------------------------------
    def forward(self, image: dict[str, Tensor]) -> list[Tensor]:
        # 1. Concatenate modalities — decoders strip T before calling the encoder
        x = torch.cat([image[mod] for mod in self.input_bands], dim=1)  # [B, C, H, W]
        B = x.shape[0]

        assert x.shape[1] == len(self.wv_list), (
            f"Channel count mismatch: image has {x.shape[1]} channels "
            f"but wv_list has {len(self.wv_list)} entries. "
            "Did set_wv_list_from_data_bands run correctly?"
        )

        # 2. Build chn_ids tensor [B, C] — wavelengths in nm
        wavelist = torch.tensor(self.wv_list, dtype=x.dtype, device=x.device)
        chn_ids  = wavelist.unsqueeze(0).expand(B, -1)  # [B, C]

        x_dict = {"imgs": x, "chn_ids": chn_ids}

        # 3. Patch embedding → [B, num_patches, embed_dim]
        vit = self.vit
        x   = vit.patch_embed(x_dict)

        # 4. Positional embedding + CLS token
        x = vit._pos_embed(x)   # [B, num_patches+1, embed_dim]
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)

        # 5. Transformer blocks — collect intermediate outputs.
        # use_norm=True: apply vit.norm to every collected output (consistent).
        # use_norm=False: return raw block outputs (consistent, matches DOFA default).
        H_out = W_out = self.input_size // self.patch_size
        outputs = []

        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            if i in self.output_layers:
                feat = x
                if self.use_norm:
                    feat = vit.norm(feat)
                feat = (
                    feat[:, vit.num_prefix_tokens:]
                    .permute(0, 2, 1)
                    .view(B, -1, H_out, W_out)
                    .contiguous()
                )
                outputs.append(feat)

        return outputs

    # ------------------------------------------------------------------
    def load_encoder_weights(self, logger: Logger) -> None:
        # Checkpoint is a pure state dict — weights_only=True is safe.
        state_dict = torch.load(
            self.encoder_weights, map_location="cpu", weights_only=True
        )
        # mask_token is used during DINO pre-training only
        state_dict.pop("mask_token", None)

        # Checkpoint stores pos_embed for 518-px input (37×37 = 1369 patch tokens).
        # Resize to our target grid size via bicubic interpolation.
        old_grid = 518 // self.patch_size   # 37
        new_grid = self.input_size // self.patch_size
        state_dict["pos_embed"] = _resize_pos_embed(
            state_dict["pos_embed"], new_grid=new_grid, old_grid=old_grid
        )

        # strict=True: checkpoint matches the model exactly after the two edits above.
        # Any future mismatch will surface immediately rather than silently using
        # random weights.
        missing, unexpected = self.vit.load_state_dict(state_dict, strict=True)
        # These two should never fire given the analysis above, but kept as a safety net.
        if missing:
            logger.error("Panopticon — missing keys:\n" +
                         "\n".join(f"  {k}" for k in sorted(missing)))
            raise RuntimeError("Panopticon checkpoint is missing required keys (see above).")
        if unexpected:
            logger.error("Panopticon — unexpected keys:\n" +
                         "\n".join(f"  {k}" for k in sorted(unexpected)))
            raise RuntimeError("Panopticon checkpoint has unexpected keys (see above).")
        logger.info("Panopticon weights loaded from %s", self.encoder_weights)
