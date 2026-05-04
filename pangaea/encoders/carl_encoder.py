# Copyright (c) 2026 Division of Intelligent Medical Systems, DKFZ.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
# Original implementation: https://github.com/IMSY-DKFZ/CARL
#
# PANGAEA integration: CARLEncoder wraps CARLModel with PANGAEA's Encoder
# interface, mapping band names to wavelengths and returning multi-scale
# feature maps for UPerNet-style decoders.
#
# Pretrained weights (CARL-SSL):
#   wget -O pretrained_models/carl_ssl_checkpoint.ckpt \
#        https://zenodo.org/records/18671944/files/ssl_checkpoint_carl.ckpt

import logging
import math
from functools import partial
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models import Eva, VisionTransformer
from timm.models.layers import trunc_normal_
from torch import Tensor

from pangaea.encoders.base import Encoder


# ---------------------------------------------------------------------------
# Positional / utility helpers
# ---------------------------------------------------------------------------

def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, theta=10000):
    """
    grid_size: int of the grid length
    return:
    pos_embed: [grid_size, embed_dim] or [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid = np.arange(grid_size, dtype=float)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid, theta=theta)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, theta=10000):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / theta**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    if len(masks) == 1:
        # Optimize single mask case
        m = masks[0]
        if x.ndim == 4:
            mask_keep = m.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(-2), x.size(-1))
        elif x.ndim == 3:
            mask_keep = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        return torch.gather(x, dim=1, index=mask_keep)
    all_x = []
    for m in masks:
        if x.ndim == 4:
            mask_keep = m.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.size(-2), x.size(-1))
        elif x.ndim == 3:
            mask_keep = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


# ---------------------------------------------------------------------------
# Core building blocks (Attention, MLP, Block)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Fourier-feature positional encoding for continuous wavelength values.

    Projects scalar wavelength values into a high-dimensional feature space
    using random Fourier features, enabling the model to distinguish bands
    by their physical wavelength.
    """

    def __init__(self, d_hid, sigma):
        """
        Args:
            d_hid (int): Output feature dimension.
            sigma (float): Scale of the Gaussian random frequencies.
                           Set to 0 to disable encoding (returns zeros).
        """
        super().__init__()
        self.d_hid = d_hid
        self.sigma = sigma
        # Draw fixed random frequencies at construction time so they are
        # reproducible and match those used during pretraining.
        rng_state = torch.get_rng_state()
        temp_gen = torch.Generator()
        temp_gen.manual_seed(0)  # fixed seed for reproducible sampling
        B_gauss = 2 * torch.tensor(math.pi) * (torch.randn((d_hid // 2), generator=temp_gen) * sigma + 0)
        torch.set_rng_state(rng_state)
        B_gauss = torch.stack([B_gauss[i // 2] for i in range(d_hid)], dim=0)
        self.register_buffer("dims", B_gauss, persistent=True)

    def get_position_angle_vec(self, position):
        position = position.unsqueeze(-1)
        dims = self.dims.unsqueeze(0).unsqueeze(0)
        return position * dims

    def forward(self, w) -> torch.Tensor:
        """Encode a batch of wavelength sequences.

        Args:
            w (Tensor): Wavelength values of shape (B, C).

        Returns:
            Tensor: Encoded features of shape (B, C, d_hid).
        """
        if self.sigma == 0:
            b, c = w.shape
            return torch.zeros(b, c, self.d_hid, device=w.device, dtype=w.dtype)
        wlens_encoded = self.get_position_angle_vec(w)
        sin_encoded = torch.sin(wlens_encoded[..., 0::2])
        cos_encoded = torch.cos(wlens_encoded[..., 1::2])
        wlens_encoded_out = torch.empty_like(wlens_encoded)
        wlens_encoded_out[..., 0::2] = sin_encoded
        wlens_encoded_out[..., 1::2] = cos_encoded
        return wlens_encoded_out


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_bias, dropout_p=self.attn_drop.p, is_causal=False)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dim_k: int = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        if dim_k is None:
            dim_k = dim
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim_k, dim, bias=qkv_bias)
        self.v = nn.Linear(dim_k, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, attn_bias=None) -> Tensor:
        B, N, C = q.shape
        B, N_k, _ = k.shape
        q = self.q(q).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(k).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(v).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_bias, dropout_p=self.attn_drop.p, is_causal=False)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    """Transformer block supporting both self-attention and cross-attention.

    The block type (self vs. cross) is determined by the ``attn_class``
    argument passed at construction. Cross-attention blocks expect additional
    ``keys`` and ``values`` arguments in the forward call.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        proj_drop=0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               proj_bias=proj_bias, attn_drop=attn_drop, proj_drop=proj_drop)
        self.is_cross_attn = isinstance(self.attn, CrossAttention)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.sample_drop_ratio = drop_path

    def forward(self, x, keys=None, values=None) -> Tensor:
        if self.is_cross_attn:
            return self._forward_cross(x, keys, values)
        # Self-attention path
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    def _forward_cross(self, queries, keys=None, values=None) -> Tensor:
        # Cross-attention: queries attend to spectral tokens (keys/values)
        out = self.ls1(self.attn(q=self.norm1(queries), k=self.norm1(keys), v=self.norm1(values)))
        queries = queries + self.drop_path1(out)
        queries = queries + self.drop_path2(self.ls2(self.mlp(self.norm2(queries))))
        return queries


# ---------------------------------------------------------------------------
# SpectralEncoder
# ---------------------------------------------------------------------------

class SpectralEncoder(nn.Module):
    """Encoder that ingests spectral tokens and produces query embeddings.

    The encoder accepts a sequence of spectral tokens ``x`` and a corresponding
    wavelength tensor ``w``. It builds wavelength positional encodings (via
    ``PositionalEncoding``), maintains a small set of learned ``queries``
    (spectral representations), and alternates self- and cross-attention blocks
    so queries can attend to the spectral tokens and produce a compact spectral
    representation for downstream tasks.

    Args:
        embed_dim: Embedding dimension for tokens and queries.
        n_queries: Number of learned spectral representations.
        pos_enc_sigma: Scaling parameter for wavelength positional encoding.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Expansion ratio for MLP hidden dimension.
        qkv_bias: Whether to include bias in query, key, value projections.
        ffn_bias: Whether to include bias in feed-forward network.
        proj_bias: Whether to include bias in projection layers.
        drop_path_rate: Stochastic depth rate.
        drop_path_uniform: Whether to use uniform stochastic depth schedule.
        act_layer: Activation function layer.
        proj_drop: Dropout rate for projections.
        drop: General dropout rate.
        attn_drop: Dropout rate for attention weights.
        layer_scale: Layer scaling factor for initialization.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        n_queries: int = 4,
        pos_enc_sigma: float = 3,
        depth: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        act_layer=nn.GELU,
        proj_drop: float = 0.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        layer_scale: float = 1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.n_queries = n_queries

        # Initialize learned query vectors and their positional encodings
        self.queries = nn.Parameter(torch.zeros(1, n_queries, embed_dim), requires_grad=True)
        self.queries_pos_enc = nn.Parameter(torch.zeros(1, n_queries, embed_dim), requires_grad=False)

        # Build stochastic depth schedule
        dpr = [drop_path_rate] * depth if drop_path_uniform else \
              [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Alternating self-attention (even) and cross-attention (odd) blocks.
        # Self-attention refines the spectral tokens; cross-attention lets the
        # learned queries attend to those tokens.
        blocks_list = []
        for i in range(depth):
            attn_class = SelfAttention if i % 2 == 0 else CrossAttention
            blocks_list.append(Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, proj_bias=proj_bias, ffn_bias=ffn_bias,
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                ffn_layer=Mlp, init_values=layer_scale, attn_class=attn_class,
                proj_drop=proj_drop, drop=drop, attn_drop=attn_drop,
            ))
        self.blocks = nn.ModuleList(blocks_list)
        self.norm = norm_layer(embed_dim)
        self.positional_embedding = PositionalEncoding(embed_dim, sigma=pos_enc_sigma)

        self.rng = torch.Generator()
        self.rng.manual_seed(0)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights from DINOv2 if embed_dim matches, else random."""
        torch.nn.init.trunc_normal_(self.queries, std=0.02, generator=self.rng)
        queries_pos_enc = get_1d_sincos_pos_embed(self.embed_dim, self.n_queries, cls_token=False)
        self.queries_pos_enc.data.copy_(torch.from_numpy(queries_pos_enc).float().unsqueeze(0))

        # Use DINOv2 block weights as initialization for supported embed dims
        if self.embed_dim == 384:
            model_init = "vit_small_patch14_dinov2.lvd142m"
        elif self.embed_dim == 768:
            model_init = "vit_base_patch14_dinov2.lvd142m"
        else:
            def _init(module):
                if isinstance(module, nn.Linear):
                    trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            self.apply(_init)
            return

        # Load DINOv2 block weights, splitting the fused qkv projection into
        # separate q, k, v Linear layers used by CARL's attention modules.
        dino = timm.create_model(model_init, pretrained=True)
        block_weights = {k: v for k, v in dino.state_dict().items() if "blocks" in k}
        new_weights = {}
        for k, v in block_weights.items():
            block_id = int(k.split(".")[1])
            if block_id >= self.n_blocks:
                continue
            if "attn.qkv.weight" in k:
                for suffix, sl in [("q", slice(None, self.embed_dim)),
                                    ("k", slice(self.embed_dim, 2 * self.embed_dim)),
                                    ("v", slice(2 * self.embed_dim, None))]:
                    new_weights[k.replace("attn.qkv.weight", f"attn.{suffix}.weight")] = v[sl]
            elif "attn.qkv.bias" in k:
                for suffix, sl in [("q", slice(None, self.embed_dim)),
                                    ("k", slice(self.embed_dim, 2 * self.embed_dim)),
                                    ("v", slice(2 * self.embed_dim, None))]:
                    new_weights[k.replace("attn.qkv.bias", f"attn.{suffix}.bias")] = v[sl]
            else:
                new_weights[k] = v
        missing, unexpected = self.load_state_dict(new_weights, strict=False)
        logging.info(f"SpectralEncoder init from {model_init}: {len(missing)} missing, {len(unexpected)} unexpected")
        del dino

    def forward(
        self, x: torch.Tensor, w: torch.Tensor, masks=None
    ) -> dict[str, torch.Tensor]:
        """Encode spectral tokens and produce query embeddings.

        The network alternates between refining the spectral tokens with
        self-attention blocks and allowing the learned queries to attend to
        the tokens with cross-attention blocks.

        Args:
            x: Spectral token embeddings of shape (batch_size, seq_len, embed_dim).
            w: Wavelength values of shape (batch_size, seq_len).
            masks: Optional channel indices to keep (used in CARL-SSL training).

        Returns:
            Dictionary with:
                - ``queries``: shape (batch_size, n_queries, embed_dim).
                - ``spectral_tokens``: shape (batch_size, seq_len, embed_dim).
        """
        # Add wavelength positional encoding to spectral tokens
        pos_enc = self.positional_embedding(w)
        repeats = x.shape[0] // pos_enc.shape[0]
        if repeats > 1:
            pos_enc = pos_enc.repeat_interleave(repeats, dim=0)
        x = x + pos_enc
        if masks is not None:
            x = apply_masks(x, masks)

        # Initialize queries with learned embeddings + positional encoding
        queries = self.queries.expand(x.shape[0], -1, -1) + \
                  self.queries_pos_enc.expand(x.shape[0], -1, -1)

        # Alternate: self-attention on spectral tokens, cross-attention for queries
        for blk in self.blocks:
            if isinstance(blk.attn, CrossAttention):
                queries = blk(queries, x, x)
            else:
                x = blk(x)

        return {"queries": self.norm(queries), "spectral_tokens": x}


# ---------------------------------------------------------------------------
# TimmWrapper (spatial encoder)
# ---------------------------------------------------------------------------

class TimmWrapper(nn.Module):
    """Wrapper for TIMM Vision Transformers.

    Provides a clean interface to pre-trained TIMM models (EVA2, ViT, ...).
    Supports dynamic image sizes and optional depth truncation.

    Attributes:
        net: The underlying TIMM Vision Transformer model.
        embed_dim: Embedding dimension of the model.
        num_prefix_tokens: Number of prefix tokens (e.g., CLS token).
    """

    def __init__(
        self,
        model_name: str = "timm/eva02_base_patch14_224.mim_in22k",
        depth: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            model_name: TIMM model identifier.
            depth: If set, only the first ``depth`` transformer blocks are used.
                   This reduces compute and changes which intermediate features
                   are available via ``get_intermediate_layers``.
            model_kwargs: Additional keyword arguments passed to ``timm.create_model``.
        """
        super().__init__()
        model_kwargs = model_kwargs or {}
        model_kwargs["dynamic_img_size"] = True  # required for non-224 inputs

        timm_module: Union[Eva, VisionTransformer] = timm.create_model(
            model_name, pretrained=True, **model_kwargs)
        if depth is not None:
            timm_module.blocks = timm_module.blocks[:depth]
        self.net = timm_module
        self.num_prefix_tokens = self.net.num_prefix_tokens
        self.embed_dim = self.net.embed_dim

    def forward(self, x: torch.Tensor, masks: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass through the spatial encoder.

        Args:
            x: Input tensor of shape (B, H, W, D) — the spatial grid produced
               by the spectral encoder before patch flattening.
            masks: Optional patch-index masks used during CARL-SSL training.

        Returns:
            Spatial token sequence of shape (B, N, D) with prefix tokens removed.
        """
        out_pos = self.net._pos_embed(x)
        # EVA2 returns (tokens, rope_embeddings); ViT returns tokens only
        if isinstance(out_pos, tuple):
            x, rot_pos_embed = out_pos
        else:
            x, rot_pos_embed = out_pos, None

        block_kwargs = {}
        if rot_pos_embed is not None:
            block_kwargs["rope"] = rot_pos_embed
        if masks is not None:
            block_kwargs["attn_mask"] = self._create_attn_mask(x, masks)

        for block in self.net.blocks:
            x = block(x, **block_kwargs)
        if self.num_prefix_tokens > 0:
            x = x[:, self.num_prefix_tokens:, :]
        if masks is not None:
            x = apply_masks(x, masks)
        return x

    def _create_attn_mask(self, x, masks):
        """Build a boolean attention mask from patch-index masks (CARL-SSL only)."""
        reg_tokens = self.num_prefix_tokens
        b, n, d = x.shape
        masks = [m.cpu() for m in masks]
        zeros = [torch.zeros((b, reg_tokens), dtype=torch.bool) for _ in masks]
        adjusted = [torch.cat([z, m + reg_tokens], dim=1) for z, m in zip(zeros, masks)][0]
        attn_mask = torch.zeros((b, n, n), dtype=torch.bool)
        attn_mask[torch.arange(b).unsqueeze(1), adjusted] = True
        return attn_mask.transpose(1, 2).unsqueeze(1).to(x.device)


# ---------------------------------------------------------------------------
# CARLModel
# ---------------------------------------------------------------------------

class CARLModel(nn.Module):
    """Collaborative Attentive Representation Learning model.

    Combines a spectral encoder and a spatial encoder for multispectral / hyperspectral
    image analysis. The forward pass has four stages:

    1. **Per-image standardization** — zero-mean unit-variance normalization matching
       CARL's pretraining distribution (applied internally, independent of dataset stats).
    2. **Patch embedding** — each spectral band is embedded independently via a shared
       Conv2d (``embedder``), producing per-patch per-band tokens.
    3. **Spectral encoding** — the ``SpectralEncoder`` aggregates band tokens into a
       compact spectral representation using wavelength-conditioned cross-attention.
    4. **Spatial encoding** — the spectral representation is passed as a spatial grid
       to the ``TimmWrapper`` (EVA02 / ViT) for spatial feature extraction.

    Attributes:
        spectral_tf: Spectral encoder (wavelength-aware transformer).
        spatial_encoder: Vision transformer for spatial feature learning.
        embedder: Conv2d patch embedder (operates on single-channel band images).
        linear_connector: Projects spectral embeddings to spatial encoder dimension.
        num_features: Output feature dimension (equals ``spatial_encoder.embed_dim``).
        wv_list: List of central wavelengths (µm) for the input bands, set externally
                 by ``CARLEncoder`` based on the active ``input_bands`` configuration.
    """

    def __init__(
        self,
        spec_encoder_kwargs: Dict[str, Any] = {},
        spat_encoder_kwargs: Dict[str, Any] = {},
        patch_size: int = 8,
        ssl_ckpt_path: Optional[Union[str, Path]] = None,
        wave_list: Optional[Dict[str, List[float]]] = None,
        sensor: str = "enmap",
        **kwargs,
    ) -> None:
        """
        Args:
            spec_encoder_kwargs: Keyword arguments forwarded to ``SpectralEncoder``.
            spat_encoder_kwargs: Keyword arguments forwarded to ``TimmWrapper``.
            patch_size: Spatial patch size (e.g. 8 → 8×8 patches).
            ssl_ckpt_path: Optional path to a CARL-SSL checkpoint; if provided,
                           weights are loaded immediately. In the PANGAEA wrapper,
                           this is always ``None`` and loading is deferred to
                           ``CARLEncoder.load_encoder_weights``.
            wave_list: Dict mapping sensor name → list of wavelengths. Not used
                       by ``CARLEncoder``, which sets ``wv_list`` directly.
            sensor: Key into ``wave_list`` to select wavelengths. Unused when
                    ``CARLEncoder`` overrides ``wv_list``.
        """
        super().__init__()
        spec_encoder_kwargs = spec_encoder_kwargs or {}
        spat_encoder_kwargs = spat_encoder_kwargs or {}

        self.wave_list = wave_list
        self.sensor = sensor
        self.wv_list = wave_list[sensor] if wave_list is not None else None

        self.spectral_tf = SpectralEncoder(**spec_encoder_kwargs)
        self.patch_size = patch_size
        # Single-channel convolution: each band is treated as a greyscale image
        self.embedder = nn.Conv2d(1, self.spectral_tf.embed_dim, patch_size, patch_size)

        if spat_encoder_kwargs:
            self.spatial_encoder = TimmWrapper(**spat_encoder_kwargs)
            self.linear_connector = nn.Linear(
                self.spectral_tf.embed_dim, self.spatial_encoder.embed_dim)

        self.num_features = (self.spatial_encoder.embed_dim
                             if hasattr(self, "spatial_encoder")
                             else self.spectral_tf.embed_dim)

        if ssl_ckpt_path is not None:
            self.load_ssl_checkpoint(ssl_ckpt_path)

    @staticmethod
    def _standardize(img: torch.Tensor) -> torch.Tensor:
        """Per-image zero-mean unit-variance standardization (matching CARL pretraining).

        Args:
            img: Tensor of shape (B, C, H, W).

        Returns:
            Standardized tensor of the same shape.
        """
        mean = img.mean(dim=(1, 2, 3), keepdim=True)
        std = img.std(dim=(1, 2, 3), keepdim=True)
        return (img - mean) / (std + 1e-6)

    def _encode_spectral(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, int, int]:
        """Run patch embedding and spectral encoding.

        Args:
            img: Input image tensor of shape (B, C, H, W).

        Returns:
            Tuple of (spec_spatial, patch_height, patch_width) where
            ``spec_spatial`` has shape (B, H', W', D_spatial) and is ready to
            be consumed by the spatial encoder.
        """
        img = self._standardize(img)
        B, C, H, W = img.shape

        # Build wavelength tensor — shape (B, C), same wavelength for all items in batch
        wavelengths = torch.tensor(self.wv_list, device=img.device).float()
        wavelengths = wavelengths.unsqueeze(0).expand(B, -1)

        # Treat each band as an independent greyscale image for patch embedding
        # (B, C, H, W) → (B*C, 1, H, W) → (B*C, D, H', W')
        img_r = rearrange(img, "b c h w -> (b c) 1 h w")
        patch_emb = self.embedder(img_r)
        ph, pw = patch_emb.shape[-2], patch_emb.shape[-1]
        D = patch_emb.shape[1]

        # Reshape to spectral-encoder input: one token per band per spatial patch
        # (B*C, D, H', W') → (B*H'*W', C, D)
        patch_seq = rearrange(patch_emb, "(b c) d nh nw -> (b nh nw) c d",
                              b=B, c=C, nh=ph, nw=pw, d=D)

        # SpectralEncoder produces n_queries compact spectral features per patch
        spec_out = self.spectral_tf(patch_seq, wavelengths)
        spec_repr = spec_out["queries"].sum(dim=1)           # (B*H'*W', D_spec)
        spec_repr = F.layer_norm(spec_repr, spec_repr.shape[-1:])
        spec_repr = self.linear_connector(spec_repr)          # (B*H'*W', D_spatial)

        # Restore spatial structure for the spatial encoder
        # (B*H'*W', D_spatial) → (B, H', W', D_spatial)
        spec_repr = rearrange(spec_repr, "(b nh nw) d -> b nh nw d", b=B, nh=ph, nw=pw)
        return spec_repr, ph, pw

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Full forward pass returning a 2-D feature map.

        Args:
            img: Input image tensor of shape (B, C, H, W).

        Returns:
            Feature map of shape (B, D_spatial, H', W').
        """
        spec_repr, ph, pw = self._encode_spectral(img)
        B = spec_repr.shape[0]
        spatial = self.spatial_encoder(spec_repr)
        return rearrange(spatial, "b (h w) d -> b d h w",
                         b=B, d=spatial.shape[-1], h=ph, w=pw)

    def _intermediate_layers(
        self,
        img: torch.Tensor,
        n: Union[int, List[int]] = 1,
    ) -> List[torch.Tensor]:
        """Collect intermediate spatial encoder block outputs (includes prefix tokens).

        Args:
            img: Input tensor of shape (B, C, H, W).
            n: If int, collect last *n* blocks. If list, collect blocks at those indices.

        Returns:
            List of tensors of shape (B, N+prefix, D) from the selected blocks.
        """
        spec_repr, _, _ = self._encode_spectral(img)
        net = self.spatial_encoder.net
        out_pos = net._pos_embed(spec_repr)
        if isinstance(out_pos, tuple):
            x, rot_pos_embed = out_pos
        else:
            x, rot_pos_embed = out_pos, None

        block_kwargs = {}
        if rot_pos_embed is not None:
            block_kwargs["rope"] = rot_pos_embed

        num_blocks = len(net.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        outputs = []
        for i, blk in enumerate(net.blocks):
            x = blk(x, **block_kwargs)
            if i in take_indices:
                outputs.append(x)
        return outputs

    def get_intermediate_layers(
        self,
        img: torch.Tensor,
        n: Union[int, List[int]] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple:
        """Public API for extracting intermediate spatial features (DINOv2-style).

        This is the main method called by ``CARLEncoder.forward``. It strips
        prefix tokens (CLS) and optionally reshapes the token sequence back to
        a 2-D spatial grid.

        Args:
            img: Input tensor of shape (B, C, H, W).
            n: Block indices to return. List of ints → specific blocks (0-indexed).
               Single int → last *n* blocks.
            reshape: If True, reshape tokens to (B, D, H', W') spatial grid.
            return_prefix_tokens: If True, also return the CLS token alongside each output.
            norm: If True, apply the spatial encoder's final LayerNorm.

        Returns:
            Tuple of tensors; each has shape (B, D, H', W') when ``reshape=True``,
            or (B, N, D) otherwise.
        """
        net = self.spatial_encoder.net
        num_prefix = self.spatial_encoder.num_prefix_tokens
        outputs = self._intermediate_layers(img, n)

        if norm:
            outputs = [net.norm(o) for o in outputs]

        # Strip prefix tokens (CLS) — only patch tokens are spatially meaningful
        prefix_tokens = [o[:, :num_prefix] for o in outputs]
        outputs = [o[:, num_prefix:] for o in outputs]

        if reshape:
            _, _, h, w = img.shape
            gh, gw = h // self.patch_size, w // self.patch_size
            outputs = [
                o.reshape(img.shape[0], gh, gw, -1).permute(0, 3, 1, 2).contiguous()
                for o in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def load_ssl_checkpoint(self, ckpt_path: Union[str, Path]) -> None:
        """Load a CARL-SSL checkpoint.

        Handles both raw state dicts and Lightning-style checkpoints with a
        ``state_dict`` key. Strips common prefixes (``model.``, ``module.``)
        added by training wrappers.

        Args:
            ckpt_path: Path to the ``.ckpt`` file.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Loading CARL SSL checkpoint from {ckpt_path}")
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
            return
        state_dict = ckpt.get("state_dict", ckpt)
        new_state = {}
        for k, v in state_dict.items():
            k = k.removeprefix("model.").removeprefix("module.")
            new_state[k] = v
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        logger.info(f"CARL SSL: {len(missing)} missing, {len(unexpected)} unexpected keys")


# ---------------------------------------------------------------------------
# PANGAEA-compatible wrapper
# ---------------------------------------------------------------------------

class CARLEncoder(Encoder):
    """PANGAEA encoder wrapper that adapts CARLModel to the PANGAEA Encoder interface.

    **Architecture overview**

    CARL has two stages:

    1. *SpectralEncoder* — a small transformer (embed_dim=384, 8 blocks) that
       processes each spatial patch as a sequence of *C* band tokens.  Wavelength
       positional encodings make the encoder band-order agnostic: the same model
       works for any subset of the known bands.

    2. *TimmWrapper* (EVA02-base) — a standard spatial ViT that receives the
       spectral encoder output as its patch token sequence (replacing the usual
       pixel patch embeddings).

    **Integration decisions**

    * ``input_bands`` (from the YAML) drives which bands are requested from the
      dataset.  ``BandFilter`` passes only those bands; ``BandPadding`` zero-fills
      any bands that the dataset does not have.  The matching wavelengths are read
      from ``BAND_WAVELENGTHS`` and stored in ``carl.wv_list``.

    * ``output_layers`` (e.g. ``[1, 3, 5, 7]``) selects which of the 8 spatial
      transformer blocks to expose.  ``get_intermediate_layers`` reshapes each
      selected output to ``(B, D, H', W')`` so UPerNet can build its feature
      pyramid.  With patch_size=8 and a 224×224 input the spatial resolution is
      28×28 per level.

    * CARL performs its own **per-image standardization** in ``_standardize``
      (zero-mean, unit-variance over all channels and pixels).  Dataset-level
      ``NormalizeMeanStd`` from the PANGAEA preprocessing pipeline is therefore
      effectively overridden — the values are rescaled again internally.

    * **SAR not supported** — ``BAND_WAVELENGTHS`` only covers optical bands.
      Datasets that include SAR (sen1floods11, pastis, croptypemapping) are
      handled by limiting ``input_bands`` to ``optical`` in this YAML, so the
      SAR channels are never passed to the model.

    * **Multi-temporal** — CARL is single-temporal (``multi_temporal=False``).
      For multi-temporal datasets (pastis_optical, croptypemapping_optical,
      dynamicen) the encoder is applied per time-step and the LTAE decoder
      (``seg_upernet_mt_ltae``) aggregates across time.  The ``forward`` method
      squeezes away the T=1 dimension when present.

    Attributes:
        BAND_WAVELENGTHS: Class-level dict mapping modality → band name → central
            wavelength in µm.  Extend this dict to support new sensors.
        carl: The underlying ``CARLModel`` instance.
        patch_size: Spatial patch size inherited from the YAML config.
    """

    # Central wavelengths (µm) for Sentinel-2 optical bands.
    # These are used to build the wavelength list passed to the spectral encoder.
    # Add entries here to support additional sensors (e.g. Landsat, Gaofen).
    BAND_WAVELENGTHS: dict[str, dict[str, float]] = {
        "optical": {
            "B1":  0.443,   # Coastal aerosol
            "B2":  0.490,   # Blue
            "B3":  0.560,   # Green
            "B4":  0.665,   # Red
            "B5":  0.705,   # Vegetation Red Edge
            "B6":  0.740,   # Vegetation Red Edge
            "B7":  0.783,   # Vegetation Red Edge
            "B8":  0.842,   # NIR
            "B8A": 0.865,   # Narrow NIR
            "B9":  0.945,   # Water vapour
            "B10": 1.375,   # Cirrus
            "B11": 1.610,   # SWIR
            "B12": 2.190,   # SWIR
        }
    }

    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: dict[str, list[str]],
        output_layers: list[int],
        output_dim: int | list[int],
        download_url: str,
        embed_dim: int,
        patch_size: int,
        spec_encoder_kwargs: dict,
        spat_encoder_kwargs: dict,
        **kwargs,
    ) -> None:
        """
        Args:
            encoder_weights: Path to the CARL-SSL checkpoint (``.ckpt``).
                             Downloaded automatically from ``download_url`` if absent.
            input_size: Expected spatial input size (224 for standard PANGAEA setup).
            input_bands: Dict mapping modality → list of band names, e.g.
                         ``{"optical": ["B2", "B3", "B4", "B8A", "B11", "B12"]}``.
                         Must only contain keys present in ``BAND_WAVELENGTHS``.
            output_layers: 0-indexed spatial block indices whose outputs are returned,
                           e.g. ``[1, 3, 5, 7]`` for 4-scale UPerNet.
            output_dim: Feature dimension of each returned tensor (768 for EVA02-base).
            download_url: URL used to auto-download ``encoder_weights``.
            embed_dim: Embedding dimension of the spatial encoder (768 for EVA02-base).
            patch_size: Spatial patch size (8 in the default CARL config).
            spec_encoder_kwargs: Forwarded to ``SpectralEncoder``.
            spat_encoder_kwargs: Forwarded to ``TimmWrapper``.
        """
        super().__init__(
            model_name="carl",
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            encoder_weights=encoder_weights,
            download_url=download_url,
        )

        # Derive the ordered wavelength list from the selected input_bands.
        # The order must match the channel order that BandFilter/BandPadding
        # will produce at runtime.
        wv_list = [
            self.BAND_WAVELENGTHS[modality][band]
            for modality, bands in input_bands.items()
            for band in bands
        ]

        # Build CARLModel without loading the checkpoint here — loading is
        # deferred to load_encoder_weights so PANGAEA can control the order.
        self.carl = CARLModel(
            spec_encoder_kwargs=spec_encoder_kwargs,
            spat_encoder_kwargs=spat_encoder_kwargs,
            patch_size=patch_size,
            ssl_ckpt_path=None,
            wave_list=None,
        )
        # Inject wavelengths after construction since CARLModel normally derives
        # them from wave_list[sensor], which we bypass here.
        self.carl.wv_list = wv_list
        self.patch_size = patch_size

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract multi-scale features for UPerNet.

        Args:
            image: Dict mapping modality name → tensor.
                   Shape is (B, C, H, W) for single-temporal datasets or
                   (B, C, T=1, H, W) for multi-temporal datasets.

        Returns:
            List of ``len(output_layers)`` feature tensors, each of shape
            (B, embed_dim, H', W') where H' = W' = input_size / patch_size.
        """
        tensors = []
        for modality in self.input_bands:
            t = image[modality]
            if t.ndim == 5:
                # Multi-temporal input with T=1 — squeeze the time dimension
                # before passing to the single-temporal CARL encoder.
                t = t.squeeze(2)
            tensors.append(t)
        # Concatenate all modalities along the channel dimension
        x = torch.cat(tensors, dim=1)

        # get_intermediate_layers reshapes outputs to (B, D, H', W') spatial maps
        outputs = self.carl.get_intermediate_layers(
            x, n=self.output_layers, reshape=True
        )
        return list(outputs)

    def load_encoder_weights(self, logger: Logger) -> None:
        """Load the CARL-SSL pretrained checkpoint.

        Called automatically by the PANGAEA training framework after the model
        is instantiated.  If the checkpoint does not exist, training proceeds
        with the timm / DINOv2 initialization used by ``SpectralEncoder``.

        Args:
            logger: Logger instance provided by the training framework.
        """
        ckpt = str(self.encoder_weights)
        if not Path(ckpt).exists():
            logger.warning(
                f"CARL checkpoint not found at '{ckpt}'. "
                "Proceeding with timm / DINOv2 weight initialization."
            )
            return
        self.carl.load_ssl_checkpoint(ckpt)
