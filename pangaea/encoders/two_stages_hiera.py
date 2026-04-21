import contextlib
import torch
import torch.nn as nn
import math
from typing import Tuple, List, Optional
from logging import Logger

# Import the necessary components from your Hiera implementation
#from .hiera import Hiera, HieraBlock
import yaml
from .base import Encoder


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
#
# Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles
#
# Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan,
# Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed,
# Judy Hoffman, Jitendra Malik, Yanghao Li, Christoph Feichtenhofer.
#
# Paper: https://arxiv.org/abs/2306.00989/
#
# References:
# slowfast: https://github.com/facebookresearch/SlowFast
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

import math
from typing import List, Tuple, Optional, Type, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence

#from .hiera import Hiera, HieraBlock

#from timm.models.layers import DropPath, Mlp

from .hiera import (
    Hiera,
    HieraWithCLS,
    HieraBlock,
    do_masked_conv,
    undo_windowing,
    get_1d_sincos_pos_embed_from_grid,  # <-- we'll use this
)

import numpy as np

# timm spectral block
try:
    from timm.models.vision_transformer import Block as ViTBlock
except Exception as e:
    raise ImportError("Please install timm (`pip install timm`) to use the Spectral Stage.") from e

import yaml

def _build_1d_sincos_pos_embed(
    embed_dim: int,
    length: int,
    cls_token: bool = True,
) -> torch.Tensor:
    """
    Build a (1, length(+1), D) sine–cosine 1D positional embedding tensor.
    - The CLS row is zeros (common choice) when cls_token=True.
    """
    # positions 0..length-1
    grid = np.arange(length, dtype=np.float32)
    pos = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)  # (length, D)
    if cls_token:
        cls_row = np.zeros((1, embed_dim), dtype=pos.dtype)
        pos = np.concatenate([cls_row, pos], axis=0)  # (1+length, D)
    return torch.from_numpy(pos).unsqueeze(0)  # (1, 1+length, D)


class _SpectralStageEncoder(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        patch_stride: Tuple[int, int],
        patch_kernel: Tuple[int, int], 
        patch_padding: Tuple[int, int],
        channel_groups: List[List[int]],
        embed_dim: int,
        spec_depth: int,
        spec_num_heads: int,
        mlp_ratio: float,
        spectral_pos_embed_type: str = "learnable",  # 'learnable' | 'sincos'
        pooling: str = "cls",                        # 'cls' | 'mean' | 'max' | 'projected_attention'
        spectral_token_dim: Optional[int] = None,
        spectral_fusion_dim: Optional[int] = None,
        spectral_fusion_heads: Optional[int] = None,
        qkv_bias: bool = True,
        norm_layer: nn.LayerNorm = nn.LayerNorm,
        # SDPA control
        sdp_policy: str = "auto",
        flash_max_seqlen: int = 4096,
        sdp_max_seqlen_dim: int = 65535,
    ):
        super().__init__()
        self.output_dim = int(embed_dim)
        self.D = int(spectral_token_dim or embed_dim)
        self.fusion_dim = int(spectral_fusion_dim or embed_dim)
        self.fusion_heads = int(spectral_fusion_heads or spec_num_heads)
        if self.D % spec_num_heads != 0:
            raise ValueError(
                f"spectral_token_dim ({self.D}) must be divisible by spec_num_heads ({spec_num_heads})."
            )
        self.img_size = tuple(img_size)
        self.patch_stride = tuple(patch_stride)
        self.patch_kernel = tuple(patch_kernel)    
        self.patch_padding = tuple(patch_padding)  
    
        self.channel_groups = channel_groups
        self.G = len(channel_groups)
        self.spec_num_heads = spec_num_heads

        # --- Pooling Strategy Configuration ---
        pooling_aliases = {
            "proj_attn": "projected_attention",
            "projected_attention": "projected_attention",
        }
        pooling = pooling_aliases.get(pooling, pooling)
        if pooling not in {"cls", "mean", "max", "projected_attention"}:
            raise ValueError(
                "Unknown pooling mode: "
                f"{pooling}. Use one of ['cls', 'mean', 'max', 'projected_attention']."
            )
        if pooling == "projected_attention" and self.fusion_dim % self.fusion_heads != 0:
            raise ValueError(
                f"spectral_fusion_dim ({self.fusion_dim}) must be divisible by "
                f"spectral_fusion_heads ({self.fusion_heads}) for projected_attention."
            )
        self.pooling = pooling

        # --- SDPA Config ---
        if sdp_policy not in {"auto", "math", "flash"}:
            raise ValueError(f"Unknown sdp_policy: {sdp_policy}")
        self.sdp_policy = sdp_policy
        self.flash_max_seqlen = flash_max_seqlen
        self.sdp_max_seqlen_dim = sdp_max_seqlen_dim

        # 1. Group Embedding
        self.patch_embed_list = nn.ModuleList([
            nn.Conv2d(
                in_channels=len(grp),
                out_channels=self.D,
                kernel_size=self.patch_kernel,
                stride=self.patch_stride,
                padding=self.patch_padding,
                bias=True,
            ) for grp in self.channel_groups
        ])

        # 2. CLS Token (Only create if needed to avoid DDP unused param errors)
        self.spec_cls_token = None
        if self.pooling == "cls":
            self.spec_cls_token = nn.Parameter(torch.zeros(1, 1, self.D))
            nn.init.trunc_normal_(self.spec_cls_token, std=0.02)
        self.spectral_mask_token = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.trunc_normal_(self.spectral_mask_token, std=0.02)

        # 3. Position Embeddings
        # If pooling='cls', sequence length is G + 1. If 'mean'/'max', it's G.
        seq_len = self.G + 1 if self.pooling == "cls" else self.G
        
        if spectral_pos_embed_type == "learnable":
            self.spec_pos_embed = nn.Parameter(torch.zeros(1, seq_len, self.D))
            nn.init.trunc_normal_(self.spec_pos_embed, std=0.02)
        elif spectral_pos_embed_type == "sincos":
            # Pass cls_token=False if we aren't using one
            has_cls = (self.pooling == "cls")
            pos = _build_1d_sincos_pos_embed(self.D, self.G, cls_token=has_cls).float()
            self.register_buffer("spec_pos_embed", pos, persistent=True)
        else:
            raise ValueError(f"Unknown spectral_pos_embed_type: {spectral_pos_embed_type}")

        # 4. Transformer Blocks
        self.spec_blocks = nn.ModuleList([
            ViTBlock(dim=self.D, num_heads=spec_num_heads, mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(spec_depth)
        ])
        self.spec_norm = norm_layer(self.D)

        # 5. Optional projected attention fusion: project first, fuse at higher dim.
        if self.pooling == "projected_attention":
            self.fusion_input_proj = nn.Linear(self.D, self.fusion_dim)
            self.fusion_query_token = nn.Parameter(torch.zeros(1, 1, self.fusion_dim))
            self.fusion_attn = nn.MultiheadAttention(
                embed_dim=self.fusion_dim,
                num_heads=self.fusion_heads,
                batch_first=True,
            )
            self.fusion_norm = norm_layer(self.fusion_dim)
            if spectral_pos_embed_type == "learnable":
                self.fusion_pos_embed = nn.Parameter(torch.zeros(1, self.G, self.fusion_dim))
                nn.init.trunc_normal_(self.fusion_pos_embed, std=0.02)
            else:
                fusion_pos = _build_1d_sincos_pos_embed(
                    self.fusion_dim, self.G, cls_token=False
                ).float()
                self.register_buffer("fusion_pos_embed", fusion_pos, persistent=True)
            nn.init.trunc_normal_(self.fusion_query_token, std=0.02)
        else:
            self.fusion_input_proj = None
            self.fusion_query_token = None
            self.fusion_attn = None
            self.fusion_norm = None
            self.fusion_pos_embed = None

        # Avoid oversized SDPA launches on very large (B*L) batches.
        self._fusion_attn_chunk_size = 16384
        pooled_dim = self.fusion_dim if self.pooling == "projected_attention" else self.D
        self.output_proj = nn.Identity() if pooled_dim == self.output_dim else nn.Linear(pooled_dim, self.output_dim)
        
        self.apply(self._init_weights)

    def _run_fusion_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        n = q.shape[0]
        if n <= self._fusion_attn_chunk_size:
            if q.device.type == "cuda":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                ):
                    out, _ = self.fusion_attn(q, k, v, need_weights=False)
                    return out
            out, _ = self.fusion_attn(q, k, v, need_weights=False)
            return out

        out_chunks: List[torch.Tensor] = []
        for start in range(0, n, self._fusion_attn_chunk_size):
            end = min(start + self._fusion_attn_chunk_size, n)
            q_i = q[start:end]
            k_i = k[start:end]
            v_i = v[start:end]
            if q.device.type == "cuda":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                ):
                    out_i, _ = self.fusion_attn(q_i, k_i, v_i, need_weights=False)
            else:
                out_i, _ = self.fusion_attn(q_i, k_i, v_i, need_weights=False)
            out_chunks.append(out_i)
        return torch.cat(out_chunks, dim=0)

    def _sdp_context(self, seq_len: int, batch_size: int, dtype: torch.dtype, device: torch.device):
        if device.type != "cuda":
            return contextlib.nullcontext()
        if self.sdp_policy == "math":
            return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
        if self.sdp_policy == "flash":
            return torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        
        head_dim = self.D // self.spec_num_heads
        dtype_ok = dtype in (torch.float16, torch.bfloat16)
        # Flash / mem-efficient CUDA kernels dispatch batch*heads along grid
        # y/z dims which are capped at 65535.  Fall back to math when the
        # product would exceed that limit.
        batch_heads_ok = (batch_size * self.spec_num_heads) <= self.sdp_max_seqlen_dim
        flash_ok = (
            dtype_ok and
            head_dim <= 128 and
            seq_len <= self.flash_max_seqlen and
            seq_len <= self.sdp_max_seqlen_dim and
            batch_heads_ok
        )
        if flash_ok:
            return torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        else:
            return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        spectral_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        group_tokens = []
        for i, grp in enumerate(self.channel_groups):
            gx = do_masked_conv(x[:, grp], self.patch_embed_list[i], mask)
            gx = gx.flatten(2).transpose(1, 2)  # (B, L, D)
            group_tokens.append(gx)

        x_grp = torch.stack(group_tokens, dim=1)  # (B, G, L, D)
        B, G, L, D = x_grp.shape
        if L == 0:
            raise ValueError(f"L=0 with input {H}x{W}")

        # Reshape to spectral sequence: (B*L, G, D)
        x_spec = x_grp.permute(0, 2, 1, 3).reshape(B * L, G, D)

        if spectral_mask is not None:
            expected_shape = (B, L, G)
            if tuple(spectral_mask.shape) != expected_shape:
                raise ValueError(
                    f"spectral_mask must have shape {expected_shape}, got {tuple(spectral_mask.shape)}"
                )
            spectral_keep = spectral_mask.to(dtype=torch.bool, device=x_spec.device).reshape(B * L, G, 1)
            mask_token = self.spectral_mask_token.expand(B * L, G, -1).to(x_spec.dtype)
            x_spec = torch.where(spectral_keep, x_spec, mask_token)

        # Conditionally add CLS token
        if self.pooling == "cls":
            cls = self.spec_cls_token.expand(B * L, -1, -1)
            x_spec = torch.cat([cls, x_spec], dim=1) # (B*L, G+1, D)
        
        # Add Pos Embed
        x_spec = x_spec + self.spec_pos_embed

        # Run Transformer
        seq_len = x_spec.shape[1]
        with self._sdp_context(seq_len, B * L, x_spec.dtype, x_spec.device):
            for blk in self.spec_blocks:
                x_spec = blk(x_spec)

        x_spec = self.spec_norm(x_spec)

        # Aggregation Strategy
        if self.pooling == "cls":
            # Take index 0
            out_feats = x_spec[:, 0]
        elif self.pooling == "mean":
            # Average over the G tokens (dim 1)
            out_feats = x_spec.mean(dim=1)
        elif self.pooling == "max":
            # Max over the G tokens
            out_feats = x_spec.max(dim=1).values
        elif self.pooling == "projected_attention":
            x_fused = self.fusion_input_proj(x_spec)
            x_fused = x_fused + self.fusion_pos_embed
            if G == 1:
                out_feats = self.fusion_norm(x_fused[:, 0])
            else:
                q = self.fusion_query_token.expand(B * L, -1, -1)
                fused = self._run_fusion_attention(q, x_fused, x_fused)
                out_feats = self.fusion_norm(fused.squeeze(1))
        else:
            # Should not happen due to init check
            out_feats = x_spec[:, 0] 

        out_feats = self.output_proj(out_feats)

        # Reshape back to Spatial Grid
        patch_feats = out_feats.view(B, L, self.output_dim)
        return patch_feats

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            w = m.weight
            nn.init.xavier_uniform_(w.view(w.shape[0], -1))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


class TwoStagesHiera(HieraWithCLS):
    """
    Hiera with a learnable/fixed spectral stage.
    Supports:
    1. Automatic grouping (spectral_groups=int)
    2. Explicit list of lists (custom_spectral_groups=[[...]])
    3. External YAML file (spectral_groups_file="path/to/groups.yaml")
    """
    def __init__(
        self,
        spectral_groups: int = 1,
        spec_depth: int = 1,
        spec_num_heads: int = 1,
        spectral_pos_embed_type: str = "learnable",
        use_cls_token: bool = True,
        spectral_pooling: str = "cls",
        spectral_token_dim: Optional[int] = None,
        spectral_fusion_dim: Optional[int] = None,
        spectral_fusion_heads: Optional[int] = None,
        custom_spectral_groups: Optional[List[List[int]]] = None,
        # NEW ARGUMENT
        spectral_groups_file: Optional[str] = None, 
        **kwargs,
    ):
        super().__init__(use_cls_token=use_cls_token, **kwargs)
        print("Adapting Hiera for Hyperspectral input (TwoStages)...")

        in_chans = kwargs["in_chans"]
        patch_stride = kwargs["patch_stride"]
        patch_kernel = kwargs["patch_kernel"]
        patch_padding = kwargs["patch_padding"]
        # --- GROUPING LOGIC ---
        channel_groups: List[List[int]] = []

        # Priority 1: Direct List passed in config
        if custom_spectral_groups is not None:
            print(f"Using custom spectral groups from config list (Groups: {len(custom_spectral_groups)}).")
            channel_groups = custom_spectral_groups

        # Priority 2: External YAML File
        elif spectral_groups_file is not None:
            print(f"Loading spectral groups from file: {spectral_groups_file}")
            try:
                with open(spectral_groups_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Handle cases where yaml is a list [[...]] or dict {'groups': [[...]]}
                if isinstance(data, list):
                    channel_groups = data
                elif isinstance(data, dict) and 'groups' in data:
                    channel_groups = data['groups']
                else:
                    raise ValueError(f"YAML file must contain a list of lists or a 'groups' key. Got: {type(data)}")
                
                print(f"Loaded {len(channel_groups)} groups from file.")
                
            except Exception as e:
                raise RuntimeError(f"Failed to load spectral groups file: {e}")

        # Priority 3: Automatic Split
        else:
            if spectral_groups <= 0:
                raise ValueError("spectral_groups must be >= 1")

            base = in_chans // spectral_groups
            rem = in_chans % spectral_groups
            start = 0
            for i in range(spectral_groups):
                sz = base + (1 if i < rem else 0)
                channel_groups.append(list(range(start, start + sz)))
                start += sz

        # --- VALIDATION ---
        # Sanity check to help debug mismatch between config in_chans and the file
        total_grouped_chans = sum(len(g) for g in channel_groups)
        # Note: We use <= because you might intentionally drop bands (e.g. water absorption)
        if total_grouped_chans > in_chans:
             raise ValueError(f"Groups define {total_grouped_chans} channels, but input has only {in_chans}.")
        if total_grouped_chans < in_chans:
            print(f"Notice: Groups use {total_grouped_chans}/{in_chans} channels. "
                  f"{in_chans - total_grouped_chans} bands will be ignored.")

        # Replace patch_embed
        self.patch_embed = _SpectralStageEncoder(
            img_size=kwargs["input_size"],
            patch_stride=patch_stride,
            patch_kernel=patch_kernel,
            patch_padding=patch_padding,
            channel_groups=channel_groups,
            embed_dim=kwargs["embed_dim"],
            spec_depth=spec_depth,
            spec_num_heads=spec_num_heads,
            mlp_ratio=kwargs.get("mlp_ratio", 4.0),
            spectral_pos_embed_type=spectral_pos_embed_type,
            pooling=spectral_pooling,
            spectral_token_dim=spectral_token_dim,
            spectral_fusion_dim=spectral_fusion_dim,
            spectral_fusion_heads=spectral_fusion_heads,

        )
        print("Hiera's PatchEmbed has been replaced with the Spectral Stage Encoder.")

    # (optional) keep your get_intermediate_layers as you had it
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 0,
        reshape: bool = False,
        norm: bool = False,
    ):
        x, intermediates = self.forward(x, return_intermediates=True)
        outputs = intermediates[-n:] if isinstance(n, int) else intermediates[:]
        if reshape:
            outputs = [out.permute(0, 3, 1, 2).contiguous() for out in outputs]
        return tuple(outputs)
    

class TwoStagesHIERA_Encoder(Encoder):
    """
    TwoStagesHiera Encoder wrapper compatible with repository structure.
    Returns list of spatial feature maps from selected stages.
    """

    def __init__(
        self,
        encoder_weights,
        input_size,
        input_bands,
        embed_dim,
        output_layers,
        output_dim,
        download_url=None,
        patch_size=4,
        in_chans=None,
        stages=(2, 3, 16, 3),
        q_pool=3,
        q_stride=(2, 2),
        mask_unit_size=(8, 8),
        mask_unit_attn=(True, True, False, False),
        dim_mul=2.0,
        head_mul=2.0,
        patch_kernel=(7, 7),
        patch_stride=(4, 4),
        patch_padding=(3, 3),
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        norm_layer="LayerNorm",
        pos_embed_type="learnable",
        use_cls_token=True,
        # -------- Spectral Stage Params --------
        spectral_groups=1,
        spec_depth=1,
        spec_num_heads=1,
        spectral_pos_embed_type="learnable",
        spectral_pooling="cls",
        spectral_token_dim=None,
        spectral_fusion_dim=None,
        spectral_fusion_heads=None,
        custom_spectral_groups=None,
        spectral_groups_file=None,
        **kwargs,
    ):

        Encoder.__init__(
            self,
            model_name="two_stages_hiera_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=True,
            download_url=download_url,
        )

        self.in_chans = (
            len(input_bands["optical"]) if in_chans is None else in_chans
        )

        # --------------------------------------------------
        # Build TwoStagesHiera Backbone
        # --------------------------------------------------
        self.backbone = TwoStagesHiera(
            input_size=(input_size, input_size),
            in_chans=self.in_chans,
            embed_dim=embed_dim,
            stages=stages,
            q_pool=q_pool,
            q_stride=q_stride,
            mask_unit_size=mask_unit_size,
            mask_unit_attn=mask_unit_attn,
            dim_mul=dim_mul,
            head_mul=head_mul,
            patch_kernel=patch_kernel,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            pos_embed_type=pos_embed_type,
            use_cls_token=use_cls_token,
            # -------- Spectral --------
            spectral_groups=spectral_groups,
            spec_depth=spec_depth,
            spec_num_heads=spec_num_heads,
            spectral_pos_embed_type=spectral_pos_embed_type,
            spectral_pooling=spectral_pooling,
            spectral_token_dim=spectral_token_dim,
            spectral_fusion_dim=spectral_fusion_dim,
            spectral_fusion_heads=spectral_fusion_heads,
            custom_spectral_groups=custom_spectral_groups,
            spectral_groups_file=spectral_groups_file,
        )

        # Track stage boundaries (same behavior as original Hiera)
        self.stage_indices = self.backbone.stage_ends

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, images):

        # Expect repository input dict format
        x = images["optical"].squeeze(2)  # (B, C, H, W)

        # Get intermediate outputs from backbone
        _, intermediates = self.backbone(
            x,
            return_intermediates=True,
        )

        output = []

        for i, feat in enumerate(intermediates):
            if i in self.output_layers:
                # feat is (B, H, W, C)
                feat = feat.permute(0, 3, 1, 2).contiguous()
                output.append(feat)

        return output

    # ---------------------------------------------------------
    # Weight Loading
    # ---------------------------------------------------------
    def load_encoder_weights(self, logger: Logger) -> None:
        if self.encoder_weights is None:
            return

        ckpt = torch.load(self.encoder_weights, map_location="cpu")

        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            ckpt, strict=False
        )

        if missing_keys:
            logger.warning(f"Missing keys (not loaded): {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys (ignored): {unexpected_keys}")
