# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# mae: https://github.com/facebookresearch/mae
# slowfast: https://github.com/facebookresearch/SlowFast
# --------------------------------------------------------


from functools import partial
from typing import List, Tuple, Optional

import math
import torch
import torch.nn as nn

from .hiera import Hiera, HieraWithCLS, HieraBlock, get_2d_sincos_pos_embed
from .hiera import undo_windowing, conv_nd

import numpy as np
from torch import Tensor


def apply_fusion_head(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(head, nn.Identity):
        return x

    B, num_mask_units = x.shape[0:2]
    # Apply head, e.g [B, #MUs, My, Mx, C] -> head([B * #MUs, C, My, Mx])
    permute = [0] + [len(x.shape) - 2] + list(range(1, len(x.shape) - 2))
    x = head(x.reshape(B * num_mask_units, *x.shape[2:]).permute(permute))

    # Restore original layout, e.g. [B * #MUs, C', My', Mx'] -> [B, #MUs, My', Mx', C']
    permute = [0] + list(range(2, len(x.shape))) + [1]
    x = x.permute(permute).reshape(B, num_mask_units, *x.shape[2:], x.shape[1])
    return x


# Add this class definition at the top of your lightning module file

class ConvPredictionHead(nn.Module):
    """
    A lightweight convolutional head for MAE reconstruction.
    Replaces a single large nn.Linear with a series of ConvTranspose2d layers
    to reintroduce spatial inductive bias.
    """
    def __init__(self, decoder_embed_dim: int, out_chans: int, pred_stride: int):
        super().__init__()
        self.pred_stride = pred_stride

        # This will upsample from a 1x1 spatial map to pred_stride x pred_stride
        # We use a simple but effective design. Example for pred_stride=16:
        # 1x1 -> 2x2 -> 4x4 -> 8x8 -> 16x16
        layers = []
        in_dim = decoder_embed_dim
        num_upsamples = int(np.log2(pred_stride)) # Assumes pred_stride is a power of 2

        for i in range(num_upsamples):
            out_dim = in_dim // 2
            # The final layer must have the correct number of output channels
            if i == num_upsamples - 1:
                out_dim = out_chans

            layers.append(nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2))
            
            # Add activation and normalization between layers, but not for the last one
            if i < num_upsamples - 1:
                layers.append(nn.GELU())
                layers.append(nn.LayerNorm([out_dim, 2**(i+1), 2**(i+1)])) # Optional but helpful
            
            in_dim = out_dim
            
        self.head = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # The input x from the decoder has shape (B, L, C) or (Total Tokens, C)
        # We only process the masked tokens. Let's assume the input is already selected.
        # Input shape: (N, C) where N is the number of masked tokens.
        
        # Reshape to a 1x1 spatial feature map: (N, C, H, W)
        x = x.view(x.shape[0], x.shape[1], 1, 1)
        
        # Upsample using the convolutional layers
        x = self.head(x) # Output: (N, out_chans, pred_stride, pred_stride)
        
        # Flatten back to the expected output shape for the loss function
        # (N, out_chans * pred_stride * pred_stride)
        x = x.flatten(1)
        return x

class GroupPredictionHead(nn.Module):
    """
    Per-group prediction head for MAE decoder reconstruction.

    Instead of a single ``nn.Linear`` mapping decoder features to all
    ``C * p^2`` pixel values at once, this module uses one ``nn.Linear``
    per spectral group.  Each head predicts only its group's channels,
    and outputs are scattered into a full-sized tensor via pre-computed
    index buffers.  This ensures correct alignment with the label
    ordering from ``get_pixel_label_2d`` regardless of whether channel
    groups are contiguous or ordered.

    Args:
        decoder_embed_dim: Input feature dimension (from decoder).
        pred_stride: Spatial prediction stride (patch side length in pixels).
        channel_groups: List of channel index lists, one per spectral group.
        in_chans: Total number of input channels.  Groups must cover
            exactly ``in_chans`` channels; a ``ValueError`` is raised
            otherwise.
    """

    def __init__(
        self,
        decoder_embed_dim: int,
        pred_stride: int,
        channel_groups: List[List[int]],
        in_chans: int,
    ):
        super().__init__()
        patch_area = pred_stride ** 2
        self.patch_dim = in_chans * patch_area

        total_grouped = sum(len(g) for g in channel_groups)
        if total_grouped != in_chans:
            raise ValueError(
                f"GroupPredictionHead requires channel_groups to cover all "
                f"{in_chans} channels, but groups cover {total_grouped}. "
                f"Use decoder_prediction_mode='full' if not all channels are grouped."
            )

        self.heads = nn.ModuleList()
        self._index_buffer_names: List[str] = []

        for i, grp in enumerate(channel_groups):
            out_dim = len(grp) * patch_area
            self.heads.append(nn.Linear(decoder_embed_dim, out_dim))

            # Build index mapping: label is laid out as (C, pH, pW) flattened,
            # i.e. [ch0_px0, ch0_px1, …, ch1_px0, …].
            indices = []
            for ch in grp:
                base = ch * patch_area
                indices.extend(range(base, base + patch_area))

            buf_name = f"_group_idx_{i}"
            self.register_buffer(
                buf_name,
                torch.tensor(indices, dtype=torch.long),
                persistent=False,
            )
            self._index_buffer_names.append(buf_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros(*x.shape[:-1], self.patch_dim)
        for head, buf_name in zip(self.heads, self._index_buffer_names):
            idx = getattr(self, buf_name)
            # AMP autocast can produce float16 from nn.Linear while out is float32;
            # explicitly match dtypes to avoid index-put errors.
            out[..., idx] = head(x).to(dtype=out.dtype)
        return out


class MaskedAutoencoderHiera(HieraWithCLS):
    """Masked Autoencoder with Hiera backbone"""

    def __init__(
        self,
        in_chans: int = 3,
        patch_stride: Tuple[int, ...] = (4, 4),
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        pos_embed_type: str = "learnable",
        multiscale_fusion: bool = True,
        use_cls_token: bool = True,
        norm_pix: bool = True,
        **kwdargs,
    ):
        super().__init__(
            in_chans=in_chans,
            patch_stride=patch_stride,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            pos_embed_type=pos_embed_type,
            use_cls_token=use_cls_token,
            **kwdargs,
        )

        del self.norm
        self.multiscale_fusion = multiscale_fusion
        self.norm_pix = norm_pix
        encoder_dim_out = self.blocks[-1].dim_out
        self.encoder_norm = norm_layer(encoder_dim_out)
        self.mask_unit_spatial_shape_final = [
            i // s ** (self.q_pool) for i, s in zip(self.mask_unit_size, self.q_stride)
        ]
        self.tokens_spatial_shape_final = [
            i // s ** (self.q_pool)
            for i, s in zip(self.tokens_spatial_shape, self.q_stride)
        ]
        # --------------------------------------------------------------------------
        # Multi-scale fusion heads
        curr_mu_size = self.mask_unit_size
        self.multi_scale_fusion_heads = nn.ModuleList()

        for i in self.stage_ends[: self.q_pool]:  # resolution constant after q_pool
            kernel = [
                i // s for i, s in zip(curr_mu_size, self.mask_unit_spatial_shape_final)
            ]
            curr_mu_size = [i // s for i, s in zip(curr_mu_size, self.q_stride)]
            self.multi_scale_fusion_heads.append(
                conv_nd(len(self.q_stride))(
                    self.blocks[i].dim_out,
                    encoder_dim_out,
                    kernel_size=kernel,
                    stride=kernel,
                )
            )
        self.multi_scale_fusion_heads.append(nn.Identity())  # final stage, no transform

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim_out, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        

        if pos_embed_type == "learnable":
            print("MAE Decoder: Using learnable positional embeddings.")
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, math.prod(self.tokens_spatial_shape_final), decoder_embed_dim)
            )
            nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        elif pos_embed_type == "sincos":
            print("MAE Decoder: Using fixed sine-cosine positional embeddings.")
            decoder_pos_embed_val = get_2d_sincos_pos_embed(decoder_embed_dim, self.tokens_spatial_shape_final)
            # Register as a non-trainable buffer
            self.register_buffer('decoder_pos_embed', decoder_pos_embed_val, persistent=True)

        self.decoder_blocks = nn.ModuleList(
            [
                HieraBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_stride = patch_stride[-1] * (
            self.q_stride[-1] ** self.q_pool
        )  # patch stride of prediction

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (self.pred_stride ** min(2, len(self.q_stride))) * in_chans,
        )  # predictor
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        if isinstance(self.decoder_pos_embed, torch.nn.Parameter) and self.decoder_pos_embed.requires_grad:
            nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._mae_init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _mae_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pixel_label_2d(
        self, input_img: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*
        input_img = input_img.permute(0, 2, 3, 1)

        size = self.pred_stride
        label = input_img.unfold(1, size, size).unfold(2, size, size)
        label = label.flatten(1, 2).flatten(2)
        label = label[mask]
        if self.norm_pix:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def get_pixel_label_3d(
        self, input_vid: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*

        # We use time strided loss, only take the first frame from each token
        input_vid = input_vid[:, :, ::self.patch_stride[0], :, :]

        size = self.pred_stride
        label = input_vid.unfold(3, size, size).unfold(4, size, size)
        label = label.permute(0, 2, 3, 4, 5, 6, 1)  # Different from 2d, mistake during training lol
        label = label.flatten(1, 3).flatten(2)
        label = label[mask]

        if self.norm_pix:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label


    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float, mask: Optional[torch.Tensor] = None
    ) ->  Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if mask is None:
            mask = self.get_random_mask(x, mask_ratio)  # [B, #MUs_all]

        # Get multi-scale representations from encoder
        _, intermediates = super().forward(x, mask, return_intermediates=True)
                
        if self.multiscale_fusion:
            # Resolution unchanged after q_pool stages, so skip those features
            intermediates_to_fuse = intermediates[: self.q_pool] + intermediates[-1:]

            # Multi-scale fusion
            x_out = 0.0
            for head, interm_x in zip(self.multi_scale_fusion_heads, intermediates_to_fuse):
                x_out += apply_fusion_head(head, interm_x)
        else:
            # Simplified single-scale logic (like standard MAE)
            x_out = intermediates[-1]

        x_out = self.encoder_norm(x_out)

        return x_out, mask

    def forward_decoder(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed tokens
        x = self.decoder_embed(x)

        # Combine visible and mask tokens

        # x: [B, #MUs, *mask_unit_spatial_shape_final, encoder_dim_out]
        # mask: [B, #MUs_all]
        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)
        mask_tokens = self.mask_token.view(
            (1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,)
        )
        mask = mask.reshape(mask.shape + (1,) * len(x.shape[2:]))
        mask = mask.expand((-1,) * 2 + x.shape[2:]).bool()
        x_dec[mask] = x.flatten()
        x_dec = ~mask * mask_tokens + mask * x_dec

        # Get back spatial order
        x = undo_windowing(
            x_dec,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )
        mask = undo_windowing(
            mask[..., 0:1],
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        mask = mask.view(x.shape[0], -1)

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        return x, mask

    def forward_loss(
        self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note: in mask, 0 is *visible*, 1 is *masked*

        x: e.g. [B, 3, H, W]
        pred: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        label: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        """
        if len(self.q_stride) == 2:
            label = self.get_pixel_label_2d(x, mask)
        elif len(self.q_stride) == 3:
            label = self.get_pixel_label_3d(x, mask)
        else:
            raise NotImplementedError

        pred = pred[mask]
        loss = (pred - label) ** 2

        return loss.mean(), pred, label

    
    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.6,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        latent, mask = self.forward_encoder(x, mask_ratio, mask=mask)
        pred, pred_mask = self.forward_decoder(
            latent, mask
        )  # pred_mask is mask at resolution of *prediction*

        # Toggle mask, to generate labels for *masked* tokens
        return *self.forward_loss(x, pred, ~pred_mask), mask



# Make sure _SpectralStageEncoder is defined in the same file or imported
from .two_stages_hiera import _SpectralStageEncoder
import yaml
from typing import List, Optional


class MaskedAutoencoderTwoStagesHiera(MaskedAutoencoderHiera):
    """
    Masked Autoencoder for the Two-Stages (Spectral-Spatial) Hiera model.
    Supports loading spectral groups from YAML or custom lists.
    """
    def __init__(
        self,
        # Spectral Args
        spectral_groups: int = 1,
        spec_depth: int = 1,
        spec_num_heads: int = 1,
        spectral_pos_embed_type: str = "learnable",
        spectral_pooling: str = "cls",
        spectral_token_dim: Optional[int] = None,
        spectral_fusion_dim: Optional[int] = None,
        spectral_fusion_heads: Optional[int] = None,
        
        # Grouping Args
        custom_spectral_groups: Optional[List[List[int]]] = None,
        spectral_groups_file: Optional[str] = None,

        # Decoder prediction mode
        decoder_prediction_mode: str = "per_group",  # "full" | "per_group"

        # Standard Hiera Args
        **kwargs
    ):
        # 1. Initialize Parent
        super().__init__(**kwargs)

        print("Adapting Hiera MAE for Spectral-Spatial (TwoStages) input...")

        # 2. Extract Config
        in_chans = kwargs.get("in_chans")
        target_embed_dim = kwargs.get("embed_dim") # Output to Spatial Hiera
        patch_stride = kwargs.get("patch_stride")
        patch_kernel = kwargs.get("patch_kernel")
        patch_padding = kwargs.get("patch_padding")
        img_size = kwargs.get("input_size")

        # 3. Channel Grouping Logic (Priority: Config List > YAML File > Auto Split)
        channel_groups: List[List[int]] = []

        if custom_spectral_groups is not None:
            print(f"MAE: Using custom spectral groups from config list (Groups: {len(custom_spectral_groups)}).")
            channel_groups = custom_spectral_groups

        elif spectral_groups_file is not None:
            print(f"MAE: Loading spectral groups from file: {spectral_groups_file}")
            try:
                with open(spectral_groups_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                if isinstance(data, list):
                    channel_groups = data
                elif isinstance(data, dict) and 'groups' in data:
                    channel_groups = data['groups']
                else:
                    raise ValueError(f"YAML file must contain a list of lists or a 'groups' key.")
                print(f"MAE: Loaded {len(channel_groups)} groups from file.")
            except Exception as e:
                raise RuntimeError(f"Failed to load spectral groups file: {e}")

        else:
            # Automatic contiguous split
            if spectral_groups <= 0: raise ValueError("spectral_groups must be >= 1")
            base = in_chans // spectral_groups
            rem = in_chans % spectral_groups
            start = 0
            for i in range(spectral_groups):
                sz = base + (1 if i < rem else 0)
                channel_groups.append(list(range(start, start + sz)))
                start += sz

        # 4. Validation
        total_grouped_chans = sum(len(g) for g in channel_groups)
        if total_grouped_chans > in_chans:
             raise ValueError(f"Groups define {total_grouped_chans} channels, but input has {in_chans}.")
        if total_grouped_chans < in_chans:
            print(f"Notice: MAE will use {total_grouped_chans}/{in_chans} channels. "
                  f"{in_chans - total_grouped_chans} bands will be ignored.")

        self.channel_groups = channel_groups

        # 5. Overwrite PatchEmbed
        self.patch_embed = _SpectralStageEncoder(
            img_size=img_size,
            patch_stride=patch_stride,
            patch_kernel=patch_kernel,
            patch_padding=patch_padding,
            channel_groups=channel_groups,
            
            # Dimensions
            embed_dim=target_embed_dim,    # Output (to Spatial Hiera)
            
            # Transformer Specs
            spec_depth=spec_depth,
            spec_num_heads=spec_num_heads,
            mlp_ratio=kwargs.get("mlp_ratio", 4.0),
            spectral_pos_embed_type=spectral_pos_embed_type,
            pooling=spectral_pooling,
            spectral_token_dim=spectral_token_dim,
            spectral_fusion_dim=spectral_fusion_dim,
            spectral_fusion_heads=spectral_fusion_heads,
        )

        # 6. Optionally overwrite decoder prediction head
        decoder_prediction_mode = decoder_prediction_mode.lower()
        if decoder_prediction_mode not in {"full", "per_group"}:
            raise ValueError(
                f"Unknown decoder_prediction_mode '{decoder_prediction_mode}'. "
                f"Use 'full' or 'per_group'."
            )
        self.decoder_prediction_mode = decoder_prediction_mode

        if self.decoder_prediction_mode == "per_group":
            self.decoder_pred = GroupPredictionHead(
                decoder_embed_dim=kwargs["decoder_embed_dim"],
                pred_stride=self.pred_stride,
                channel_groups=self.channel_groups,
                in_chans=in_chans,
            )
            print(f"Replaced decoder_pred with GroupPredictionHead ({len(self.channel_groups)} groups).")
        else:
            print("Keeping single-linear decoder_pred (full mode).")


import torch
import torch.nn as nn
from logging import Logger
from typing import List, Optional

from .base import Encoder
from .two_stages_hiera_mae import MaskedAutoencoderTwoStagesHiera


def _resolve_norm(norm_layer):
    if isinstance(norm_layer, str):
        norm_layer = norm_layer.lower()
        if norm_layer == "layernorm":
            return nn.LayerNorm
        if norm_layer == "batchnorm":
            return nn.BatchNorm2d
        raise ValueError(f"Unknown norm layer: {norm_layer}")
    return norm_layer


class MaskedAutoencoderTwoStagesHIERA_Encoder(Encoder):
    """
    Hydra-compatible encoder wrapper for
    MaskedAutoencoderTwoStagesHiera.
    """

    def __init__(
        self,

        # -------------------------------------------------
        # Required by repository
        # -------------------------------------------------
        encoder_weights,
        input_size,
        input_bands,
        output_layers,
        output_dim,
        download_url=None,

        # -------------------------------------------------
        # Model params from config
        # -------------------------------------------------
        in_chans=None,
        spectral_groups=1,
        spec_depth=1,
        spec_num_heads=1,
        spectral_pos_embed_type="learnable",
        spectral_pooling="cls",
        spectral_groups_file=None,

        embed_dim=192,
        num_heads=2,
        stages=(2, 6, 10),
        q_pool=2,
        q_stride=(2, 2),
        mask_unit_size=(4, 4),
        mask_unit_attn=(True, True, False),
        dim_mul=2.0,
        head_mul=2.0,
        multiscale_fusion=True,
        pos_embed_type="learnable",
        patch_kernel=(7, 7),
        patch_stride=(6, 6),
        patch_padding=(3, 3),
        mlp_ratio=4.0,
        drop_path_rate=0.0,
        head_dropout=0.0,
        head_init_scale=0.001,
        sep_pos_embed=False,
        patch_size=2,
        decoder_embed_dim=512,
        decoder_depth=1,
        decoder_num_heads=16,
        use_cls_token=True,

        norm_layer="LayerNorm",

        **kwargs,
    ):

        # -------------------------------------------------
        # Base encoder init
        # -------------------------------------------------
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

        # Resolve input channels
        self.in_chans = (
            in_chans
            if in_chans is not None
            else len(input_bands["optical"])
        )

        norm_layer = _resolve_norm(norm_layer)

        # -------------------------------------------------
        # Build Two-Stages MAE backbone
        # -------------------------------------------------
        self.backbone = MaskedAutoencoderTwoStagesHiera(

            # Core spatial Hiera args
            input_size=(input_size, input_size),
            in_chans=self.in_chans,
            embed_dim=embed_dim,
            num_heads=num_heads,
            stages=stages,
            q_pool=q_pool,
            q_stride=q_stride,
            mask_unit_size=mask_unit_size,
            mask_unit_attn=mask_unit_attn,
            dim_mul=dim_mul,
            head_mul=head_mul,
            multiscale_fusion=multiscale_fusion,
            pos_embed_type=pos_embed_type,
            patch_kernel=patch_kernel,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            head_dropout=head_dropout,
            head_init_scale=head_init_scale,
            sep_pos_embed=sep_pos_embed,
            patch_size=patch_size,
            use_cls_token=use_cls_token,
            norm_layer=norm_layer,

            # Spectral stage args
            spectral_groups=spectral_groups,
            spec_depth=spec_depth,
            spec_num_heads=spec_num_heads,
            spectral_pos_embed_type=spectral_pos_embed_type,
            spectral_pooling=spectral_pooling,
            spectral_groups_file=spectral_groups_file,

            # Decoder args
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,

            **kwargs,
        )

        self.stage_indices = self.backbone.stage_ends

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, images):

        x = images["optical"].squeeze(2)  # (B,C,H,W)

        # Patch embedding
        x = self.backbone.patch_embed(x)

        intermediates = []

        for i, blk in enumerate(self.backbone.blocks):
            x = blk(x)

            if i in self.backbone.stage_ends:
                # x is (B, N, C)
                B, N, C = x.shape

                # Recover spatial resolution
                H = W = int(N ** 0.5)  # assumes square

                feat = x.view(B, H, W, C)
                intermediates.append(feat)

        x = self.backbone.encoder_norm(x)

        outputs = []
        for i, feat in enumerate(intermediates):
            if i in self.output_layers:
                feat = feat.permute(0, 3, 1, 2).contiguous()
                outputs.append(feat)

        return outputs
    
    # ---------------------------------------------------------
    # Weight Loading
    # ---------------------------------------------------------
    def load_encoder_weights(self, logger: Logger):

        if self.encoder_weights is None:
            return

        ckpt = torch.load(self.encoder_weights, map_location="cpu")

        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        new_ckpt = {}
        for k, v in ckpt.items():
            # Strip 'module.' prefix from DDP checkpoints
            if k.startswith("module."):
                k = k[7:]

            # Skip decoder/mask token
            if k.startswith("decoder") or "mask_token" in k:
                continue

            # Rename final encoder LayerNorm
            if k == "norm.weight":
                new_ckpt["encoder_norm.weight"] = v
                continue
            if k == "norm.bias":
                new_ckpt["encoder_norm.bias"] = v
                continue

            # Keep everything else
            new_ckpt[k] = v

        missing, unexpected = self.backbone.load_state_dict(new_ckpt, strict=False)

        logger.warning(f"Missing keys: {missing}")
        logger.warning(f"Unexpected keys: {unexpected}")
