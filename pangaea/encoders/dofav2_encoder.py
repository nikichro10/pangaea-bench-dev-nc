from functools import partial
from logging import Logger
from pathlib import Path
import math

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from pangaea.encoders.base import Encoder
from pangaea.encoders.dofa_encoder import Dynamic_MLP_OFA


class DOFAv2_Encoder(Encoder):
    """
    DOFAv2 encoder matching the official implementation structure:
      - Uses timm VisionTransformer with dynamic_img_size=True for native
        positional embedding interpolation at any input resolution.
      - LayerScale enabled via init_values=1e-5.
      - patch_embed.proj deleted; Dynamic_MLP_OFA used for patch embedding.
      - Intermediate features collected pre-norm (matches official forward).
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_bands: dict[str, list[str]],
        input_size: int,
        embed_dim: int,
        output_dim: int | list[int],
        output_layers: int | list[int],
        wave_list: dict[str, dict[str, float]],
        download_url: str,
        patch_size: int = 14,
        depth: int = 12,
        num_heads: int = 12,
        wv_planes: int = 128,
        mlp_ratio: float = 4.0,
        use_norm: bool = True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            model_name="dofav2_encoder",
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

        self.output_layers = output_layers
        self.patch_size = patch_size
        self.wave_list = wave_list
        self.use_norm = use_norm
        self.wv_list = [
            self.wave_list[m][bi] for m, b in self.input_bands.items() for bi in b
        ]

        # Dynamic_MLP_OFA replaces the standard conv patch embedding.
        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=wv_planes, inter_dim=128, kernel_size=patch_size, embed_dim=embed_dim
        )

        # VisionTransformer provides cls_token, pos_embed, blocks, and norm.
        # dynamic_img_size=True makes _pos_embed accept (B, H, W, C) and
        # interpolate pos_embed automatically for non-224 inputs.
        # init_values=1e-5 enables LayerScale, required by DOFAv2 checkpoints.
        self.model = VisionTransformer(
            img_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
            norm_layer=norm_layer,
        )
        # Remove the conv proj; we use Dynamic_MLP_OFA instead.
        # The PatchEmbed object itself is kept so _pos_embed can read grid_size.
        del self.model.patch_embed.proj

    def forward(self, image):
        x = [image[m].squeeze(2) for m in self.input_bands.keys()]
        x = torch.cat(x, dim=1)

        self.waves = torch.tensor(self.wv_list, device=x.device).float()
        x, _ = self.patch_embed(x, self.waves)

        # Reshape to (B, H, W, C) so _pos_embed can infer grid size and
        # interpolate pos_embed when the input differs from input_size.
        B, HW, C = x.shape
        hw = int(math.sqrt(HW))
        x = x.view(B, hw, hw, C)

        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)

        # Collect intermediate features pre-norm, matching the official forward.
        output = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.output_layers:
                out = (
                    x[:, 1:]
                    .permute(0, 2, 1)
                    .view(B, -1, hw, hw)
                    .contiguous()
                )
                output.append(out)

        self.model.norm(x)
        return output

    def set_wv_list_from_data_bands(self, data_bands: dict[str, list[str]]) -> None:
        """Restrict wv_list to bands actually present in the dataset.
        Call this when BandPadding is NOT used so the encoder receives only real
        bands with their true wavelengths rather than zero-padded phantom channels.
        """
        wv_list = [
            self.wave_list[m][bi]
            for m, b in self.input_bands.items()
            for bi in b
            if m in data_bands and bi in data_bands[m]
        ]
        if not wv_list:
            raise ValueError("No wavelengths remain after filtering to available data bands.")
        self.wv_list = wv_list

    def load_encoder_weights(self, logger: Logger) -> None:
        pretrained_model = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)
        # No prefix stripping needed: self.model = VisionTransformer, so
        # named_parameters() naturally yields 'model.blocks.*', 'model.pos_embed', etc.,
        # which directly match the checkpoint keys.
        k = pretrained_model.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}
        for name, param in self.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained_model[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained_model[name].shape)
            else:
                pretrained_encoder[name] = pretrained_model[name]

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)
