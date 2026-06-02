# Copernicus-FM encoder for PANGAEA benchmark.
# Model: "Towards a Unified Copernicus Foundation Model for Earth Vision" (ICCV 2025)
# https://github.com/zhu-xlab/Copernicus-FM
#
# Dependencies inlined to avoid requiring the original repo:
#   - FourierExpansion   (aurora/fourier.py)
#   - resize_abs_pos_embed (flexivit/utils.py)
#   - pi_resize_patch_embed (flexivit/patch_embed.py, torch.vmap replaces functorch.vmap)
#   - Dynamic_MLP_OFA_spectral (dynamic_hypernetwork.py)
#   - CopernicusFMViT    (model_vit.py)

from functools import partial
from logging import Logger
from pathlib import Path
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
from timm.models.vision_transformer import Block

from pangaea.encoders.base import Encoder
from pangaea.encoders.pos_embed import get_1d_sincos_pos_embed_from_grid_torch


# ---------------------------------------------------------------------------
# FourierExpansion — from aurora/fourier.py (Microsoft, MIT licence)
# ---------------------------------------------------------------------------

class FourierExpansion(nn.Module):
    """Fourier series-style expansion into a high-dimensional space.

    Args:
        lower: Lower wavelength bound.
        upper: Upper wavelength bound.
        assert_range: Raise if inputs fall outside [lower, upper].
    """

    def __init__(self, lower: float, upper: float, assert_range: bool = True) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range

    def forward(self, x: torch.Tensor, d: int) -> torch.Tensor:
        in_range = torch.logical_and(
            self.lower <= x.abs(),
            torch.all(x.abs() <= self.upper),
        )
        in_range_or_zero = torch.all(torch.logical_or(in_range, x == 0))
        if self.assert_range and not in_range_or_zero:
            raise AssertionError(
                f"Input not within configured range [{self.lower}, {self.upper}]."
            )
        if d % 2 != 0:
            raise ValueError("Dimensionality d must be a multiple of two.")
        x = x.double()
        wavelengths = torch.logspace(
            math.log10(self.lower), math.log10(self.upper),
            d // 2, base=10, device=x.device, dtype=x.dtype,
        )
        prod = torch.einsum("...i,j->...ij", x, 2 * np.pi / wavelengths)
        return torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1).float()


# ---------------------------------------------------------------------------
# resize_abs_pos_embed — from flexivit/utils.py
# ---------------------------------------------------------------------------

def _resize_abs_pos_embed(
    pos_embed: torch.Tensor,
    new_size,
    old_size=None,
    num_prefix_tokens: int = 1,
) -> torch.Tensor:
    """Resize absolute position embeddings via bicubic interpolation."""
    if not isinstance(new_size, tuple):
        new_size = (new_size, new_size)
    new_ntok = new_size[0] * new_size[1]

    if old_size is None:
        old_size = int(math.sqrt(pos_embed.shape[1] - num_prefix_tokens))
    if not isinstance(old_size, tuple):
        old_size = (old_size, old_size)

    if new_size == old_size:
        return pos_embed

    if num_prefix_tokens:
        posemb_prefix = pos_embed[:, :num_prefix_tokens]
        pos_embed = pos_embed[:, num_prefix_tokens:]
    else:
        posemb_prefix = None

    pos_embed = pos_embed.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    pos_embed = F.interpolate(pos_embed, size=new_size, mode="bicubic", antialias=True)
    pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, new_ntok, -1)

    if posemb_prefix is not None:
        pos_embed = torch.cat([posemb_prefix, pos_embed], dim=1)
    return pos_embed


# ---------------------------------------------------------------------------
# pi_resize_patch_embed — from flexivit/patch_embed.py
# functorch.vmap replaced with torch.vmap (available in PyTorch >= 2.1)
# ---------------------------------------------------------------------------

def _pi_resize_patch_embed(
    patch_embed: torch.Tensor,
    new_patch_size,
    interpolation: str = "bicubic",
    antialias: bool = True,
) -> torch.Tensor:
    """Resize patch embedding weights via pseudo-inverse resizing."""
    assert len(patch_embed.shape) == 4
    old_patch_size = tuple(patch_embed.shape[2:])
    if not isinstance(new_patch_size, tuple):
        new_patch_size = (new_patch_size, new_patch_size)
    if old_patch_size == new_patch_size:
        return patch_embed

    def resize(x, shape):
        return F.interpolate(x[None, None], shape, mode=interpolation, antialias=antialias)[0, 0]

    def calculate_pinv(old_shape, new_shape):
        mat = []
        for i in range(int(np.prod(old_shape))):
            basis_vec = torch.zeros(old_shape)
            basis_vec[tuple(np.unravel_index(i, old_shape))] = 1.0
            mat.append(resize(basis_vec, new_shape).reshape(-1))
        return torch.linalg.pinv(torch.stack(mat))

    pinv = calculate_pinv(old_patch_size, new_patch_size).to(patch_embed.device)

    def resample(pe):
        h, w = new_patch_size
        return rearrange(pinv @ pe.reshape(-1), "(h w) -> h w", h=h, w=w)

    return torch.vmap(torch.vmap(resample, 0, 0), 1, 1)(patch_embed)


# ---------------------------------------------------------------------------
# TransformerWeightGenerator + FCResLayer — from dynamic_hypernetwork.py
# ---------------------------------------------------------------------------

class _TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, activation="gelu",
            norm_first=False, batch_first=False, dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
        )
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        out = self.transformer_encoder(x)
        weights = self.fc_weight(out[self.wt_num:-1] + pos_wave)
        bias = self.fc_bias(out[-1])
        return weights, bias


class _FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super().__init__()
        self.w1 = nn.Linear(linear_size, linear_size)
        self.w2 = nn.Linear(linear_size, linear_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.relu(self.w2(self.relu(self.w1(x))))


# ---------------------------------------------------------------------------
# Dynamic_MLP_OFA_spectral — from dynamic_hypernetwork.py
# Extends DOFA's patch embedding with bandwidth support via FourierExpansion.
# ---------------------------------------------------------------------------

class _DynamicMLPSpectral(nn.Module):
    """Dynamic patch embedding conditioned on per-channel wavelength + bandwidth."""

    def __init__(self, wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=768):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self._num_kernel = kernel_size * kernel_size * embed_dim

        # Fourier embeddings for wavelength (100–1e9 nm) and bandwidth (1–1e9 nm)
        self.spectrum_central_expansion = FourierExpansion(100, 1e9)
        self.spectrum_bandwidth_expansion = FourierExpansion(1, 1e9)

        self.weight_generator = _TransformerWeightGenerator(wv_planes, self._num_kernel, embed_dim)
        self.scaler = 0.01
        self.fclayer = _FCResLayer(wv_planes)
        self._init_weights()

    def _init_weights(self):
        def weight_init(m):
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.weight_generator.apply(weight_init)
        self.fclayer.apply(weight_init)

    def forward(self, img_feat, wvs, bandwidths, kernel_size=None):
        """
        Args:
            img_feat: (B, C, H, W) image tensor.
            wvs: (C,) tensor of central wavelengths in nm.
            bandwidths: (C,) tensor of bandwidths in nm.
            kernel_size: override patch size for flexible patching (None = use default).
        Returns:
            x: (B, num_patches, embed_dim) patch tokens.
            waves: (C, wv_planes) wavelength embeddings.
        """
        inplanes = wvs.size(0)
        emb_central = self.spectrum_central_expansion(wvs, self.wv_planes)
        emb_bandwidth = self.spectrum_bandwidth_expansion(bandwidths, self.wv_planes)
        waves = emb_central + emb_bandwidth
        waves = self.fclayer(waves)

        weight, bias = self.weight_generator(waves)

        dynamic_weight = weight.view(inplanes, self.kernel_size, self.kernel_size, self.embed_dim)
        dynamic_weight = dynamic_weight.permute(3, 0, 1, 2)  # (embed_dim, C, k, k)

        if kernel_size is not None and kernel_size != self.kernel_size:
            dynamic_weight = _pi_resize_patch_embed(dynamic_weight, (kernel_size, kernel_size))
        else:
            kernel_size = self.kernel_size

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler
        x = F.conv2d(img_feat, weights, bias=bias, stride=kernel_size, padding=1, dilation=1)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x, waves


# ---------------------------------------------------------------------------
# CopernicusFMViT — from model_vit.py
# Stripped to spectral mode only; always returns intermediate feature maps.
# ---------------------------------------------------------------------------

class _CopernicusFMViT(nn.Module):
    """Vision Transformer backbone for Copernicus-FM (spectral input mode)."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        wv_planes=128,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        intermediate_indices=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.intermediate_indices = set(intermediate_indices or [])

        self.patch_embed = _DynamicMLPSpectral(
            wv_planes=wv_planes, kernel_size=patch_size, embed_dim=embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Fixed sin-cos positional embedding (not trained)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False,
        )

        # Metadata positional encodings (lon/lat, area, time)
        self.coord_expansion = FourierExpansion(0.0001, 720)
        self.scale_expansion = FourierExpansion(0.001, 5.1e8)
        self.time_expansion = FourierExpansion(1, 365.25, assert_range=False)

        self.coord_fc = nn.Linear(embed_dim, embed_dim)
        self.scale_fc = nn.Linear(embed_dim, embed_dim)
        self.time_fc = nn.Linear(embed_dim, embed_dim)

        # Learned fallback tokens used when metadata is nan
        self.coord_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scale_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.time_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def _coord_pos_embed(self, lons, lats, d):
        emb_lon = self.coord_expansion(lons + 180, d // 2)
        emb_lat = self.coord_expansion(lats + 90, d // 2)
        emb = torch.cat([emb_lon, emb_lat], dim=-1)
        if emb.shape[-1] < d:
            pad = torch.zeros(emb.shape[0], d - emb.shape[-1], device=emb.device)
            emb = torch.cat([emb, pad], dim=-1)
        return emb.unsqueeze(1)

    def _area_pos_embed(self, areas, d):
        return self.scale_expansion(areas, d).unsqueeze(1)

    def _time_pos_embed(self, times, d):
        return self.time_expansion(times, d).unsqueeze(1)

    def forward_features(self, x, meta_info, wv_list, bw_list):
        """
        Args:
            x: (B, C, H, W) image tensor.
            meta_info: (B, 4) tensor [lon, lat, time_delta, area]. Use nan to disable.
            wv_list: list of central wavelengths in nm, one per channel.
            bw_list: list of bandwidths in nm, one per channel.
        Returns:
            list of (B, embed_dim, H', W') intermediate feature maps.
        """
        wvs = torch.tensor(wv_list, device=x.device, dtype=torch.float32)
        bws = torch.tensor(bw_list, device=x.device, dtype=torch.float32)

        x, _ = self.patch_embed(x, wvs, bws)  # (B, num_patches, embed_dim)

        # Resize positional embedding if input spatial size differs from training size
        gh = gw = int(math.sqrt(x.size(1)))
        base_g = int(math.sqrt(self.num_patches))
        pos_embed = _resize_abs_pos_embed(self.pos_embed, (gh, gw), (base_g, base_g), num_prefix_tokens=1)

        d = pos_embed.shape[-1]
        lons, lats, times, areas = (meta_info[:, i] for i in range(4))

        # Metadata embeddings — fall back to learned tokens when nan
        if torch.isnan(lons).any() or torch.isnan(lats).any():
            coord_embed = self.coord_token
        else:
            coord_embed = self._coord_pos_embed(lons, lats, d)
        coord_embed = self.coord_fc(coord_embed)

        if torch.isnan(areas).any():
            area_embed = self.scale_token
        else:
            area_embed = self._area_pos_embed(areas, d)
        area_embed = self.scale_fc(area_embed)

        if torch.isnan(times).any():
            time_embed = self.time_token
        else:
            time_embed = self._time_pos_embed(times, d)
        time_embed = self.time_fc(time_embed)

        pos_embed = pos_embed + coord_embed + area_embed + time_embed

        x = x + pos_embed[:, 1:, :]
        cls_tokens = (self.cls_token + pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        output = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == len(self.blocks) - 1:
                x = self.norm(x)
            if i in self.intermediate_indices:
                feat = x[:, 1:]  # drop cls token
                B, _, C = feat.shape
                feat = feat.reshape(B, gh, gw, C).permute(0, 3, 1, 2).contiguous()
                output.append(feat)

        return output


# ---------------------------------------------------------------------------
# PANGAEA encoder wrapper
# ---------------------------------------------------------------------------

class CopernicusFM_Encoder(Encoder):
    """
    Copernicus-FM encoder for the PANGAEA benchmark.

    Paper: https://arxiv.org/abs/2503.11849
    Weights: https://huggingface.co/wangyi111/Copernicus-FM

    Metadata (lon, lat, time_delta, patch_area) is set to nan by default, which
    activates the model's learned fallback tokens. Per-dataset metadata can be
    wired in later via the 'metadata' field in the dataset's __getitem__ output.

    Args:
        encoder_weights: Path to the .pth checkpoint.
        input_bands: Dict mapping modality → list of band names.
        input_size: Spatial size of input patches (e.g. 224).
        embed_dim: Transformer embedding dimension.
        output_layers: Block indices from which to extract feature maps.
        output_dim: Output channel dim (equals embed_dim for all layers).
        wave_list: Dict {modality: {band_name: wavelength_nm}} for central wavelengths.
        bandwidth_list: Dict {modality: {band_name: bandwidth_nm}} for bandwidths.
        download_url: URL to auto-download weights if missing.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        patch_size: Patch size in pixels.
        wv_planes: Dimension of the wavelength embedding.
        mlp_ratio: MLP expansion ratio in transformer blocks.
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
        bandwidth_list: dict[str, dict[str, float]],
        download_url: str,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 16,
        wv_planes: int = 128,
        mlp_ratio: float = 4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__(
            model_name="copernicus_fm",
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

        self.wave_list = wave_list
        self.bandwidth_list = bandwidth_list
        self.output_layers = output_layers

        # Flatten wave/bandwidth values in the same order as input_bands
        self.wv_list = [
            self.wave_list[m][b]
            for m, bands in self.input_bands.items()
            for b in bands
        ]
        self.bw_list = [
            self.bandwidth_list[m][b]
            for m, bands in self.input_bands.items()
            for b in bands
        ]

        self.model = _CopernicusFMViT(
            img_size=input_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            wv_planes=wv_planes,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            intermediate_indices=output_layers,
        )

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            image: Dict with modality keys. Each value has shape (B, C, T, H, W).
                   T is squeezed (single-temporal only).
        Returns:
            List of feature maps, one per output_layer, each (B, embed_dim, H', W').
        """
        # Concatenate all modality channels, squeeze the time dimension
        x = torch.cat(
            [image[m].squeeze(2) for m in self.input_bands.keys()], dim=1
        )  # (B, total_C, H, W)

        # All metadata nan — activates learned fallback tokens in the ViT
        meta_info = torch.full((x.shape[0], 4), float("nan"), device=x.device)

        return self.model.forward_features(x, meta_info, self.wv_list, self.bw_list)

    def load_encoder_weights(self, logger: Logger) -> None:
        if self.encoder_weights is None:
            return

        pretrained = torch.load(self.encoder_weights, map_location="cpu", weights_only=False)

        # The checkpoint may wrap weights under a top-level key
        if isinstance(pretrained, dict) and "model" in pretrained:
            pretrained = pretrained["model"]

        # The official checkpoint names the spectral patch embedding "patch_embed_spectral"
        # and also contains a "patch_embed_variable" we don't use. Remap so keys align.
        remapped = {}
        for ck, cv in pretrained.items():
            if ck.startswith("patch_embed_spectral."):
                remapped[ck.replace("patch_embed_spectral.", "patch_embed.", 1)] = cv
            elif ck.startswith("patch_embed_variable."):
                pass  # variable mode not used — skip
            else:
                remapped[ck] = cv
        pretrained = remapped

        k = pretrained.keys()
        pretrained_encoder = {}
        incompatible_shape = {}
        missing = {}

        for name, param in self.model.named_parameters():
            if name not in k:
                missing[name] = param.shape
            elif pretrained[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, pretrained[name].shape)
            else:
                pretrained_encoder[name] = pretrained[name]

        self.model.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)
