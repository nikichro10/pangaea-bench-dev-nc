# SMARTIES encoder for the PANGAEA benchmark.
# Paper: "SMARTIES: Foundation Model for Multi-Sensor Remote Sensing Imagery" (ICCV 2025)
# https://github.com/gsumbul/SMARTIES
#
# Dependencies inlined to avoid requiring the original repo (MIT license):
#   - SpectrumRangeProjection    (utils/model_utils.py)
#   - SpectrumRangeProjectionAvg (utils/model_utils.py)
#   - SpectrumAwareProjection    (utils/model_utils.py)
#   - positional embedding utils (utils/model_utils.py)
#   - _tensor_patchify           (utils/model_utils.py)

from functools import partial
from logging import Logger
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from pangaea.encoders.base import Encoder


# ---------------------------------------------------------------------------
# Spectrum-aware patch projection — inlined from SMARTIES/utils/model_utils.py
# ---------------------------------------------------------------------------

class SpectrumRangeProjection(nn.Module):
    """Linear projector for a single spectral range."""

    def __init__(self, spectral_range, spectrum_spec, patch_size, embed_dim, bias=True):
        super().__init__()
        self.spectral_range = spectral_range
        self.name = spectrum_spec["name"]
        self.min_wavelength = spectrum_spec["min_wavelength"]
        self.max_wavelength = spectrum_spec["max_wavelength"]
        self.sensors = spectrum_spec["sensors"]
        self.nb_pixels = patch_size ** 2
        self.proj = nn.Linear(self.nb_pixels, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.view(-1, self.nb_pixels))


class SpectrumRangeProjectionAvg(nn.Module):
    """Aggregated projector: wavelength-weighted average of existing projectors."""

    def __init__(self, spectrum_projections, spectrum_spec, embed_dim):
        super().__init__()
        self.min_wavelength = spectrum_spec["min_wavelength"]
        self.max_wavelength = spectrum_spec["max_wavelength"]
        central = 0.5 * (float(self.min_wavelength) + float(self.max_wavelength))
        self.spectrum_projections = spectrum_projections  # plain list — weights already in parent
        raw_weights = [abs(central - 0.5 * (float(sp.min_wavelength) + float(sp.max_wavelength)))
                       for sp in spectrum_projections]
        total = sum(raw_weights) or 1.0
        self.weights = [w / total for w in raw_weights]
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0.0
        for sp, w in zip(self.spectrum_projections, self.weights):
            out = out + sp(x) * w
        return out


class SpectrumAwareProjection(nn.Module):
    """One linear projector per spectral range, indexed by projection_idx."""

    def __init__(self, spectrum_specs: dict, patch_size: int, embed_dim: int, bias: bool = True):
        super().__init__()
        self.spectrum_embeds = nn.ModuleList()

        # First pass: direct (non-aggregate) projectors, ordered by projection_idx
        for key in sorted(spectrum_specs, key=lambda k: spectrum_specs[k]["projection_idx"]):
            spec = spectrum_specs[key]
            if spec["projection_idx"] != -1 and len(spec["agg_projections"]) == 0:
                self.spectrum_embeds.append(
                    SpectrumRangeProjection(key, spec, patch_size, embed_dim, bias)
                )

        # Second pass: aggregate projectors (reference already-built direct projectors)
        for key in sorted(spectrum_specs, key=lambda k: spectrum_specs[k]["projection_idx"]):
            spec = spectrum_specs[key]
            if spec["projection_idx"] != -1 and len(spec["agg_projections"]) > 0:
                refs = [self.spectrum_embeds[i] for i in spec["agg_projections"]]
                self.spectrum_embeds.append(
                    SpectrumRangeProjectionAvg(refs, spec, embed_dim)
                )

    def forward(self, x: torch.Tensor, projection_idx: int) -> torch.Tensor:
        return self.spectrum_embeds[projection_idx](x)


# ---------------------------------------------------------------------------
# Positional embedding helpers — inlined from SMARTIES/utils/model_utils.py
# ---------------------------------------------------------------------------

def _get_1d_sincos_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=float) / (embed_dim / 2.0)
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=float)
    grid_w = np.arange(grid_size, dtype=float)
    grid = np.stack(np.meshgrid(grid_w, grid_h), axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _get_1d_sincos_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed(embed_dim // 2, grid[1])
    pos_embed = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _tensor_patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    imgs: (B, C, H, W)
    returns: (B, nb_patch_h, nb_patch_w, C, patch_size, patch_size)
    """
    p = patch_size
    B, C, H, W = imgs.shape
    h, w = H // p, W // p
    x = imgs.reshape(B, C, h, p, w, p)
    x = torch.einsum("nchpwq->nhwpqc", x)
    return x.reshape(B, h, w, p, p, C).permute(0, 1, 2, 5, 3, 4)


# ---------------------------------------------------------------------------
# Electromagnetic spectrum specs — verbatim from spectrum_specs.yaml (HF repo)
# ---------------------------------------------------------------------------

_SPECTRUM_SPECS: dict = {
    "aerosol":               {"min_wavelength": 422,    "max_wavelength": 463,    "sensors": ["SENTINEL2-L1C", "SENTINEL2-L2A"], "name": "B01 (aerosol)",            "projection_idx": 0,  "agg_projections": []},
    "blue_1":                {"min_wavelength": 427,    "max_wavelength": 558,    "sensors": ["SENTINEL2-L1C", "SENTINEL2-L2A"], "name": "B02 (blue)",               "projection_idx": 1,  "agg_projections": []},
    "blue_2":                {"min_wavelength": 452,    "max_wavelength": 512,    "sensors": ["Landsat8-L2"],                    "name": "B2 (blue)",                "projection_idx": 18, "agg_projections": [0, 1]},
    "blue_3":                {"min_wavelength": 430,    "max_wavelength": 545,    "sensors": ["RGB"],                            "name": "blue",                     "projection_idx": 2,  "agg_projections": []},
    "green_1":               {"min_wavelength": 466,    "max_wavelength": 620,    "sensors": ["RGB"],                            "name": "green",                    "projection_idx": 3,  "agg_projections": []},
    "green_2":               {"min_wavelength": 524,    "max_wavelength": 595,    "sensors": ["SENTINEL2-L1C", "SENTINEL2-L2A"], "name": "B03 (green)",              "projection_idx": 4,  "agg_projections": []},
    "red_1":                 {"min_wavelength": 590,    "max_wavelength": 710,    "sensors": ["RGB"],                            "name": "red",                      "projection_idx": 5,  "agg_projections": []},
    "red_2":                 {"min_wavelength": 634,    "max_wavelength": 696,    "sensors": ["SENTINEL2"],                      "name": "B04 (red)",                "projection_idx": 6,  "agg_projections": []},
    "red_edge_1":            {"min_wavelength": 689,    "max_wavelength": 719,    "sensors": ["SENTINEL2-L1C", "SENTINEL2-L2A"], "name": "B05 (red edge 1)",         "projection_idx": 7,  "agg_projections": []},
    "red_edge_2":            {"min_wavelength": 726,    "max_wavelength": 755,    "sensors": ["SENTINEL2-L1C", "SENTINEL2-L2A"], "name": "B06 (red edge 2)",         "projection_idx": 8,  "agg_projections": []},
    "near_infrared_1":       {"min_wavelength": 728,    "max_wavelength": 938,    "sensors": ["SENTINEL2-L1C", "SENTINEL2-L2A"], "name": "B08 (NIR 1)",              "projection_idx": 9,  "agg_projections": []},
    "near_infrared_2":       {"min_wavelength": 761,    "max_wavelength": 802,    "sensors": ["SENTINEL2-L1C", "SENTINEL2-L2A"], "name": "B07 (NIR 2)",              "projection_idx": 10, "agg_projections": []},
    "near_infrared_3":       {"min_wavelength": 843,    "max_wavelength": 886,    "sensors": ["SENTINEL2-L1C", "SENTINEL2-L2A"], "name": "B8A (NIR 3)",              "projection_idx": 11, "agg_projections": []},
    "short_wave_infrared_1": {"min_wavelength": 923,    "max_wavelength": 964,    "sensors": ["SENTINEL2-L1C", "SENTINEL2-L2A"], "name": "B09 (SWIR water vapour)",  "projection_idx": 12, "agg_projections": []},
    "short_wave_infrared_2": {"min_wavelength": 1345,   "max_wavelength": 1406,   "sensors": ["SENTINEL2-L1C"],                  "name": "B10 (SWIR cirrus)",        "projection_idx": -1, "agg_projections": []},
    "short_wave_infrared_3": {"min_wavelength": 1516,   "max_wavelength": 1704,   "sensors": ["SENTINEL2-L1C", "SENTINEL2-L2A"], "name": "B11 (SWIR 1)",             "projection_idx": 13, "agg_projections": []},
    "short_wave_infrared_4": {"min_wavelength": 2002,   "max_wavelength": 2376,   "sensors": ["SENTINEL2-L1C", "SENTINEL2-L2A"], "name": "B12 (SWIR 2)",             "projection_idx": 14, "agg_projections": []},
    "thermal_infrared_1":    {"min_wavelength": 10600,  "max_wavelength": 11190,  "sensors": ["Landsat8-L2"],                    "name": "B10 (surface temperature)","projection_idx": 17, "agg_projections": [14, 15]},
    "microwave_1":           {"min_wavelength": 5.5e7,  "max_wavelength": 5.6e7,  "sensors": ["SENTINEL1-GRD"],                  "name": "VV",                       "projection_idx": 15, "agg_projections": []},
    "microwave_2":           {"min_wavelength": 5.5e7,  "max_wavelength": 5.6e7,  "sensors": ["SENTINEL1-GRD"],                  "name": "VH",                       "projection_idx": 16, "agg_projections": []},
}


# ---------------------------------------------------------------------------
# Band name → projection_idx for Pangaea's naming convention
# projection_idx = -1 means the band has no SMARTIES projector (excluded).
# ---------------------------------------------------------------------------

_BAND_TO_PROJ_IDX: dict[str, dict[str, int]] = {
    "optical": {
        # Sentinel-2
        "B1":   0,    # aerosol
        "B2":   1,    # blue_1
        "B3":   4,    # green_2
        "B4":   6,    # red_2
        "B5":   7,    # red_edge_1
        "B6":   8,    # red_edge_2
        "B7":   10,   # near_infrared_2
        "B8":   9,    # near_infrared_1
        "B8A":  11,   # near_infrared_3
        "B9":   12,   # short_wave_infrared_1
        "B10":  -1,   # SWIR cirrus — not supported by SMARTIES
        "B11":  13,   # short_wave_infrared_3
        "B12":  14,   # short_wave_infrared_4
        # RGB
        "red":   5,   # red_1
        "green": 3,   # green_1
        "blue":  2,   # blue_3
        # Landsat-8 (closest SMARTIES projectors)
        "L8_B2": 18,  # blue_2 (Landsat-specific aggregate of aerosol + blue_1)
        "L8_B3": 4,   # green_2  (524–595 nm; L8 green ~562 nm)
        "L8_B4": 6,   # red_2    (634–696 nm; L8 red ~655 nm)
        "L8_B5": 11,  # near_infrared_3 (843–886 nm; L8 NIR ~865 nm)
        "L8_B6": 13,  # short_wave_infrared_3 (1516–1704 nm; L8 SWIR1 ~1609 nm)
        "L8_B7": 14,  # short_wave_infrared_4 (2002–2376 nm; L8 SWIR2 ~2201 nm)
    },
    "sar": {
        "VV":     15,
        "VH":     16,
        "ASC_VV": 15,
        "ASC_VH": 16,
        "DSC_VV": 15,
        "DSC_VH": 16,
        "VV-VH":  -1,  # derived band — no SMARTIES projector
    },
}


# ---------------------------------------------------------------------------
# Pangaea encoder wrapper
# ---------------------------------------------------------------------------

class SMARTIES_Encoder(Encoder):
    """
    SMARTIES ViT-B encoder for the PANGAEA benchmark.

    Paper: https://arxiv.org/pdf/2501.01890 (ICCV 2025)
    GitHub: https://github.com/gsumbul/SMARTIES
    Weights: https://huggingface.co/gsumbul/SMARTIES-v1-ViT-B

    Each spectral band is embedded by the linear projector trained for its
    wavelength range (spectrum-aware projection). The band embeddings are
    averaged before being fed into a standard ViT encoder.

    Note: S2 B10 (SWIR cirrus, projection_idx=-1) has no SMARTIES projector
    and is silently skipped. The derived SAR band VV-VH is also excluded.
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
        patch_size: int = 16,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        proj_idx_override: dict[str, dict[str, int]] | None = None,
    ):
        super().__init__(
            model_name="smarties_vitb",
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
        nb_patch = input_size // patch_size

        # Determine which input channels are valid and their projection indices.
        # Channel ordering follows the concatenation order of input_bands dict.
        valid_channels: list[int] = []
        proj_idx_per_band: list[int] = []
        excluded: list[str] = []

        # Merge optional per-config overrides on top of the global defaults.
        # proj_idx_override = {"optical": {"B4": 5, "B3": 3, "B2": 2}} etc.
        effective_map: dict[str, dict[str, int]] = {
            mod: dict(mapping) for mod, mapping in _BAND_TO_PROJ_IDX.items()
        }
        if proj_idx_override:
            for mod, overrides in proj_idx_override.items():
                effective_map.setdefault(mod, {}).update(overrides)

        channel_idx = 0
        for modality, bands in input_bands.items():
            modality_map = effective_map.get(modality, {})
            for band in bands:
                proj_idx = modality_map.get(band, -1)
                if proj_idx == -1:
                    excluded.append(f"{modality}/{band}")
                else:
                    valid_channels.append(channel_idx)
                    proj_idx_per_band.append(proj_idx)
                channel_idx += 1

        if not valid_channels:
            raise ValueError(
                "No valid SMARTIES band mappings found for the provided input_bands. "
                f"Excluded bands: {excluded}"
            )
        if excluded:
            import warnings
            warnings.warn(
                f"SMARTIES_Encoder: the following bands have no projector and will be "
                f"excluded from the forward pass: {excluded}",
                stacklevel=2,
            )

        self._valid_channels: list[int] = valid_channels
        # Register as buffer so it moves to the right device automatically
        self.register_buffer(
            "_proj_idx_buf",
            torch.tensor(proj_idx_per_band, dtype=torch.int32),
        )

        # ViT components (mirror of SMARTIESHF in the HuggingFace repo)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, nb_patch * nb_patch + 1, embed_dim), requires_grad=False
        )
        pos_np = _get_2d_sincos_pos_embed(embed_dim, nb_patch, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_np).float().unsqueeze(0))

        self.pos_drop = nn.Dropout(p=0.0)
        self.projection_scaler = 12

        self.spectrum_projection = SpectrumAwareProjection(
            spectrum_specs=_SPECTRUM_SPECS,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        # Concatenate modalities: (B, total_C, H, W)
        imgs = torch.cat([x[m] for m in self.input_bands], dim=1)
        B, _, H, W = imgs.shape
        nb_patch_h = H // self.patch_size
        nb_patch_w = W // self.patch_size

        # Patchify all channels: (B, nb_patch_h, nb_patch_w, total_C, p, p)
        all_patches = _tensor_patchify(imgs, self.patch_size)

        # Select valid bands: (B, nb_patch_h, nb_patch_w, nb_valid, p, p)
        img_patches = all_patches[:, :, :, self._valid_channels, :, :]
        nb_valid = img_patches.shape[3]

        # proj_indices: (B, nb_patch_h, nb_patch_w, nb_valid)
        proj_indices = self._proj_idx_buf.view(1, 1, 1, nb_valid).expand(
            B, nb_patch_h, nb_patch_w, nb_valid
        )

        # Spectrum-aware embedding per band
        img_spectrum_embeds = torch.zeros(
            B, nb_patch_h, nb_patch_w, nb_valid, self.embed_dim,
            device=imgs.device, dtype=imgs.dtype,
        )
        for proj_idx in torch.unique(proj_indices):
            mask = proj_indices == proj_idx
            # img_patches[mask]: (N_true, p, p)  →  spectrum_projection: (N_true, embed_dim)
            img_spectrum_embeds[mask] = self.spectrum_projection(
                img_patches[mask], int(proj_idx.item())
            )

        # Average over bands, scale, flatten patches: (B, nb_patches, embed_dim)
        x_seq = self.projection_scaler * img_spectrum_embeds.mean(dim=3)
        x_seq = x_seq.reshape(B, nb_patch_h * nb_patch_w, self.embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, x_seq), dim=1)
        x_seq = x_seq + self.pos_embed
        x_seq = self.pos_drop(x_seq)

        output: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            x_seq = blk(x_seq)
            if i == len(self.blocks) - 1:
                x_seq = self.norm(x_seq)
            if i in self.output_layers:
                feat = (
                    x_seq[:, 1:]
                    .permute(0, 2, 1)
                    .reshape(B, self.embed_dim, nb_patch_h, nb_patch_w)
                    .contiguous()
                )
                output.append(feat)

        return output

    def load_encoder_weights(self, logger: Logger) -> None:
        from safetensors.torch import load_file

        pretrained = load_file(self.encoder_weights)

        # Skip decoder-side keys (MAE decoder not needed for inference)
        encoder_keys = {
            k: v for k, v in pretrained.items()
            if not k.startswith(("decoder_", "mask_token"))
        }

        missing: dict[str, torch.Size] = {}
        incompatible_shape: dict[str, tuple[torch.Size, torch.Size]] = {}
        pretrained_encoder: dict[str, torch.Tensor] = {}

        for name, param in self.named_parameters():
            if name not in encoder_keys:
                missing[name] = param.shape
            elif encoder_keys[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, encoder_keys[name].shape)
            else:
                pretrained_encoder[name] = encoder_keys[name]

        self.load_state_dict(pretrained_encoder, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)
