import math
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .hiera import (
    HieraBlock,
    Unroll,
    Reroll,
    do_masked_conv,
    get_2d_sincos_pos_embed,
)

from .two_stages_hiera import _SpectralStageEncoder
from .base import Encoder


def _normalize_sensor_name(name: str) -> str:
    return str(name).upper()


def _build_channel_groups(in_chans: int, num_groups: int) -> List[List[int]]:
    if in_chans <= 0:
        raise ValueError(f"in_chans must be > 0, got {in_chans}")
    if num_groups <= 0:
        raise ValueError(f"num_groups must be > 0, got {num_groups}")

    # Avoid empty groups on low-channel sensors.
    groups = min(in_chans, num_groups)
    base = in_chans // groups
    rem = in_chans % groups

    out: List[List[int]] = []
    start = 0
    for i in range(groups):
        size = base + (1 if i < rem else 0)
        out.append(list(range(start, start + size)))
        start += size
    return out


def _coerce_channel_groups(
    channel_groups: Sequence[Sequence[int]],
    *,
    context: str,
) -> List[List[int]]:
    if not isinstance(channel_groups, (list, tuple)):
        raise ValueError(f"{context}: groups must be a list of lists.")

    out: List[List[int]] = []
    for i, group in enumerate(channel_groups):
        if not isinstance(group, (list, tuple)):
            raise ValueError(
                f"{context}: group index {i} must be a list of band indices."
            )
        group_indices = [int(v) for v in group]
        if not group_indices:
            raise ValueError(f"{context}: group index {i} is empty.")
        out.append(group_indices)
    return out


def _load_channel_groups_from_yaml(path: str, sensor: str) -> List[List[int]]:
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load spectral groups file for sensor '{sensor}' at '{path}': {e}"
        ) from e

    if isinstance(data, list):
        groups = data
    elif isinstance(data, dict) and "groups" in data:
        groups = data["groups"]
    else:
        raise ValueError(
            f"Spectral groups file for sensor '{sensor}' must contain a list "
            "or a dict with key 'groups'."
        )

    return _coerce_channel_groups(
        groups,
        context=f"sensor '{sensor}' spectral_groups_file",
    )


def _validate_channel_groups(
    channel_groups: List[List[int]],
    *,
    in_chans: int,
    sensor: str,
) -> List[List[int]]:
    if in_chans <= 0:
        raise ValueError(f"Sensor '{sensor}' has invalid in_chans={in_chans}.")

    total_grouped_chans = sum(len(g) for g in channel_groups)
    for i, group in enumerate(channel_groups):
        for band_idx in group:
            if band_idx < 0 or band_idx >= in_chans:
                raise ValueError(
                    f"Sensor '{sensor}' group {i} contains band index {band_idx}, "
                    f"valid range is [0, {in_chans - 1}]."
                )

    if total_grouped_chans > in_chans:
        raise ValueError(
            f"Sensor '{sensor}' groups define {total_grouped_chans} channels, "
            f"but input has only {in_chans}."
        )
    if total_grouped_chans < in_chans:
        print(
            f"Notice: sensor '{sensor}' groups use {total_grouped_chans}/{in_chans} channels. "
            f"{in_chans - total_grouped_chans} bands will be ignored."
        )
    return channel_groups


class LinearSpectralEncoder(nn.Module):
    """
    Lightweight spectral encoder for low-band modalities (e.g. S1, LT).

    This is intentionally simple: a single masked conv patch projection.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        patch_kernel: Tuple[int, int],
        patch_stride: Tuple[int, int],
        patch_padding: Tuple[int, int],
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_padding,
            bias=True,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = do_masked_conv(x, self.proj, mask)
        x = x.flatten(2).transpose(1, 2)
        return x


class SensorTokenFusion(nn.Module):
    """
    Fuse modality-specific token grids into one shared token grid.

    Supported modes:
    - mean: arithmetic average
    - weighted: learnable per-sensor scalar weights
    - attention: learnable fusion query attending over available sensors per token
    - projected_attention: project to fusion dim, attend, and project back
    - projected_weighted: project to fusion dim, apply learnable scalar sensor weights, project back
    - multi_query_concat: token-wise multi-query attention over sensors followed by query concatenation
    """

    def __init__(
        self,
        sensors: Sequence[str],
        embed_dim: int,
        mode: str = "mean",
        attention_heads: int = 4,
        projection_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        mode = str(mode).lower()
        mode_alias = {
            "proj_attn": "projected_attention",
            "projected_attention": "projected_attention",
            "weighted_proj": "projected_weighted",
            "proj_weighted": "projected_weighted",
            "weighted_projection": "projected_weighted",
            "projected_weighted": "projected_weighted",
            "concat_query": "multi_query_concat",
            "mq_concat": "multi_query_concat",
            "multi_query_concat": "multi_query_concat",
        }
        mode = mode_alias.get(mode, mode)
        valid_modes = {
            "mean",
            "weighted",
            "attention",
            "projected_attention",
            "projected_weighted",
            "multi_query_concat",
        }
        if mode not in valid_modes:
            raise ValueError(
                "Unknown fusion mode "
                f"'{mode}'. Use one of: mean, weighted, attention, "
                "projected_attention, projected_weighted, multi_query_concat."
            )
        if mode in {"attention", "multi_query_concat"} and embed_dim % attention_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by attention_heads ({attention_heads})"
            )
        proj_dim = int(projection_dim or embed_dim)
        if mode == "projected_attention" and proj_dim % attention_heads != 0:
            raise ValueError(
                f"projection_dim ({proj_dim}) must be divisible by attention_heads ({attention_heads})"
            )
        if mode == "multi_query_concat" and proj_dim % embed_dim != 0:
            raise ValueError(
                f"projection_dim ({proj_dim}) must be divisible by embed_dim ({embed_dim}) "
                "for multi_query_concat."
            )

        self.mode = mode
        self.sensor_order = [_normalize_sensor_name(s) for s in sensors]
        self.sensor_to_index = {s: i for i, s in enumerate(self.sensor_order)}
        self.embed_dim = int(embed_dim)
        self.projection_dim = proj_dim

        if self.mode in {"weighted", "projected_weighted"}:
            self.sensor_logits = nn.Parameter(torch.zeros(len(self.sensor_order)))
        else:
            self.register_parameter("sensor_logits", None)

        if self.mode in {"attention", "projected_attention"}:
            attn_dim = self.embed_dim if self.mode == "attention" else self.projection_dim
            self.query_token = nn.Parameter(torch.zeros(1, 1, attn_dim))
            self.attn = nn.MultiheadAttention(
                embed_dim=attn_dim,
                num_heads=attention_heads,
                batch_first=True,
            )
            nn.init.trunc_normal_(self.query_token, std=0.02)
        else:
            self.register_parameter("query_token", None)
            self.attn = None

        if self.mode in {"projected_attention", "projected_weighted"}:
            self.input_proj = nn.Linear(self.embed_dim, self.projection_dim)
            self.output_proj = nn.Linear(self.projection_dim, self.embed_dim)
            self.fusion_norm = nn.LayerNorm(self.projection_dim)
            self.sensor_embed = nn.Parameter(
                torch.zeros(len(self.sensor_order), self.projection_dim)
            )
            nn.init.trunc_normal_(self.sensor_embed, std=0.02)
        else:
            self.input_proj = None
            self.output_proj = None
            self.fusion_norm = None
            self.sensor_embed = None

        if self.mode == "multi_query_concat":
            self.concat_query_count = self.projection_dim // self.embed_dim
            self.concat_query_token = nn.Parameter(
                torch.zeros(1, self.concat_query_count, self.embed_dim)
            )
            self.concat_attn = nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=attention_heads,
                batch_first=True,
            )
            self.concat_norm = nn.LayerNorm(self.embed_dim)
            self.concat_output_proj = (
                nn.Identity()
                if self.projection_dim == self.embed_dim
                else nn.Linear(self.projection_dim, self.embed_dim)
            )
            nn.init.trunc_normal_(self.concat_query_token, std=0.02)
        else:
            self.concat_query_count = 0
            self.register_parameter("concat_query_token", None)
            self.concat_attn = None
            self.concat_norm = None
            self.concat_output_proj = None

        self._attn_chunk_size = 8192
        self.last_modality_contributions: Dict[str, float] = {
            s: 0.0 for s in self.sensor_order
        }

    def _cache_last_modality_contributions(
        self,
        sensor_names: Sequence[str],
        weights: torch.Tensor,
    ) -> None:
        contrib = {s: 0.0 for s in self.sensor_order}
        weights = weights.detach().float().cpu()
        for idx, sensor in enumerate(sensor_names):
            s_norm = _normalize_sensor_name(sensor)
            if idx < int(weights.shape[0]):
                contrib[s_norm] = float(weights[idx].item())
        self.last_modality_contributions = contrib

    def _run_attention(
        self,
        attn_module: nn.MultiheadAttention,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_avg_src_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n = q.shape[0]
        if n <= self._attn_chunk_size:
            if q.device.type == "cuda":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                ):
                    out, attn_w = attn_module(
                        q,
                        k,
                        v,
                        need_weights=return_avg_src_weights,
                        average_attn_weights=True,
                    )
                    if return_avg_src_weights and attn_w is not None:
                        return out, attn_w.mean(dim=(0, 1))
                    return out, None
            out, attn_w = attn_module(
                q,
                k,
                v,
                need_weights=return_avg_src_weights,
                average_attn_weights=True,
            )
            if return_avg_src_weights and attn_w is not None:
                return out, attn_w.mean(dim=(0, 1))
            return out, None

        out_chunks: List[torch.Tensor] = []
        weight_sum: Optional[torch.Tensor] = None
        weight_count = 0
        for start in range(0, n, self._attn_chunk_size):
            end = min(start + self._attn_chunk_size, n)
            q_i = q[start:end]
            k_i = k[start:end]
            v_i = v[start:end]
            if q.device.type == "cuda":
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False,
                    enable_math=True,
                    enable_mem_efficient=False,
                ):
                    out_i, attn_w_i = attn_module(
                        q_i,
                        k_i,
                        v_i,
                        need_weights=return_avg_src_weights,
                        average_attn_weights=True,
                    )
            else:
                out_i, attn_w_i = attn_module(
                    q_i,
                    k_i,
                    v_i,
                    need_weights=return_avg_src_weights,
                    average_attn_weights=True,
                )
            out_chunks.append(out_i)
            if return_avg_src_weights and attn_w_i is not None:
                src_sum = attn_w_i.sum(dim=(0, 1))
                weight_sum = src_sum if weight_sum is None else (weight_sum + src_sum)
                weight_count += int(attn_w_i.shape[0] * attn_w_i.shape[1])
        out = torch.cat(out_chunks, dim=0)
        if return_avg_src_weights and weight_sum is not None and weight_count > 0:
            return out, (weight_sum / float(weight_count))
        return out, None

    def forward(
        self,
        tokens_per_sensor: Dict[str, torch.Tensor],
        available_sensors: Sequence[str],
    ) -> torch.Tensor:
        if not available_sensors:
            raise ValueError("No sensors available for fusion.")

        sensor_names = [_normalize_sensor_name(s) for s in available_sensors]
        self.last_modality_contributions = {s: 0.0 for s in self.sensor_order}
        stacked = torch.stack([tokens_per_sensor[s] for s in sensor_names], dim=1)
        # stacked: [B, M, N, D]

        if self.mode == "mean":
            modality_weights = torch.full(
                (stacked.shape[1],),
                1.0 / max(int(stacked.shape[1]), 1),
                device=stacked.device,
                dtype=stacked.dtype,
            )
            self._cache_last_modality_contributions(sensor_names, modality_weights)
            if stacked.shape[1] == 1:
                return stacked[:, 0]
            return stacked.mean(dim=1)

        if self.mode == "weighted":
            if stacked.shape[1] == 1:
                self._cache_last_modality_contributions(
                    sensor_names,
                    torch.ones((1,), device=stacked.device, dtype=stacked.dtype),
                )
                return stacked[:, 0]
            idx = torch.tensor(
                [self.sensor_to_index[s] for s in sensor_names],
                device=stacked.device,
                dtype=torch.long,
            )
            weights = torch.softmax(self.sensor_logits[idx], dim=0)
            self._cache_last_modality_contributions(sensor_names, weights)
            weights = weights.view(1, -1, 1, 1)
            return (stacked * weights).sum(dim=1)

        bsz, num_modalities, num_tokens, dim = stacked.shape
        token_major = stacked.permute(0, 2, 1, 3).reshape(bsz * num_tokens, num_modalities, dim)

        if self.mode == "attention":
            if num_modalities == 1:
                self._cache_last_modality_contributions(
                    sensor_names,
                    torch.ones((1,), device=stacked.device, dtype=stacked.dtype),
                )
                return stacked[:, 0]
            q = self.query_token.expand(bsz * num_tokens, -1, -1)
            fused, attn_w = self._run_attention(
                self.attn,
                q,
                token_major,
                token_major,
                return_avg_src_weights=True,
            )
            if attn_w is not None:
                self._cache_last_modality_contributions(sensor_names, attn_w)
            return fused.squeeze(1).reshape(bsz, num_tokens, dim)

        if self.mode == "projected_attention":
            token_major = self.input_proj(token_major)
            idx = torch.tensor(
                [self.sensor_to_index[s] for s in sensor_names],
                device=stacked.device,
                dtype=torch.long,
            )
            token_major = token_major + self.sensor_embed[idx].view(1, num_modalities, -1)

            if num_modalities == 1:
                self._cache_last_modality_contributions(
                    sensor_names,
                    torch.ones((1,), device=stacked.device, dtype=stacked.dtype),
                )
                fused_vec = self.fusion_norm(token_major[:, 0, :])
            else:
                q = self.query_token.expand(bsz * num_tokens, -1, -1)
                fused, attn_w = self._run_attention(
                    self.attn,
                    q,
                    token_major,
                    token_major,
                    return_avg_src_weights=True,
                )
                if attn_w is not None:
                    self._cache_last_modality_contributions(sensor_names, attn_w)
                fused_vec = self.fusion_norm(fused.squeeze(1))

            fused_vec = self.output_proj(fused_vec)
            return fused_vec.reshape(bsz, num_tokens, dim)

        if self.mode == "projected_weighted":
            token_major = self.input_proj(token_major)
            idx = torch.tensor(
                [self.sensor_to_index[s] for s in sensor_names],
                device=stacked.device,
                dtype=torch.long,
            )
            token_major = token_major + self.sensor_embed[idx].view(1, num_modalities, -1)
            weights = torch.softmax(self.sensor_logits[idx], dim=0)
            self._cache_last_modality_contributions(sensor_names, weights)
            weights = weights.view(1, -1, 1)
            fused_vec = (token_major * weights).sum(dim=1)
            fused_vec = self.fusion_norm(fused_vec)
            fused_vec = self.output_proj(fused_vec)
            return fused_vec.reshape(bsz, num_tokens, dim)

        if self.mode == "multi_query_concat":
            if num_modalities == 1:
                self._cache_last_modality_contributions(
                    sensor_names,
                    torch.ones((1,), device=stacked.device, dtype=stacked.dtype),
                )
                fused_multi = token_major.expand(-1, self.concat_query_count, -1)
                fused_multi = self.concat_norm(fused_multi)
            else:
                q = self.concat_query_token.expand(bsz * num_tokens, -1, -1)
                fused_multi, attn_w = self._run_attention(
                    self.concat_attn,
                    q,
                    token_major,
                    token_major,
                    return_avg_src_weights=True,
                )
                if attn_w is not None:
                    self._cache_last_modality_contributions(sensor_names, attn_w)
                fused_multi = self.concat_norm(fused_multi)
            fused_concat = fused_multi.reshape(bsz * num_tokens, self.projection_dim)
            fused_out = self.concat_output_proj(fused_concat)
            return fused_out.reshape(bsz, num_tokens, dim)

        raise RuntimeError(f"Unsupported fusion mode '{self.mode}'")


class MultiSensorTwoStagesHiera(nn.Module):
    """
    Multi-sensor two-stages Hiera backbone.

    Pipeline:
    1) Sensor-specific spectral encoder (linear or spectral-transformer)
    2) Sensor-specific local Hiera stages
    3) Cross-sensor token fusion
    4) Shared final Hiera stage (global spatial attention)
    """

    def __init__(
        self,
        sensors: Sequence[str],
        sensor_in_chans: Dict[str, int],
        sensor_input_size: Dict[str, Tuple[int, int]],
        sensor_encoder_types: Optional[Dict[str, str]] = None,
        embed_dim: int = 192,
        num_heads: int = 3,
        num_classes: int = 0,
        stages: Tuple[int, ...] = (2, 6, 10),
        q_pool: int = 2,
        q_stride: Tuple[int, int] = (2, 2),
        mask_unit_size: Tuple[int, int] = (4, 4),
        mask_unit_attn: Tuple[bool, ...] = (True, True, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, int] = (5, 5),
        patch_stride: Tuple[int, int] = (2, 2),
        patch_padding: Tuple[int, int] = (2, 2),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: Union[str, nn.Module] = "LayerNorm",
        head_dropout: float = 0.0,
        head_init_scale: float = 0.001,
        sep_pos_embed: bool = False,
        patch_size: int = 2,
        pos_embed_type: str = "learnable",
        use_cls_token: bool = False,
        # Spectral-stage defaults (for spectral-transformer sensors)
        spectral_groups: int = 10,
        spec_depth: int = 4,
        spec_num_heads: int = 2,
        spectral_pos_embed_type: str = "learnable",
        spectral_pooling: str = "cls",
        spectral_token_dim: Optional[int] = None,
        spectral_fusion_dim: Optional[int] = None,
        spectral_fusion_heads: Optional[int] = None,
        # Optional per-sensor Hiera overrides.
        per_sensor_patch_kernel: Optional[Dict[str, Tuple[int, int]]] = None,
        per_sensor_patch_stride: Optional[Dict[str, Tuple[int, int]]] = None,
        per_sensor_patch_padding: Optional[Dict[str, Tuple[int, int]]] = None,
        per_sensor_stages: Optional[Dict[str, Tuple[int, ...]]] = None,
        per_sensor_q_pool: Optional[Dict[str, int]] = None,
        per_sensor_q_stride: Optional[Dict[str, Tuple[int, int]]] = None,
        per_sensor_mask_unit_size: Optional[Dict[str, Tuple[int, int]]] = None,
        per_sensor_mask_unit_attn: Optional[Dict[str, Tuple[bool, ...]]] = None,
        per_sensor_num_heads: Optional[Dict[str, int]] = None,
        # Optional per-sensor overrides for spectral-transformer sensors
        per_sensor_spectral_groups: Optional[Dict[str, int]] = None,
        per_sensor_custom_spectral_groups: Optional[Dict[str, List[List[int]]]] = None,
        per_sensor_spectral_groups_file: Optional[Dict[str, str]] = None,
        per_sensor_spec_depth: Optional[Dict[str, int]] = None,
        per_sensor_spec_num_heads: Optional[Dict[str, int]] = None,
        # Fusion
        fusion_mode: str = "mean",
        fusion_attention_heads: int = 4,
        fusion_projection_dim: Optional[int] = None,
        norm_pix: bool = True,
        # Optional canonical sensor(s) for downstream modules.
        # Accepts a single sensor string or a list of sensors.
        canonical_sensor: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:
        super().__init__()

        if sep_pos_embed:
            raise ValueError("sep_pos_embed is not supported in MultiSensorTwoStagesHiera.")
        if num_classes != 0:
            raise ValueError("This backbone is encoder-only. Set num_classes=0.")
        if head_dropout != 0.0 or head_init_scale != 0.001:
            # Keep args for config compatibility, but this backbone has no classifier head.
            pass
        if len(stages) < 2:
            raise ValueError("stages must have at least 2 entries (local + shared).")
        if q_pool >= len(stages):
            raise ValueError(f"q_pool must be < len(stages). Got q_pool={q_pool}, stages={stages}")

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        default_patch_kernel = tuple(int(v) for v in patch_kernel)
        default_patch_stride = tuple(int(v) for v in patch_stride)
        default_patch_padding = tuple(int(v) for v in patch_padding)
        default_stages = tuple(int(v) for v in stages)
        default_q_stride = tuple(int(v) for v in q_stride)
        default_mask_unit_size = tuple(int(v) for v in mask_unit_size)
        default_mask_unit_attn = tuple(bool(v) for v in mask_unit_attn)
        if len(default_stages) < 2:
            raise ValueError("stages must have at least 2 entries (local + shared).")
        if q_pool >= len(default_stages):
            raise ValueError(
                f"q_pool must be < len(stages). Got q_pool={q_pool}, stages={default_stages}"
            )

        self.patch_size = int(patch_size)
        self.use_cls_token = bool(use_cls_token)
        # Kept for backward compatibility with callers checking this flag.
        self.resize_inputs_to_model = False
        self.norm_pix = bool(norm_pix)

        self.sensors = [_normalize_sensor_name(s) for s in sensors]
        if not self.sensors:
            raise ValueError("sensors must not be empty.")

        sensor_in_chans_norm = {
            _normalize_sensor_name(k): int(v) for k, v in sensor_in_chans.items()
        }
        sensor_input_size_norm = {
            _normalize_sensor_name(k): tuple(int(x) for x in v)
            for k, v in sensor_input_size.items()
        }
        self.sensor_in_chans = sensor_in_chans_norm
        self.sensor_input_size = sensor_input_size_norm

        for s in self.sensors:
            if s not in self.sensor_in_chans:
                raise ValueError(f"Missing sensor_in_chans entry for sensor '{s}'.")
            if s not in self.sensor_input_size:
                raise ValueError(f"Missing sensor_input_size entry for sensor '{s}'.")
            if len(self.sensor_input_size[s]) != 2:
                raise ValueError(
                    f"sensor_input_size for sensor '{s}' must have exactly two entries (H, W)."
                )

        # Optionally expose single-sensor-style attributes expected by some downstream
        # modules (e.g. semantic segmentors eager-initializing decoders).
        # We only do this when canonical_sensor is explicitly provided so that
        # pretraining configs remain unchanged and downstream can opt-in per use case.
        self.canonical_sensor: Optional[str] = None
        self.canonical_sensors: List[str] = []
        if canonical_sensor is not None:
            if isinstance(canonical_sensor, str):
                canonical_values = [canonical_sensor]
            elif isinstance(canonical_sensor, Sequence):
                canonical_values = [str(v) for v in canonical_sensor]
            else:
                raise ValueError(
                    "canonical_sensor must be a sensor string or a sequence of sensor names."
                )

            if not canonical_values:
                raise ValueError("canonical_sensor sequence must not be empty.")

            seen: set[str] = set()
            canonical_norm: List[str] = []
            for raw_sensor in canonical_values:
                cs = _normalize_sensor_name(raw_sensor)
                if cs not in self.sensors:
                    raise ValueError(
                        f"canonical_sensor entry '{raw_sensor}' is not one of configured sensors {self.sensors}."
                    )
                if cs not in seen:
                    canonical_norm.append(cs)
                    seen.add(cs)

            self.canonical_sensors = canonical_norm
            # Backward-compatible single canonical sensor alias.
            self.canonical_sensor = canonical_norm[0]
            self.in_chans = int(self.sensor_in_chans[self.canonical_sensor])
            self.img_size = tuple(int(x) for x in self.sensor_input_size[self.canonical_sensor])

        default_encoder_types = {s: "spectral_transformer" for s in self.sensors}
        for s in self.sensors:
            if s in {"S1", "LT"}:
                default_encoder_types[s] = "linear"

        if sensor_encoder_types is not None:
            for k, v in sensor_encoder_types.items():
                default_encoder_types[_normalize_sensor_name(k)] = str(v).lower()

        for s, t in default_encoder_types.items():
            if t not in {"linear", "spectral_transformer"}:
                raise ValueError(
                    f"Invalid encoder type '{t}' for sensor '{s}'. "
                    "Use 'linear' or 'spectral_transformer'."
                )
        self.sensor_encoder_types = default_encoder_types

        def _norm_tuple_map(
            src: Optional[Dict[str, Any]],
            *,
            cast=int,
        ) -> Dict[str, Tuple[int, int]]:
            out: Dict[str, Tuple[int, int]] = {}
            for k, v in (src or {}).items():
                if v is None:
                    continue
                out[_normalize_sensor_name(k)] = tuple(cast(x) for x in v)  # type: ignore[arg-type]
            return out

        per_sensor_patch_kernel = _norm_tuple_map(per_sensor_patch_kernel)
        per_sensor_patch_stride = _norm_tuple_map(per_sensor_patch_stride)
        per_sensor_patch_padding = _norm_tuple_map(per_sensor_patch_padding)
        per_sensor_q_stride = _norm_tuple_map(per_sensor_q_stride)
        per_sensor_mask_unit_size = _norm_tuple_map(per_sensor_mask_unit_size)

        per_sensor_stages = {
            _normalize_sensor_name(k): tuple(int(x) for x in v)
            for k, v in (per_sensor_stages or {}).items()
            if v is not None
        }
        per_sensor_q_pool = {
            _normalize_sensor_name(k): int(v)
            for k, v in (per_sensor_q_pool or {}).items()
            if v is not None
        }
        per_sensor_mask_unit_attn = {
            _normalize_sensor_name(k): tuple(bool(x) for x in v)
            for k, v in (per_sensor_mask_unit_attn or {}).items()
            if v is not None
        }
        per_sensor_num_heads = {
            _normalize_sensor_name(k): int(v)
            for k, v in (per_sensor_num_heads or {}).items()
            if v is not None
        }

        per_sensor_spectral_groups = {
            _normalize_sensor_name(k): int(v)
            for k, v in (per_sensor_spectral_groups or {}).items()
        }
        per_sensor_custom_spectral_groups = {
            _normalize_sensor_name(k): v
            for k, v in (per_sensor_custom_spectral_groups or {}).items()
            if v is not None
        }
        per_sensor_spectral_groups_file = {
            _normalize_sensor_name(k): str(v)
            for k, v in (per_sensor_spectral_groups_file or {}).items()
            if v is not None
        }
        per_sensor_spec_depth = {
            _normalize_sensor_name(k): int(v)
            for k, v in (per_sensor_spec_depth or {}).items()
        }
        per_sensor_spec_num_heads = {
            _normalize_sensor_name(k): int(v)
            for k, v in (per_sensor_spec_num_heads or {}).items()
        }

        self.sensor_patch_kernels: Dict[str, Tuple[int, int]] = {}
        self.sensor_patch_strides: Dict[str, Tuple[int, int]] = {}
        self.sensor_patch_paddings: Dict[str, Tuple[int, int]] = {}
        self.sensor_stages: Dict[str, Tuple[int, ...]] = {}
        self.sensor_q_pool: Dict[str, int] = {}
        self.sensor_q_stride: Dict[str, Tuple[int, int]] = {}
        self.sensor_mask_unit_sizes: Dict[str, Tuple[int, int]] = {}
        self.sensor_mask_unit_attn: Dict[str, Tuple[bool, ...]] = {}
        self.sensor_num_heads: Dict[str, int] = {}
        self.sensor_tokens_spatial_shape: Dict[str, Tuple[int, int]] = {}
        self.sensor_mask_spatial_shape: Dict[str, Tuple[int, int]] = {}
        self.sensor_mu_sizes: Dict[str, int] = {}
        self.sensor_pred_strides: Dict[str, int] = {}

        for sensor in self.sensors:
            sensor_embed_dim = int(embed_dim)
            if sensor_embed_dim != int(embed_dim):
                raise ValueError(
                    "Per-sensor embed_dim overrides are not supported for MultiSensorTwoStagesHiera."
                )

            sensor_patch_kernel = per_sensor_patch_kernel.get(sensor, default_patch_kernel)
            sensor_patch_stride = per_sensor_patch_stride.get(sensor, default_patch_stride)
            sensor_patch_padding = per_sensor_patch_padding.get(sensor, default_patch_padding)
            sensor_stages = per_sensor_stages.get(sensor, default_stages)
            sensor_q_pool = per_sensor_q_pool.get(sensor, int(q_pool))
            sensor_q_stride = per_sensor_q_stride.get(sensor, default_q_stride)
            sensor_mask_unit_size = per_sensor_mask_unit_size.get(sensor, default_mask_unit_size)
            sensor_mask_unit_attn = per_sensor_mask_unit_attn.get(sensor, default_mask_unit_attn)
            sensor_num_heads = per_sensor_num_heads.get(sensor, int(num_heads))

            if len(sensor_stages) < 2:
                raise ValueError(
                    f"Sensor '{sensor}' must have at least 2 stages (local + shared). "
                    f"Got {sensor_stages}."
                )
            if sensor_q_pool >= len(sensor_stages):
                raise ValueError(
                    f"Sensor '{sensor}' has invalid q_pool={sensor_q_pool} for stages={sensor_stages}."
                )
            if len(sensor_mask_unit_attn) != len(sensor_stages):
                raise ValueError(
                    f"Sensor '{sensor}' mask_unit_attn length ({len(sensor_mask_unit_attn)}) "
                    f"must match stages length ({len(sensor_stages)})."
                )

            input_h, input_w = self.sensor_input_size[sensor]
            if input_h % sensor_patch_stride[0] != 0 or input_w % sensor_patch_stride[1] != 0:
                raise ValueError(
                    f"Sensor '{sensor}' input size {self.sensor_input_size[sensor]} must be divisible "
                    f"by patch_stride {sensor_patch_stride}."
                )

            tokens_h = input_h // sensor_patch_stride[0]
            tokens_w = input_w // sensor_patch_stride[1]
            if tokens_h % sensor_mask_unit_size[0] != 0 or tokens_w % sensor_mask_unit_size[1] != 0:
                raise ValueError(
                    f"Sensor '{sensor}' tokens shape {(tokens_h, tokens_w)} must be divisible by "
                    f"mask_unit_size {sensor_mask_unit_size}."
                )

            mask_h = tokens_h // sensor_mask_unit_size[0]
            mask_w = tokens_w // sensor_mask_unit_size[1]
            pred_stride_h = input_h // mask_h
            pred_stride_w = input_w // mask_w
            if pred_stride_h != pred_stride_w:
                raise ValueError(
                    f"Sensor '{sensor}' has anisotropic pred stride {(pred_stride_h, pred_stride_w)}; "
                    "only isotropic pred strides are currently supported."
                )

            self.sensor_patch_kernels[sensor] = sensor_patch_kernel
            self.sensor_patch_strides[sensor] = sensor_patch_stride
            self.sensor_patch_paddings[sensor] = sensor_patch_padding
            self.sensor_stages[sensor] = sensor_stages
            self.sensor_q_pool[sensor] = sensor_q_pool
            self.sensor_q_stride[sensor] = sensor_q_stride
            self.sensor_mask_unit_sizes[sensor] = sensor_mask_unit_size
            self.sensor_mask_unit_attn[sensor] = sensor_mask_unit_attn
            self.sensor_num_heads[sensor] = sensor_num_heads
            self.sensor_tokens_spatial_shape[sensor] = (tokens_h, tokens_w)
            self.sensor_mask_spatial_shape[sensor] = (mask_h, mask_w)
            self.sensor_mu_sizes[sensor] = int(math.prod(sensor_mask_unit_size))
            self.sensor_pred_strides[sensor] = int(pred_stride_h)

        ref_sensor = self.sensors[0]
        ref_tokens_shape = self.sensor_tokens_spatial_shape[ref_sensor]
        ref_mask_shape = self.sensor_mask_spatial_shape[ref_sensor]
        for sensor in self.sensors[1:]:
            if self.sensor_tokens_spatial_shape[sensor] != ref_tokens_shape:
                raise ValueError(
                    f"All sensors must produce the same token grid before fusion. "
                    f"Sensor '{sensor}' has {self.sensor_tokens_spatial_shape[sensor]}, "
                    f"reference '{ref_sensor}' has {ref_tokens_shape}."
                )
            if self.sensor_mask_spatial_shape[sensor] != ref_mask_shape:
                raise ValueError(
                    f"All sensors must produce the same mask-unit grid. "
                    f"Sensor '{sensor}' has {self.sensor_mask_spatial_shape[sensor]}, "
                    f"reference '{ref_sensor}' has {ref_mask_shape}."
                )

        # Backward-compatible canonical attributes derived from the reference sensor.
        self.input_size = self.sensor_input_size[ref_sensor]
        self.img_size = int(self.input_size[0])
        self.patch_stride = self.sensor_patch_strides[ref_sensor]
        self.q_pool = int(self.sensor_q_pool[ref_sensor])
        self.q_stride = self.sensor_q_stride[ref_sensor]
        self.mask_unit_size = self.sensor_mask_unit_sizes[ref_sensor]
        self.mu_size = int(math.prod(self.mask_unit_size))
        self.stages = self.sensor_stages[ref_sensor]
        self.tokens_spatial_shape = list(ref_tokens_shape)
        self.mask_spatial_shape = list(ref_mask_shape)
        num_tokens = int(math.prod(self.tokens_spatial_shape))

        # Positional embedding at spatial-token resolution.
        if pos_embed_type == "learnable":
            self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed_type == "sincos":
            pos_embed_val = get_2d_sincos_pos_embed(embed_dim, tuple(self.tokens_spatial_shape))
            self.register_buffer("pos_embed", pos_embed_val, persistent=True)
        else:
            raise ValueError(f"Unknown pos_embed_type: '{pos_embed_type}'")

        def _build_block_specs(
            *,
            sensor_stages: Tuple[int, ...],
            sensor_q_pool: int,
            sensor_q_stride: Tuple[int, int],
            sensor_mask_unit_attn: Tuple[bool, ...],
            sensor_num_heads: int,
            sensor_mask_unit_size: Tuple[int, int],
        ) -> Tuple[List[Dict[str, Union[int, float, bool, nn.Module]]], List[int]]:
            depth = int(sum(sensor_stages))
            stage_ends = [sum(sensor_stages[:i]) - 1 for i in range(1, len(sensor_stages) + 1)]
            q_pool_blocks = [x + 1 for x in stage_ends[:sensor_q_pool]]
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            flat_q_stride = int(math.prod(sensor_q_stride))
            cur_dim = int(embed_dim)
            cur_heads = int(sensor_num_heads)
            cur_mu = int(math.prod(sensor_mask_unit_size))
            cur_stage = 0

            block_specs: List[Dict[str, Union[int, float, bool, nn.Module]]] = []
            for i in range(depth):
                dim_out = cur_dim
                use_mask_unit_attn = bool(sensor_mask_unit_attn[cur_stage])
                if i - 1 in stage_ends:
                    dim_out = int(cur_dim * dim_mul)
                    cur_heads = int(cur_heads * head_mul)
                    cur_stage += 1
                    if i in q_pool_blocks:
                        cur_mu //= int(flat_q_stride)
                block_specs.append(
                    {
                        "dim": cur_dim,
                        "dim_out": dim_out,
                        "heads": cur_heads,
                        "mlp_ratio": mlp_ratio,
                        "drop_path": dpr[i],
                        "norm_layer": norm_layer,
                        "q_stride": int(flat_q_stride) if i in q_pool_blocks else 1,
                        "window_size": cur_mu,
                        "use_mask_unit_attn": use_mask_unit_attn,
                    }
                )
                cur_dim = dim_out
            return block_specs, stage_ends

        # Sensor-specific spectral encoders and local branches.
        self.sensor_patch_embeds = nn.ModuleDict()
        self.sensor_channel_groups: Dict[str, List[List[int]]] = {}
        self.sensor_unroll = nn.ModuleDict()
        self.sensor_reroll = nn.ModuleDict()
        self.sensor_local_blocks = nn.ModuleDict()
        self.sensor_local_output_proj = nn.ModuleDict()

        shared_specs_ref: Optional[List[Dict[str, Union[int, float, bool, nn.Module]]]] = None
        stage_ends_ref: Optional[List[int]] = None
        local_depth_ref = 0

        for sensor in self.sensors:
            in_chans = self.sensor_in_chans[sensor]
            enc_type = self.sensor_encoder_types[sensor]
            sensor_patch_kernel = self.sensor_patch_kernels[sensor]
            sensor_patch_stride = self.sensor_patch_strides[sensor]
            sensor_patch_padding = self.sensor_patch_paddings[sensor]

            if enc_type == "linear":
                self.sensor_patch_embeds[sensor] = LinearSpectralEncoder(
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    patch_kernel=sensor_patch_kernel,
                    patch_stride=sensor_patch_stride,
                    patch_padding=sensor_patch_padding,
                )
            else:
                sensor_spec_depth = per_sensor_spec_depth.get(sensor, spec_depth)
                sensor_spec_heads = per_sensor_spec_num_heads.get(sensor, spec_num_heads)

                if sensor in per_sensor_custom_spectral_groups:
                    channel_groups = _coerce_channel_groups(
                        per_sensor_custom_spectral_groups[sensor],
                        context=f"sensor '{sensor}' custom groups",
                    )
                elif sensor in per_sensor_spectral_groups_file:
                    channel_groups = _load_channel_groups_from_yaml(
                        per_sensor_spectral_groups_file[sensor],
                        sensor=sensor,
                    )
                else:
                    sensor_groups = per_sensor_spectral_groups.get(sensor, spectral_groups)
                    channel_groups = _build_channel_groups(in_chans, sensor_groups)

                channel_groups = _validate_channel_groups(
                    channel_groups,
                    in_chans=in_chans,
                    sensor=sensor,
                )
                self.sensor_channel_groups[sensor] = channel_groups

                self.sensor_patch_embeds[sensor] = _SpectralStageEncoder(
                    img_size=self.sensor_input_size[sensor],
                    patch_stride=sensor_patch_stride,
                    patch_kernel=sensor_patch_kernel,
                    patch_padding=sensor_patch_padding,
                    channel_groups=channel_groups,
                    embed_dim=embed_dim,
                    spec_depth=sensor_spec_depth,
                    spec_num_heads=sensor_spec_heads,
                    mlp_ratio=mlp_ratio,
                    spectral_pos_embed_type=spectral_pos_embed_type,
                    pooling=spectral_pooling,
                    spectral_token_dim=spectral_token_dim,
                    spectral_fusion_dim=spectral_fusion_dim,
                    spectral_fusion_heads=spectral_fusion_heads,
                )

            block_specs, stage_ends = _build_block_specs(
                sensor_stages=self.sensor_stages[sensor],
                sensor_q_pool=self.sensor_q_pool[sensor],
                sensor_q_stride=self.sensor_q_stride[sensor],
                sensor_mask_unit_attn=self.sensor_mask_unit_attn[sensor],
                sensor_num_heads=self.sensor_num_heads[sensor],
                sensor_mask_unit_size=self.sensor_mask_unit_sizes[sensor],
            )

            local_depth_sensor = int(sum(self.sensor_stages[sensor][:-1]))
            local_specs = block_specs[:local_depth_sensor]
            shared_specs = block_specs[local_depth_sensor:]
            if not shared_specs:
                raise ValueError(
                    f"Shared stage is empty for sensor '{sensor}'. Check sensor stages={self.sensor_stages[sensor]}."
                )

            if shared_specs_ref is None:
                shared_specs_ref = shared_specs
                stage_ends_ref = stage_ends
                local_depth_ref = local_depth_sensor
            else:
                if len(shared_specs) != len(shared_specs_ref):
                    raise ValueError(
                        f"All sensors must use the same shared-stage depth. "
                        f"Sensor '{sensor}' has {len(shared_specs)} shared blocks, "
                        f"reference '{ref_sensor}' has {len(shared_specs_ref)}."
                    )
                compare_keys = ("dim", "dim_out", "heads", "q_stride", "window_size", "use_mask_unit_attn")
                for blk_idx, (cur_blk, ref_blk) in enumerate(zip(shared_specs, shared_specs_ref)):
                    for key in compare_keys:
                        if cur_blk[key] != ref_blk[key]:
                            raise ValueError(
                                f"Shared stage mismatch at sensor '{sensor}', block {blk_idx}, key '{key}': "
                                f"{cur_blk[key]} vs reference {ref_blk[key]}."
                            )

            # Sensor-specific token reordering used by local blocks.
            self.sensor_unroll[sensor] = Unroll(
                self.sensor_input_size[sensor],
                sensor_patch_stride,
                [self.sensor_q_stride[sensor]] * len(stage_ends[:-1]),
            )
            self.sensor_reroll[sensor] = Reroll(
                self.sensor_input_size[sensor],
                sensor_patch_stride,
                [self.sensor_q_stride[sensor]] * len(stage_ends[:-1]),
                stage_ends,
                self.sensor_q_pool[sensor],
            )

            self.sensor_local_blocks[sensor] = nn.ModuleList(
                [HieraBlock(**spec) for spec in local_specs]
            )

            local_out_dim = int(local_specs[-1]["dim_out"]) if local_specs else int(shared_specs[0]["dim"])
            shared_input_dim_sensor = int(shared_specs[0]["dim"])
            if local_out_dim != shared_input_dim_sensor:
                raise ValueError(
                    f"Sensor '{sensor}' local output dim ({local_out_dim}) must match shared input dim "
                    f"({shared_input_dim_sensor})."
                )
            self.sensor_local_output_proj[sensor] = nn.Identity()

        if shared_specs_ref is None or stage_ends_ref is None:
            raise RuntimeError("Failed to build shared specs for multi-sensor backbone.")

        self.stage_ends = stage_ends_ref
        self.local_depth = int(local_depth_ref)
        self.shared_depth = int(len(shared_specs_ref))
        self.unroll = self.sensor_unroll[ref_sensor]
        self.reroll = Reroll(
            self.sensor_input_size[ref_sensor],
            self.sensor_patch_strides[ref_sensor],
            [self.sensor_q_stride[ref_sensor]] * len(self.stage_ends[:-1]),
            self.stage_ends,
            self.sensor_q_pool[ref_sensor],
        )

        self.shared_blocks = nn.ModuleList([HieraBlock(**spec) for spec in shared_specs_ref])
        self.shared_input_dim = int(shared_specs_ref[0]["dim"])
        final_dim = int(shared_specs_ref[-1]["dim_out"])

        self.fusion = SensorTokenFusion(
            sensors=self.sensors,
            embed_dim=self.shared_input_dim,
            mode=fusion_mode,
            attention_heads=fusion_attention_heads,
            projection_dim=fusion_projection_dim,
        )
        self.last_fusion_modality_contributions: Dict[str, float] = {
            s: 0.0 for s in self.sensors
        }
        self.last_spectral_group_contributions: Dict[str, Dict[str, List[float]]] = {}

        self.norm = norm_layer(final_dim)
        self.num_features = final_dim
        self.embed_dim = final_dim

        # CLS insertion mirrors HieraWithCLS logic but on the shared trunk only.
        self.cls_token_inserted_at = -1
        if self.use_cls_token:
            target_dim = self.shared_input_dim
            for i, blk in enumerate(self.shared_blocks):
                if not blk.attn.use_mask_unit_attn:
                    self.cls_token_inserted_at = i
                    target_dim = blk.dim
                    break

            if self.cls_token_inserted_at >= 0:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, target_dim))
                nn.init.trunc_normal_(self.cls_token, std=0.02)
            else:
                self.use_cls_token = False
                self.cls_token = None
        else:
            self.cls_token = None

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module, init_bias: float = 0.0) -> None:
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self) -> List[str]:
        names = ["pos_embed"]
        if self.cls_token is not None:
            names.append("cls_token")
        return names

    def get_pos_embed(self) -> torch.Tensor:
        return self.pos_embed

    def get_random_mask(self, x: Dict[str, torch.Tensor], mask_ratio: float) -> torch.Tensor:
        # Use the first available sensor tensor to infer batch size.
        sensors = self._find_available_sensors(x)
        if not sensors:
            raise ValueError("Cannot generate mask without any available sensor tensor.")
        first_tensor = self._get_sensor_tensor(x, sensors[0])
        bsz = first_tensor.shape[0]

        num_windows = math.prod(self.mask_spatial_shape)
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = torch.rand(bsz, num_windows, device=first_tensor.device)

        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        mask = torch.zeros([bsz, num_windows], device=first_tensor.device)
        mask[:, :len_keep] = 1
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask.bool()

    def _find_available_sensors(self, batch: Dict[str, torch.Tensor]) -> List[str]:
        available: List[str] = []
        for sensor in self.sensors:
            if sensor in batch and isinstance(batch[sensor], torch.Tensor):
                available.append(sensor)
                continue
            lower = sensor.lower()
            if lower in batch and isinstance(batch[lower], torch.Tensor):
                available.append(sensor)
                continue
        return available

    def _get_sensor_tensor(self, batch: Dict[str, torch.Tensor], sensor: str) -> torch.Tensor:
        if sensor in batch:
            return batch[sensor]
        lower = sensor.lower()
        if lower in batch:
            return batch[lower]
        raise KeyError(f"Sensor '{sensor}' not found in input batch.")

    def _preprocess_sensor_tensor(self, x: torch.Tensor, sensor: str) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected sensor tensor [B, C, H, W], got shape {tuple(x.shape)}")
        expected_size = self.sensor_input_size[_normalize_sensor_name(sensor)]
        if tuple(int(v) for v in x.shape[-2:]) != expected_size:
            raise ValueError(
                f"Sensor '{sensor}' tensor must match configured size {expected_size}, "
                f"got {tuple(int(v) for v in x.shape[-2:])}."
            )
        return x

    def _run_sensor_branch(
        self,
        sensor: str,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        mask_map: Optional[torch.Tensor],
        spectral_mask: Optional[torch.Tensor] = None,
        mask_mode: str = "drop",
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if self.sensor_encoder_types[sensor] == "spectral_transformer":
            x = self.sensor_patch_embeds[sensor](x, mask=mask_map, spectral_mask=spectral_mask)
        else:
            x = self.sensor_patch_embeds[sensor](x, mask=mask_map)
        x = x + self.get_pos_embed()
        x = self.sensor_unroll[sensor](x)

        if mask is not None:
            sensor_mu_size = int(self.sensor_mu_sizes[sensor])
            if mask_mode == "drop":
                x = x[mask[..., None].tile(1, sensor_mu_size, x.shape[2])].view(
                    x.shape[0], -1, x.shape[-1]
                )
            elif mask_mode == "keep":
                # Keep full token sequence (needed for iBOT-style patch objectives).
                # Masking signal is already injected at patch-embedding level via mask_map.
                pass
            else:
                raise ValueError(f"Unknown mask_mode '{mask_mode}'. Use 'drop' or 'keep'.")

        local_intermediates: List[torch.Tensor] = []
        for local_block_idx, blk in enumerate(self.sensor_local_blocks[sensor]):
            x = blk(x)
            if return_intermediates and local_block_idx in self.stage_ends:
                local_intermediates.append(
                    self.sensor_reroll[sensor](
                        x,
                        local_block_idx,
                        mask=mask if mask_mode == "drop" else None,
                    )
                )

        x = self.sensor_local_output_proj[sensor](x)
        if return_intermediates:
            return x, local_intermediates
        return x

    def _run_shared_trunk(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        mask_mode: str,
        return_intermediates: bool,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], bool]:
        intermediates: List[torch.Tensor] = []
        cls_active = False

        for i, blk in enumerate(self.shared_blocks):
            global_block_idx = self.local_depth + i

            if self.use_cls_token and i == self.cls_token_inserted_at:
                cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                cls_active = True

            if cls_active and blk.dim != blk.dim_out:
                cls_t = x[:, :1]
                x_spatial = x[:, 1:]

                if hasattr(blk, "proj"):
                    cls_t = blk.proj(blk.norm1(cls_t))

                x_spatial = blk(x_spatial)
                x = torch.cat((cls_t, x_spatial), dim=1)
            else:
                x = blk(x)

            if return_intermediates and global_block_idx in self.stage_ends:
                spatial_tokens = x[:, 1:] if cls_active else x
                intermediates.append(
                    self.reroll(
                        spatial_tokens,
                        global_block_idx,
                        mask=mask if mask_mode == "drop" else None,
                    )
                )

        return x, intermediates, cls_active

    def forward(
        self,
        x: Union[Dict[str, torch.Tensor], torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        mask_mode: str = "drop",
        return_intermediates: bool = False,
        spectral_mask_by_sensor: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # Allow tensor input for quick debugging with a single configured sensor.
        if isinstance(x, torch.Tensor):
            x = {self.sensors[0]: x}

        if not isinstance(x, dict):
            raise TypeError("Input must be a dict sensor->tensor or a 4D tensor.")

        self.last_fusion_modality_contributions = {s: 0.0 for s in self.sensors}
        self.last_spectral_group_contributions = {}

        available_sensors = self._find_available_sensors(x)
        if not available_sensors:
            raise ValueError(
                f"No configured sensors found in input batch. Configured={self.sensors}, keys={list(x.keys())}"
            )

        first_sensor = available_sensors[0]
        first = self._preprocess_sensor_tensor(
            self._get_sensor_tensor(x, first_sensor),
            sensor=first_sensor,
        )
        batch_size = first.shape[0]

        if mask is not None:
            expected_windows = math.prod(self.mask_spatial_shape)
            if mask.shape != (batch_size, expected_windows):
                raise ValueError(
                    f"Mask shape must be {(batch_size, expected_windows)}, got {tuple(mask.shape)}"
                )
            mask_map = mask.view(batch_size, 1, *self.mask_spatial_shape)
        else:
            mask_map = None

        tokens_per_sensor: Dict[str, torch.Tensor] = {}
        local_intermediate_sensors: List[str] = []
        if return_intermediates:
            if self.canonical_sensors:
                # Only use canonical sensors that are actually available in this batch;
                # silently skip missing ones (e.g. SAR not provided for optical-only datasets).
                local_intermediate_sensors = [
                    s for s in self.canonical_sensors if s in available_sensors
                ]
            elif len(available_sensors) == 1:
                # Keep previous default behavior for single-sensor forwarding.
                local_intermediate_sensors = [available_sensors[0]]
        local_intermediate_sensor_set = set(local_intermediate_sensors)
        local_intermediates_by_sensor: Dict[str, List[torch.Tensor]] = {}

        for sensor in available_sensors:
            xi = self._preprocess_sensor_tensor(
                self._get_sensor_tensor(x, sensor),
                sensor=sensor,
            )
            if xi.shape[0] != batch_size:
                raise ValueError(
                    f"All sensor tensors must have same batch size. "
                    f"Expected {batch_size}, got {xi.shape[0]} for sensor {sensor}."
                )
            sensor_spectral_mask: Optional[torch.Tensor] = None
            if spectral_mask_by_sensor is not None:
                if sensor in spectral_mask_by_sensor:
                    sensor_spectral_mask = spectral_mask_by_sensor[sensor]
                elif sensor.lower() in spectral_mask_by_sensor:
                    sensor_spectral_mask = spectral_mask_by_sensor[sensor.lower()]

                if sensor_spectral_mask is not None:
                    sensor_spectral_mask = sensor_spectral_mask.to(device=xi.device, dtype=torch.bool)
                    if self.sensor_encoder_types[sensor] == "spectral_transformer":
                        groups = int(getattr(self.sensor_patch_embeds[sensor], "G"))
                        expected = (
                            batch_size,
                            int(math.prod(self.sensor_tokens_spatial_shape[sensor])),
                            groups,
                        )
                        if tuple(sensor_spectral_mask.shape) != expected:
                            raise ValueError(
                                f"spectral mask for sensor {sensor} must have shape {expected}, "
                                f"got {tuple(sensor_spectral_mask.shape)}"
                            )
                    else:
                        # Linear spectral sensors do not use spectral masks.
                        sensor_spectral_mask = None

            sensor_out = self._run_sensor_branch(
                sensor,
                xi,
                mask,
                mask_map,
                spectral_mask=sensor_spectral_mask,
                mask_mode=mask_mode,
                return_intermediates=sensor in local_intermediate_sensor_set,
            )

            if isinstance(sensor_out, tuple):
                sensor_tokens, sensor_local_intermediates = sensor_out
                local_intermediates_by_sensor[sensor] = sensor_local_intermediates
            else:
                sensor_tokens = sensor_out
            tokens_per_sensor[sensor] = sensor_tokens

            if self.sensor_encoder_types[sensor] == "spectral_transformer":
                patch_embed = self.sensor_patch_embeds[sensor]
                sensor_stats: Dict[str, List[float]] = {}
                energy = getattr(patch_embed, "last_group_energy_contrib", None)
                pooling = getattr(patch_embed, "last_group_pool_contrib", None)
                if energy is not None:
                    sensor_stats["energy"] = [float(v) for v in energy.tolist()]
                if pooling is not None:
                    sensor_stats["pooling"] = [float(v) for v in pooling.tolist()]
                if sensor_stats:
                    self.last_spectral_group_contributions[sensor] = sensor_stats

        fused_tokens = self.fusion(tokens_per_sensor, available_sensors)
        self.last_fusion_modality_contributions = {
            _normalize_sensor_name(k): float(v)
            for k, v in getattr(
                self.fusion, "last_modality_contributions", {}
            ).items()
        }
        local_intermediates: List[torch.Tensor] = []
        if local_intermediate_sensors:
            if len(local_intermediate_sensors) == 1:
                local_intermediates = local_intermediates_by_sensor.get(
                    local_intermediate_sensors[0], []
                )
            else:
                per_sensor_counts: List[int] = []
                for sensor in local_intermediate_sensors:
                    if sensor not in local_intermediates_by_sensor:
                        raise RuntimeError(
                            f"Missing local intermediates for canonical sensor '{sensor}'."
                        )
                    per_sensor_counts.append(len(local_intermediates_by_sensor[sensor]))
                expected_count = per_sensor_counts[0] if per_sensor_counts else 0
                if any(c != expected_count for c in per_sensor_counts):
                    raise RuntimeError(
                        "Canonical sensors have inconsistent numbers of local intermediates: "
                        f"{dict(zip(local_intermediate_sensors, per_sensor_counts))}"
                    )

                for stage_idx in range(expected_count):
                    stage_maps: List[torch.Tensor] = []
                    ref_shape_hw: Optional[Tuple[int, int, int]] = None
                    for sensor in local_intermediate_sensors:
                        fmap = local_intermediates_by_sensor[sensor][stage_idx]
                        shape_hw = (
                            int(fmap.shape[0]),
                            int(fmap.shape[1]),
                            int(fmap.shape[2]),
                        )
                        if ref_shape_hw is None:
                            ref_shape_hw = shape_hw
                        elif shape_hw != ref_shape_hw:
                            raise RuntimeError(
                                f"Cannot concatenate local intermediates at stage {stage_idx}: "
                                f"sensor '{sensor}' has shape prefix {shape_hw}, "
                                f"reference is {ref_shape_hw}."
                            )
                        stage_maps.append(fmap)
                    local_intermediates.append(torch.cat(stage_maps, dim=-1))

        x_out, shared_intermediates, cls_active = self._run_shared_trunk(
            fused_tokens, mask, mask_mode, return_intermediates
        )
        intermediates = local_intermediates + shared_intermediates

        if mask is None:
            x_out = self.norm(x_out)
            x_out = x_out[:, 0] if cls_active else x_out.mean(dim=1)

        if return_intermediates:
            return x_out, intermediates
        return x_out

    def get_intermediate_layers(
        self,
        x: Union[Dict[str, torch.Tensor], torch.Tensor],
        n: Union[int, Sequence[int]] = 0,
        reshape: bool = False,
        norm: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        _, intermediates = self.forward(x, return_intermediates=True)

        if isinstance(n, int):
            outputs = intermediates[-n:] if n > 0 else intermediates
        else:
            outputs = intermediates

        if norm:
            outputs = [F.layer_norm(out, (out.shape[-1],)) for out in outputs]

        if reshape:
            outputs = [out.permute(0, 3, 1, 2).contiguous() for out in outputs]

        return tuple(outputs)
    

class MultiSensorTwoStagesHiera_Encoder(Encoder):
    """
    MultiSensorTwoStagesHiera Encoder wrapper compatible with repository structure.
    Returns list of spatial feature maps from selected stages.
    """

    def __init__(
        self,
        encoder_weights,
        download_url,
        canonical_sensor,
        input_bands,
        output_layers,
        output_dim,
        sensors,
        sensor_in_chans,
        sensor_input_size,
        embed_dim,
        multi_temporal=False,
        multi_temporal_output=False,
        pyramid_output=True,
        rescales=None,
        **backbone_kwargs,
    ):
        # --------------------------------------------------
        # Normalize canonical_sensor to a list internally,
        # but keep a primary (first) sensor for input_size.
        # --------------------------------------------------
        if isinstance(canonical_sensor, str):
            canonical_sensor_list = [canonical_sensor]
        elif isinstance(canonical_sensor, Sequence) and not isinstance(canonical_sensor, str):
            canonical_sensor_list = [str(s) for s in canonical_sensor]
        else:
            raise ValueError(
                "canonical_sensor must be a sensor string or a list of sensor names."
            )

        if not canonical_sensor_list:
            raise ValueError("canonical_sensor must not be empty.")

        for cs in canonical_sensor_list:
            if cs not in sensor_input_size:
                raise ValueError(
                    f"canonical_sensor entry '{cs}' not found in sensor_input_size."
                )

        # --------------------------------------------------
        # Derive input_size from the first canonical sensor
        # --------------------------------------------------
        primary_sensor = canonical_sensor_list[0]
        canonical_size = sensor_input_size[primary_sensor]

        if len(canonical_size) != 2:
            raise ValueError(
                f"sensor_input_size[{primary_sensor}] must be [H, W]"
            )

        input_size = canonical_size[0]

        # --------------------------------------------------
        # Initialize base Encoder
        # --------------------------------------------------
        Encoder.__init__(
            self,
            model_name="multisensor_two_stages_hiera_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=embed_dim,
            output_layers=output_layers,
            output_dim=output_dim,
            multi_temporal=multi_temporal,
            multi_temporal_output=multi_temporal_output,
            pyramid_output=pyramid_output,
            download_url=download_url,
        )

        # --------------------------------------------------
        # Build backbone
        # --------------------------------------------------
        self.backbone = MultiSensorTwoStagesHiera(
            sensors=sensors,
            sensor_in_chans=sensor_in_chans,
            sensor_input_size=sensor_input_size,
            embed_dim=embed_dim,
            canonical_sensor=canonical_sensor,
            **backbone_kwargs,
        )

        # Track stage ends like HIERA_Encoder
        self.stage_indices = self.backbone.stage_ends
        # Keep the original value (str or list) for the backbone;
        # expose the list and the primary sensor for the wrapper.
        self.canonical_sensor = canonical_sensor
        self.canonical_sensor_list = canonical_sensor_list
        self.primary_sensor = primary_sensor
        # Optional rescales to be consumed by the decoder (e.g. SegUPerNet).
        self.rescales = [float(r) for r in rescales] if rescales is not None else None

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, images):
        # --------------------------------------------------
        # Repository passes {"optical": tensor}
        # Convert to canonical sensor format using the
        # primary (first) canonical sensor name.
        # --------------------------------------------------
        # Map generic modality keys ("optical", "sar") to canonical sensor names.
        # Make a shallow copy only if we need to modify the dict.
        if any(k in images for k in ("optical", "sar")):
            images = dict(images)  # shallow copy to avoid mutating caller

        if "optical" in images:
            # Map "optical" to the first canonical sensor that is not yet present.
            for cs in self.canonical_sensor_list:
                if cs not in images and cs.lower() not in images:
                    images[cs] = images.pop("optical")
                    break

        if "sar" in images:
            # Map "sar" to the first canonical sensor (after optical) that is not yet present.
            for cs in self.canonical_sensor_list:
                if cs not in images and cs.lower() not in images:
                    images[cs] = images.pop("sar")
                    break

        _, intermediates = self.backbone(
            images,
            return_intermediates=True,
        )

        output = []

        for i, feat in enumerate(intermediates):
            if i in self.output_layers:
                feat = feat.permute(0, 3, 1, 2).contiguous()
                output.append(feat)

        return output

    # ---------------------------------------------------------
    # Weight Loading
    # ---------------------------------------------------------
    def load_encoder_weights(self, logger) -> None:
        if self.encoder_weights is None:
            return

        ckpt = torch.load(self.encoder_weights, map_location="cpu")

        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            ckpt,
            strict=True,
        )

        if missing_keys:
            logger.warning(f"Missing keys (not loaded): {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys (ignored): {unexpected_keys}")