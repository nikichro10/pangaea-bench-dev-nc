# Adapted from: https://github.com/gastruc/UniverSat (commit f6df2eec54955b0f7524cc95fe21a5e80c0239d9)

from functools import partial
from logging import Logger
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from pangaea.encoders.base import Encoder

_UNIVERSAT_REPO = "gastruc/UniverSat:f6df2eec54955b0f7524cc95fe21a5e80c0239d9"

# (dataset, modality) pairs the released checkpoint was pretrained with. The
# constructor instantiates one MLP projector per non-"modis" entry, so this
# set must match the checkpoint or load_state_dict will complain. We never
# route through these projectors at inference (dataset="" everywhere below),
# but the submodules must still exist for the state dict to line up.
_DEFAULT_MODALITIES_DICT = {
    "flair": ["spotRGBN", "aerialflair", "s2flair", "s1flair", "dem"],
    "pastishd": ["spot", "s2", "s1"],
    "planted": ["s2", "s1", "l7", "alos", "modis"],
    "tsaits": ["aerial", "s2", "s1"],
    "s2naip": ["naip", "l8", "s2", "s1"],
    "hyperglobal": ["EO1"],
    "earthview": ["rgbneon", "ndemneon", "neon"],
    "spectralearth": ["enmap"],
}


def _load_universat_classes():
    """Fetch UniverSat's source (pinned commit) via torch.hub and import its classes.

    ``torch.hub.help`` imports the repo's ``hubconf.py`` (which inserts its
    ``src/`` onto ``sys.path``) without constructing any model -- cheaper
    than ``torch.hub.load`` and side-effect-free beyond the path setup.
    """
    torch.hub.help(_UNIVERSAT_REPO, "universat", trust_repo=True)
    from models.networks.encoder.UniversalPatchEncoder import UniversalPatchEncoder
    from models.networks.encoder.UniverSat import UniverSat as _UniverSatModel
    from models.networks.encoder.utils.pos_embed import get_coords

    return _UniverSatModel, UniversalPatchEncoder, get_coords


class UniverSatEncoder(Encoder):
    """Frozen UniverSat (AnySat v2) encoder, Base size.

    UniverSat's public forward pass only exposes the representation after
    the *entire* trunk (Bi_ACA_in -> 12x self-attention (SA) -> Bilinear_out
    -> CA_Sub), with no hook for intermediate depths the way a plain ViT
    stack allows. To get pangaea's usual multi-depth ``output_layers``
    feature list, this class re-implements the trunk loop (see
    ``_forward_trunk``, adapted from ``UniverSat.ViT_forward`` at the pinned
    commit above) so it can snapshot real intermediate SA-block activations.

    ``output_grid`` is set equal to ``latent_grid`` so ``Bilinear_out``
    becomes a no-op resize -- but the deepest requested level still runs
    through ``CA_Sub``, the model's own sub-patch skip-connection stage, so
    that last feature keeps the fine-detail enrichment the model was
    actually trained to produce (rather than being just another raw SA
    snapshot). This was verified to bit-match calling the official
    ``ViT_forward`` directly for that last level.

    Because of this, ``output_layers`` must be given in ascending order and
    its last entry must equal 11 (the final one of the 12 SA blocks,
    0-indexed) -- that's the only depth at which CA_Sub is architecturally
    reachable.
    """

    def __init__(
        self,
        encoder_weights: str | Path,
        input_size: int,
        input_bands: dict[str, list[str]],
        output_layers: list[int],
        output_dim: int,
        download_url: str,
        wave_list: dict[str, dict[str, float | str]],
        input_res: dict[str, float],
        patch_size_m: float = 160.0,
    ) -> None:
        super().__init__(
            model_name="universat_encoder",
            encoder_weights=encoder_weights,
            input_bands=input_bands,
            input_size=input_size,
            embed_dim=768,
            output_dim=output_dim,
            output_layers=output_layers,
            multi_temporal=False,
            multi_temporal_output=False,
            pyramid_output=False,
            download_url=download_url,
        )

        n_sa_blocks = 12
        if sorted(self.output_layers) != list(self.output_layers) or self.output_layers[-1] != n_sa_blocks - 1:
            raise ValueError(
                f"output_layers must be ascending and end at {n_sa_blocks - 1} "
                f"(the final SA block) so CA_Sub enrichment is reachable; got {self.output_layers}"
            )

        _UniverSatModel, UniversalPatchEncoder, self._get_coords = _load_universat_classes()

        embed_dim = self.embed_dim
        spatial_embed = embed_dim // 8
        spatial_encoder = partial(
            UniversalPatchEncoder,
            embed_dim=spatial_embed,
            final_dim=embed_dim,
            n_queries=[1, 1, 1, 1],
            expand_dim=[2, 2, 2, 2],
            order=["S1", "C", "T", "S"],
            num_heads=12,
            mlp_ratio=4.0,
            attn_drop_rate=0.0,
            gating=True,
        )
        # Fixed Base-size recipe, matching the released checkpoint exactly
        # (verified: load_state_dict reports 0 missing / 0 unexpected keys).
        self.universat = _UniverSatModel(
            spatial_encoder=spatial_encoder,
            block_type=["Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub"],
            embed_dim=embed_dim,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=False,
            n_registers=4,
            pre_norm=False,
            drop_rate=0.0,
            drop_path_rate=0.0,
            attn_drop_rate=0.0,
            gating=True,
            proba_drop_modalities=0.0,
            modalities_dict=_DEFAULT_MODALITIES_DICT,
        )

        self.wave_list = wave_list
        self.scale = patch_size_m / 10.0
        self.input_res = input_res
        self.subpatches = {modality: 1 for modality in input_bands}

        self.wv_list = self._build_wavelengths(self.input_bands)

    def _build_wavelengths(self, data_bands: dict[str, list[str]]) -> dict[str, list]:
        return {
            modality: [self.wave_list[modality][band] for band in bands]
            for modality, bands in data_bands.items()
        }

    def set_wv_list_from_data_bands(self, data_bands: dict[str, list[str]]) -> None:
        """Restrict wavelengths to bands actually present in the dataset.

        Call this when BandPadding is NOT used, so UniverSat receives only
        the real bands with their true wavelengths/codes rather than
        zero-padded phantom channels (mirrors dofa_encoder.py/
        panopticon_encoder.py).
        """
        restricted = {
            modality: [band for band in bands if band in data_bands[modality]]
            for modality, bands in self.input_bands.items()
            if modality in data_bands
        }
        restricted = {modality: bands for modality, bands in restricted.items() if bands}
        if not restricted:
            raise ValueError("No bands remain after filtering to available data bands.")
        self.wv_list = self._build_wavelengths(restricted)

    def _forward_trunk(
        self, tokens, spatial, coords_spatial, coords_in, token_sizes, latent_grid, modalities
    ) -> list[torch.Tensor]:
        universat = self.universat
        B = tokens[0].shape[0]
        n_modalities = len(spatial)

        spatial = torch.cat(spatial, dim=1).detach()
        coords_spatial = torch.cat(coords_spatial, dim=1)
        coords_in = torch.cat(coords_in, dim=1)

        # output_grid == latent_grid by construction: Bilinear_out becomes a
        # no-op resize below, so coords_out == coords.
        coords = self._get_coords(tokens[0], int(latent_grid**0.5), 1, universat.n_registers, 1)

        output_layers = set(self.output_layers)
        max_sa_index = 11  # last of the 12 SA blocks, 0-indexed

        intermediates = []
        sa_index = -1
        for i, blk in enumerate(universat.blocks):
            bt = universat.block_type[i]
            if bt == "Bi_ACA_in":
                tokens_out = []
                for t, m, s in zip(tokens, modalities, token_sizes):
                    if s != int(latent_grid**0.5):
                        t = t.view(B, s, s, universat.embed_dim).permute(0, 3, 1, 2)
                        out_size = (int(latent_grid**0.5), int(latent_grid**0.5))
                        t = F.interpolate(t, size=out_size, mode="bilinear", align_corners=True)
                        t = t.permute(0, 2, 3, 1).flatten(1, 2)
                    tokens_out.append(t)
                tokens = torch.stack(tokens_out, dim=2)
                Bb, N, M, D = tokens.shape
                tokens = tokens.flatten(0, 1)
                tokens = blk(tokens)
                tokens = tokens.view(Bb, N, D)
                if universat.registers is not None:
                    registers = universat.registers.expand(tokens.shape[0], -1, -1)
                    tokens = torch.cat((registers, tokens), dim=1)
            elif bt == "SA":
                sa_index += 1
                tokens = blk(tokens, coords)
                # The deepest requested level is handled after the loop via
                # CA_Sub instead of a raw snapshot here (see class docstring).
                if sa_index in output_layers and sa_index != max_sa_index:
                    intermediates.append(tokens[:, universat.n_registers :])
            elif bt == "Bilinear_out":
                pass  # no-op: output_grid == latent_grid by construction
            elif bt == "CA_Sub":
                tokens = blk(
                    tokens,
                    coords,
                    spatial,
                    coords_spatial,
                    n_registers=0,
                    n_registers_q=universat.n_registers,
                    n_modalities=n_modalities,
                )
                intermediates.append(tokens[:, universat.n_registers :])
            else:
                raise ValueError(f"Unsupported block type for frozen UniverSat forward: {bt}")

        return intermediates

    def forward(self, image: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        modalities = [m for m in self.wv_list if m in image]
        x = {m: image[m].squeeze(2) for m in modalities}

        B, _, H, _ = x[modalities[0]].shape
        patch_px = max(int(self.scale * 10 / self.input_res[modalities[0]]), 1)
        grid_side = H // patch_px
        latent_grid = grid_side**2

        (
            tokens,
            coords_in,
            token_sizes,
            spatial,
            coords_spatial,
            _,
            kept_modalities,
        ) = self.universat.UPE_forward(
            x, modalities, self.wv_list, self.input_res, self.scale, self.subpatches,
            mask_in=None, dataset="",
        )

        intermediates = self._forward_trunk(
            tokens, spatial, coords_spatial, coords_in, token_sizes, latent_grid, kept_modalities
        )

        return [
            feat.permute(0, 2, 1).view(B, self.embed_dim, grid_side, grid_side).contiguous()
            for feat in intermediates
        ]

    def load_encoder_weights(self, logger: Logger) -> None:
        state_dict = load_file(self.encoder_weights)
        prefix = "model."
        stripped = {
            name[len(prefix) :]: param
            for name, param in state_dict.items()
            if name.startswith(prefix)
        }

        missing = {}
        incompatible_shape = {}
        pretrained = {}
        for name, param in self.universat.named_parameters():
            if name not in stripped:
                missing[name] = param.shape
            elif stripped[name].shape != param.shape:
                incompatible_shape[name] = (param.shape, stripped[name].shape)
            else:
                pretrained[name] = stripped[name]

        self.universat.load_state_dict(pretrained, strict=False)
        self.parameters_warning(missing, incompatible_shape, logger)
