"""
Run test evaluation on a saved checkpoint and generate segmentation visualizations.

Usage (via torchrun):
    torchrun --standalone --nnodes=1 --nproc_per_node=1 scripts/generate_seg_viz.py \
        --ckpt_dir <path_to_run_dir> \
        --save_preds_dir <output_dir> \
        [--use_final_ckpt]
"""

import argparse
import pathlib
import sys

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pangaea.utils.collate_fn import get_collate_fn
from pangaea.datasets.base import GeoFMDataset, RawGeoFMDataset
from pangaea.utils.utils import get_best_model_ckpt_path, get_final_model_ckpt_path
from pangaea.utils.logger import init_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--save_preds_dir", required=True)
    parser.add_argument("--use_final_ckpt", action="store_true")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(dist.get_rank())
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    exp_dir = pathlib.Path(args.ckpt_dir)
    save_preds_dir = pathlib.Path(args.save_preds_dir)
    save_preds_dir.mkdir(parents=True, exist_ok=True)

    logger = init_logger(exp_dir / "viz.log", rank=local_rank)

    cfg = OmegaConf.load(exp_dir / "configs" / "config.yaml")
    logger.info(f"Loaded config from {exp_dir}/configs/config.yaml")
    logger.info(f"Saving visualizations to {save_preds_dir}")

    encoder = instantiate(cfg.encoder)
    encoder.load_encoder_weights(logger)

    uses_band_padding = "BandPadding" in OmegaConf.to_yaml(cfg.preprocessing)
    if not uses_band_padding and hasattr(encoder, "set_wv_list_from_data_bands"):
        data_bands = OmegaConf.to_container(cfg.dataset.bands, resolve=True)
        encoder.set_wv_list_from_data_bands(data_bands)

    decoder = instantiate(cfg.decoder, encoder=encoder)
    decoder.to(device)
    decoder = torch.nn.parallel.DistributedDataParallel(
        decoder, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=cfg.finetune,
    )

    modalities = list(encoder.input_bands.keys())
    collate_fn = get_collate_fn(modalities)

    test_preprocessor = instantiate(
        cfg.preprocessing.test,
        dataset_cfg=cfg.dataset,
        encoder_cfg=cfg.encoder,
        _recursive_=False,
    )
    raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
    test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)
    test_loader = DataLoader(
        test_dataset,
        sampler=DistributedSampler(test_dataset),
        batch_size=cfg.test_batch_size,
        num_workers=cfg.test_num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    # Patch evaluator config: enable save_preds_dir, disable wandb
    evaluator_cfg = OmegaConf.to_container(cfg.task.evaluator, resolve=True)
    evaluator_cfg["save_preds_dir"] = str(save_preds_dir)
    evaluator_cfg["use_wandb"] = False

    test_evaluator = instantiate(
        OmegaConf.create(evaluator_cfg),
        val_loader=test_loader,
        exp_dir=exp_dir,
        device=device,
    )

    ckpt_path = get_final_model_ckpt_path(exp_dir) if args.use_final_ckpt else get_best_model_ckpt_path(exp_dir)
    logger.info(f"Using checkpoint: {ckpt_path}")
    test_evaluator.evaluate(decoder, "test_model", ckpt_path)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
