import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer


def MultiStepLR(
    optimizer: Optimizer, total_iters: int, lr_milestones: list[float]
) -> LRScheduler:
    
    """
    This module provides a utility function for creating a multi-step learning rate scheduler.

    Functions:
        MultiStepLR(optimizer: Optimizer, total_iters: int, lr_milestones: list[float]) -> LRScheduler
            Creates a MultiStepLR scheduler with specified milestones and decay factor.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        total_iters (int): The total number of iterations for training.
        lr_milestones (list[float]): A list of fractions representing the milestones at which the learning rate will be decayed.

    Returns:
        LRScheduler: A PyTorch learning rate scheduler that decays the learning rate at specified milestones.
    """

    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, [int(total_iters * r) for r in lr_milestones], gamma=0.1
    )


def OneCycleLR(
    optimizer: Optimizer, total_iters: int, max_lr: float
) -> LRScheduler:
    """OneCycleLR with pct_start=0 (no warmup) and cosine annealing.

    Matches the Panopticon paper training schedule: LR decays from max_lr
    to ~0 following a cosine curve starting immediately at step 0.

    Args:
        optimizer: The optimizer to schedule.
        total_iters: Total number of training iterations.
        max_lr: Peak (starting) learning rate — set equal to optimizer.lr.

    Returns:
        LRScheduler: OneCycleLR scheduler.
    """
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_iters,
        pct_start=0.0,
        anneal_strategy="cos",
    )


def CosineAnnealingLR(
    optimizer: Optimizer, total_iters: int
) -> LRScheduler:
    """Creates a CosineAnnealingLR scheduler that decays LR from its initial
    value to 0 following a cosine curve over total_iters steps.

    Args:
        optimizer (Optimizer): The optimizer to schedule.
        total_iters (int): Total number of training iterations (T_max).

    Returns:
        LRScheduler: CosineAnnealingLR scheduler.
    """
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_iters, eta_min=0
    )

