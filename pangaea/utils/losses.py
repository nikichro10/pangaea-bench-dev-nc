import torch
from torch.nn import functional as F


class WeightedCrossEntropy(torch.nn.Module):
    def __init__(self, ignore_index: int, distribution: list[float]) -> None:
        super(WeightedCrossEntropy, self).__init__()
        # Initialize the weights based on the given distribution
        self.weights = [1 / w for w in distribution]

        # Convert weights to a tensor and move to CUDA
        loss_weights = torch.Tensor(self.weights).to("cuda")
        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=loss_weights
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute the weighted cross-entropy loss
        return self.loss(logits, target)


class WeightedCrossEntropyAuto(torch.nn.Module):
    """Cross-entropy with inverse-frequency class weights computed per batch.

    Weights are derived from pixel counts in each batch, so no pre-computed
    class distribution is needed. Rare classes (e.g. cattle, tree) receive
    higher weight proportional to how underrepresented they are.

    Args:
        ignore_index (int): Label value to ignore in loss and weight computation.
        num_classes (int): Total number of classes.
    """

    def __init__(self, ignore_index: int, num_classes: int) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target != self.ignore_index
        valid = target[mask]
        counts = torch.bincount(valid.flatten(), minlength=self.num_classes).float()
        # inverse frequency; clamp avoids division by zero for absent classes
        weights = 1.0 / counts.clamp(min=1)
        # normalize so weights sum to num_classes (keeps loss scale stable)
        weights = weights * (self.num_classes / weights.sum())
        return F.cross_entropy(
            logits, target, weight=weights.to(logits.device), ignore_index=self.ignore_index
        )


class DICELoss(torch.nn.Module):
    def __init__(self, ignore_index: int) -> None:
        super(DICELoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        num_classes = logits.shape[1]

        # Convert logits to probabilities using softmax or sigmoid
        if num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        # Create a mask to ignore the specified index
        mask = target != self.ignore_index
        target = target.clone()
        target[~mask] = 0

        # Convert target to one-hot encoding if necessary
        if num_classes == 1:
            target = target.unsqueeze(1)
        else:
            target = F.one_hot(target, num_classes=num_classes)
            target = target.permute(0, 3, 1, 2)

        # Apply the mask to the target
        target = target.float() * mask.unsqueeze(1).float()
        intersection = torch.sum(probs * target, dim=(2, 3))
        union = torch.sum(probs + target, dim=(2, 3))

        # Compute the Dice score
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        valid_dice = dice_score[mask.any(dim=1).any(dim=1)]
        dice_loss = 1 - valid_dice.mean()  # Dice loss is 1 minus the Dice score

        return dice_loss

