"""
Loss Functions for Enhanced MDF-Net

Implements:
- Weighted Binary Cross-Entropy (BCE) for class imbalance
- Focal Loss for hard examples and rare classes
- Combined Loss (0.7 BCE + 0.3 Focal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for Multi-Label Classification

    Handles class imbalance by weighting positive examples according to
    their rarity in the dataset.
    """

    def __init__(self, pos_weights: Optional[torch.Tensor] = None):
        """
        Args:
            pos_weights: [num_classes] - Positive class weights
                         Default: uniform (all ones)
        """
        super().__init__()
        self.pos_weights = pos_weights

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, num_classes] - Sigmoid probabilities (0-1)
            targets: [B, num_classes] - Binary ground truth (0 or 1)

        Returns:
            loss: Scalar loss value
        """
        # More aggressive clamping for numerical stability
        predictions = torch.clamp(predictions, min=1e-6, max=1 - 1e-6)

        if self.pos_weights is None:
            # Uniform weights
            pos_weights = torch.ones_like(predictions)
        else:
            # Expand pos_weights to batch dimension
            pos_weights = self.pos_weights.unsqueeze(0).expand_as(predictions)

        # Binary cross-entropy with positive class weighting
        # BCE = -[w * y * log(p) + (1-y) * log(1-p)]
        loss = -(
            pos_weights * targets * torch.log(predictions + 1e-8) +  # Added epsilon
            (1 - targets) * torch.log(1 - predictions + 1e-8)  # Added epsilon
        )

        return loss.mean()

    def update_pos_weights(self, pos_weights: torch.Tensor):
        """Update positive class weights (e.g., after computing from training data)"""
        self.pos_weights = pos_weights


class FocalLoss(nn.Module):
    """
    Focal Loss for Hard Example Mining

    Focuses training on hard-to-classify examples by down-weighting
    easy examples. Particularly useful for extremely imbalanced classes.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class (0-1)
            gamma: Focusing parameter (higher = more focus on hard examples)
                  Typical values: 1.0-3.0, recommended: 2.0
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: [B, num_classes] - Sigmoid probabilities (0-1)
            targets: [B, num_classes] - Binary ground truth (0 or 1)

        Returns:
            loss: Scalar loss value
        """
        # More aggressive clamping for numerical stability
        predictions = torch.clamp(predictions, min=1e-6, max=1 - 1e-6)

        # Standard binary cross-entropy with epsilon for safety
        bce = -(targets * torch.log(predictions + 1e-8) + (1 - targets) * torch.log(1 - predictions + 1e-8))

        # Compute p_t: probability of correct class
        # p_t = p if y=1, else 1-p
        p_t = predictions * targets + (1 - predictions) * (1 - targets)

        # Focal weight: (1 - p_t)^gamma
        # Easy examples (p_t → 1) get very small weight
        # Hard examples (p_t → 0) get large weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting: balance positive/negative examples
        # alpha_t = alpha if y=1, else 1-alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Final focal loss
        loss = alpha_t * focal_weight * bce

        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined Loss Function: Weighted BCE + Focal Loss

    Combines the strengths of both losses:
    - BCE handles general class imbalance
    - Focal Loss focuses on hard examples and rare classes

    Default weights: 0.7 BCE + 0.3 Focal
    """

    def __init__(self,
                 pos_weights: Optional[torch.Tensor] = None,
                 lambda_bce: float = 0.7,
                 lambda_focal: float = 0.3,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        """
        Args:
            pos_weights: [num_classes] - Positive class weights for BCE
            lambda_bce: Weight for BCE loss
            lambda_focal: Weight for focal loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()

        self.lambda_bce = lambda_bce
        self.lambda_focal = lambda_focal

        self.bce_loss = WeightedBCELoss(pos_weights=pos_weights)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Args:
            predictions: [B, num_classes] - Sigmoid probabilities (0-1)
            targets: [B, num_classes] - Binary ground truth (0 or 1)

        Returns:
            Dictionary with:
                - 'loss': Total combined loss
                - 'bce_loss': BCE component
                - 'focal_loss': Focal component
        """
        bce = self.bce_loss(predictions, targets)
        focal = self.focal_loss(predictions, targets)

        combined = self.lambda_bce * bce + self.lambda_focal * focal

        return {
            'loss': combined,
            'bce_loss': bce.item(),
            'focal_loss': focal.item()
        }

    def update_pos_weights(self, pos_weights: torch.Tensor):
        """Update positive class weights"""
        self.bce_loss.update_pos_weights(pos_weights)


def compute_pos_weights(targets: torch.Tensor, epsilon: float = 1e-5) -> torch.Tensor:
    """
    Compute positive class weights from training data

    pos_weight[i] = n_negative[i] / n_positive[i]

    Args:
        targets: [N, num_classes] - Binary labels from training set
        epsilon: Small value to avoid division by zero

    Returns:
        pos_weights: [num_classes] - Positive class weights
    """
    # Count positive and negative examples per class
    n_positive = targets.sum(dim=0)  # [num_classes]
    n_negative = targets.size(0) - n_positive  # [num_classes]

    # Compute weights (avoid division by zero)
    pos_weights = n_negative / (n_positive + epsilon)

    # Clip extreme values
    pos_weights = torch.clamp(pos_weights, min=0.1, max=100.0)

    return pos_weights


def compute_class_weights_from_counts(class_counts: dict, total_samples: int) -> torch.Tensor:
    """
    Compute class weights from count dictionary

    Example:
        class_counts = {
            'Atelectasis': 1500,
            'Cardiomegaly': 1200,
            ...
        }

    Args:
        class_counts: Dictionary mapping class names to positive sample counts
        total_samples: Total number of samples in dataset

    Returns:
        pos_weights: [num_classes] - Positive class weights
    """
    num_classes = len(class_counts)
    pos_weights = torch.zeros(num_classes)

    for idx, (class_name, pos_count) in enumerate(class_counts.items()):
        neg_count = total_samples - pos_count
        pos_weights[idx] = neg_count / (pos_count + 1e-5)

    # Clip extreme values
    pos_weights = torch.clamp(pos_weights, min=0.1, max=100.0)

    return pos_weights


# Example positive weights based on CheXpert distribution
CHEXPERT_POS_WEIGHTS = torch.tensor([
    1.0,    # No Finding
    2.8,    # Atelectasis
    3.5,    # Cardiomegaly
    8.2,    # Consolidation
    4.1,    # Edema
    12.3,   # Enlarged Cardiomediastinum
    45.7,   # Fracture
    67.4,   # Lung Lesion
    1.9,    # Lung Opacity
    3.2,    # Pleural Effusion
    89.3,   # Pleural Other
    15.6,   # Pneumonia
    21.4,   # Pneumothorax
    1.6     # Support Devices
])


if __name__ == '__main__':
    # Test loss functions
    batch_size = 32
    num_classes = 14

    # Dummy data
    predictions = torch.rand(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    print("Testing Loss Functions\n" + "=" * 50)

    # Test Weighted BCE
    print("\n1. Weighted BCE Loss:")
    bce_loss = WeightedBCELoss(pos_weights=CHEXPERT_POS_WEIGHTS)
    bce_val = bce_loss(predictions, targets)
    print(f"   Loss: {bce_val.item():.4f}")

    # Test Focal Loss
    print("\n2. Focal Loss:")
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    focal_val = focal_loss(predictions, targets)
    print(f"   Loss: {focal_val.item():.4f}")

    # Test Combined Loss
    print("\n3. Combined Loss:")
    combined_loss = CombinedLoss(
        pos_weights=CHEXPERT_POS_WEIGHTS,
        lambda_bce=0.7,
        lambda_focal=0.3
    )
    loss_dict = combined_loss(predictions, targets)
    print(f"   Total Loss: {loss_dict['loss'].item():.4f}")
    print(f"   BCE Component: {loss_dict['bce_loss']:.4f}")
    print(f"   Focal Component: {loss_dict['focal_loss']:.4f}")

    # Test pos_weights computation
    print("\n4. Computing Positive Weights from Data:")
    train_targets = torch.randint(0, 2, (10000, num_classes)).float()
    computed_weights = compute_pos_weights(train_targets)
    print(f"   Computed weights (first 5): {computed_weights[:5].tolist()}")
    print(f"   Min weight: {computed_weights.min().item():.2f}")
    print(f"   Max weight: {computed_weights.max().item():.2f}")
