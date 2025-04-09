# src/training/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Enhanced Focal Loss with better class weighting for imbalanced datasets.
    """

    def __init__(self, alpha=3.0, gamma=2.0, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = class_weights  # Added support for explicit class weights

    def forward(self, inputs, targets):
        """
        Forward pass with improved stability and class weighting.
        """
        # Get log probabilities
        log_softmax = F.log_softmax(inputs, dim=1)

        # Gather log probabilities with respect to target
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        log_pt = torch.sum(log_softmax * targets_one_hot, dim=1)
        pt = torch.exp(log_pt)

        # Compute focal loss with optional class weighting
        if self.class_weights is not None:
            # Apply class-specific weights
            weight_tensor = self.class_weights.to(inputs.device)[targets]
            focal_weight = weight_tensor * (1 - pt).pow(self.gamma)
        else:
            # Use standard focal loss weighting
            focal_weight = self.alpha * (1 - pt).pow(self.gamma)

        # Calculate loss with improved numerical stability
        loss = -focal_weight * log_pt

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultitaskLoss(nn.Module):
    """
    Enhanced multitask loss function with adaptive weighting and regularization.
    """

    def __init__(self, task_weights=None, class_weights=None):
        super(MultitaskLoss, self).__init__()

        # Default weights from the paper
        self.task_weights = task_weights or {
            'multiclass': 1.0,
            'contralateral': 0.5,
            'ipsilateral': 0.5,
            'neural_activity': 0.5
        }

        # Class weights for handling imbalance (calculated based on class frequencies)
        self.class_weights = class_weights

        # Task-specific losses with improved parameters
        if self.class_weights is not None:
            self.focal_loss = FocalLoss(alpha=3.0, gamma=2.0, class_weights=self.class_weights)
        else:
            self.focal_loss = FocalLoss(alpha=3.0, gamma=2.0)

        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        """
        Forward pass with dynamic task weighting and regularization.
        """
        losses = {}
        total_loss = 0.0

        # Calculate individual task losses
        for task_name, output in outputs.items():
            if task_name in targets:
                if task_name == 'neural_activity':
                    # Regression task with L1 regularization
                    target = targets[task_name].view(-1, 1)
                    output = output.view(-1, 1)
                    losses[task_name] = self.mse_loss(output, target)
                else:
                    # Classification task with focal loss
                    losses[task_name] = self.focal_loss(output, targets[task_name])

                # Apply task weight
                if task_name in self.task_weights:
                    total_loss += self.task_weights[task_name] * losses[task_name]

        return total_loss, losses


