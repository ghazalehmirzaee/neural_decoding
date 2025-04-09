# src/training/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss as described in equation (20) of the paper.

    This is used to address class imbalance in the classification tasks.
    """

    def __init__(self, alpha=2.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass implementing equation (20).

        Args:
            inputs: Model outputs (logits) for a classification task
            targets: Target labels

        Returns:
            Focal loss value
        """
        # Get log probabilities
        log_softmax = F.log_softmax(inputs, dim=1)

        # Gather log probabilities with respect to target
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        log_pt = torch.sum(log_softmax * targets_one_hot, dim=1)
        pt = torch.exp(log_pt)

        # Compute focal loss
        focal_weight = self.alpha * (1 - pt).pow(self.gamma)
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
    Multitask loss function as described in equation (19) of the paper.

    This combines focal loss for classification tasks and MSE for neural activity prediction.
    """

    def __init__(self, task_weights=None):
        super(MultitaskLoss, self).__init__()

        # Default weights from the paper
        self.task_weights = task_weights or {
            'multiclass': 1.0,
            'contralateral': 0.5,
            'ipsilateral': 0.5,
            'neural_activity': 0.5
        }

        # Task-specific losses
        self.focal_loss = FocalLoss(alpha=2.0, gamma=2.0)
        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        """
        Forward pass implementing equation (19).

        Args:
            outputs: Dictionary of model outputs for each task
            targets: Dictionary of target labels for each task

        Returns:
            tuple: (total_loss, individual_task_losses)
        """
        losses = {}
        total_loss = 0.0

        # Calculate individual task losses
        for task_name, output in outputs.items():
            if task_name in targets:
                if task_name == 'neural_activity':
                    # Regression task
                    target = targets[task_name].view(-1, 1)
                    output = output.view(-1, 1)
                    losses[task_name] = self.mse_loss(output, target)
                else:
                    # Classification task
                    losses[task_name] = self.focal_loss(output, targets[task_name])

                # Apply task weight
                if task_name in self.task_weights:
                    total_loss += self.task_weights[task_name] * losses[task_name]

        return total_loss, losses


