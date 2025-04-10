# src/training/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation as in equation (20) of the paper with α=2.0, γ=2.0.
    """

    def __init__(self, alpha=2.0, gamma=2.0, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Set to 2.0 as specified in the paper
        self.gamma = gamma  # Set to 2.0 as specified in the paper
        self.reduction = reduction
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        """
        Forward pass calculating the focal loss as in equation (20).
        """
        # Get log probabilities
        log_softmax = F.log_softmax(inputs, dim=1)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        # Calculate pt (probability of the true class)
        log_pt = torch.sum(log_softmax * targets_one_hot, dim=1)
        pt = torch.exp(log_pt)

        # Calculate focal weights with proper alpha and gamma
        if self.class_weights is not None:
            weight_tensor = self.class_weights.to(inputs.device)[targets]
            focal_weight = weight_tensor * (1 - pt).pow(self.gamma)
        else:
            focal_weight = self.alpha * (1 - pt).pow(self.gamma)

        # Calculate loss
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
    Multi-task loss function as in equation (19).
    """

    def __init__(self, task_weights=None, class_weights=None):
        super(MultitaskLoss, self).__init__()

        # Default weights from the paper
        self.task_weights = task_weights or {
            'multiclass': 1.0,
            'contralateral': 0.5,
            'ipsilateral': 0.5,
            'neural_activity': 1.0  # Set to 1.0 as in Table 2
        }

        # Class weights for handling imbalance
        self.class_weights = class_weights

        # Task-specific losses with parameters from the paper
        if self.class_weights is not None:
            self.focal_loss = FocalLoss(alpha=2.0, gamma=2.0, class_weights=self.class_weights)
        else:
            self.focal_loss = FocalLoss(alpha=2.0, gamma=2.0)

        self.mse_loss = nn.MSELoss()

    def forward(self, outputs, targets):
        """
        Forward pass calculating the combined loss as in equation (19).

        Supports mixup augmentation by handling mixed targets.
        """
        losses = {}
        total_loss = 0.0

        # Calculate individual task losses
        for task_name, output in outputs.items():
            if task_name in targets:
                if task_name == 'neural_activity':
                    # Ensure dimensions match
                    batch_size = output.size(0)
                    target = targets[task_name]

                    # Reshape if needed
                    if output.dim() > 2:
                        output = output.view(batch_size, -1).mean(dim=1, keepdim=True)
                    else:
                        output = output.view(batch_size, -1)

                    if target.dim() > 2:
                        target = target.view(batch_size, -1).mean(dim=1, keepdim=True)
                    else:
                        target = target.view(batch_size, -1)

                    # Apply MSE loss for neural activity
                    losses[task_name] = self.mse_loss(output, target)
                else:
                    # Check for mixup-related data
                    mixup_lambda_key = f"{task_name}_mixup_lambda"
                    mixup_index_key = f"{task_name}_mixup_index"

                    # Handle mixup if present
                    if mixup_lambda_key in targets and mixup_index_key in targets:
                        # Get mixup parameters
                        lam = targets[mixup_lambda_key]
                        index = targets[mixup_index_key]

                        # Apply regular loss with original targets
                        loss1 = self.focal_loss(output, targets[task_name])

                        # Apply regular loss with shuffled targets
                        loss2 = self.focal_loss(output, targets[task_name][index])

                        # Mix losses according to lambda
                        losses[task_name] = lam * loss1 + (1 - lam) * loss2
                    else:
                        # Apply regular focal loss for classification tasks
                        losses[task_name] = self.focal_loss(output, targets[task_name])

                # Apply task weight
                if task_name in self.task_weights:
                    total_loss += self.task_weights[task_name] * losses[task_name]

        return total_loss, losses