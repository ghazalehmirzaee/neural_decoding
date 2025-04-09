# src/training/trainer.py

import os
import json
import torch
import numpy as np
import wandb
from tqdm import tqdm
from collections import defaultdict

from src.utils.metrics import calculate_metrics
from src.training.early_stopping import EarlyStopping
from src.training.losses import FocalLoss, MultitaskLoss


class Trainer:
    """
    Trainer for neural decoding models.

    This implements the training process described in the paper for both the LSTM
    and hybrid CNN-BiLSTM models, including learning rate scheduling, early stopping,
    and gradient clipping.
    """

    def __init__(
            self,
            model,
            model_type,
            train_loader,
            val_loader,
            test_loader,
            config,
            device
    ):
        self.model = model
        self.model_type = model_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device

        # Move model to device
        self.model.to(device)

        # Set up optimizer
        if model_type == 'lstm':
            # Adam optimizer for LSTM as specified in Table 1
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=1e-5  # L2 regularization
            )
        else:
            # AdamW optimizer for hybrid model as specified in equation (21)
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )

        # Set up loss function
        self.criterion = MultitaskLoss()

        # Set up learning rate scheduler
        if model_type == 'lstm':
            # ReduceLROnPlateau scheduler for LSTM as in Table 1
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            # One-cycle LR scheduler for hybrid model as in equation (22)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.training.learning_rate,
                epochs=config.training.epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3
            )

        # Set up early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=1e-4
        )

        # Create output directory
        self.output_dir = os.path.join(config.paths.output_dir, model_type)
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup W&B
        self.run = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=self._get_flattened_config(),
            name=f"{model_type}_{wandb.util.generate_id()}",
            group=config.wandb.group
        )

    def _get_flattened_config(self):
        """Flatten nested config for W&B."""
        flat_config = {}

        def _flatten(config, prefix=''):
            for key, value in config.items():
                if isinstance(value, dict):
                    _flatten(value, f"{prefix}{key}.")
                else:
                    flat_config[f"{prefix}{key}"] = value

        _flatten(self.config)
        return flat_config

    def train(self):
        """
        Train the model following the procedure described in the paper.

        Returns:
            tuple: (test_predictions, test_targets)
        """
        print(f"\nStarting training for {self.model_type} model...")

        best_val_loss = float('inf')
        best_epoch = 0
        train_history = defaultdict(list)
        val_history = defaultdict(list)

        for epoch in range(self.config.training.epochs):
            # Train
            train_loss, train_metrics = self._train_epoch(epoch)

            # Validate
            val_loss, val_metrics = self._validate_epoch(epoch)

            # Update learning rate for LSTM model
            if self.model_type == 'lstm':
                self.scheduler.step(val_loss)

            # Save metrics
            train_history['loss'].append(train_loss)
            for task, task_metrics in train_metrics.items():
                for metric, value in task_metrics.items():
                    train_history[f"{task}_{metric}"].append(value)

            val_history['loss'].append(val_loss)
            for task, task_metrics in val_metrics.items():
                for metric, value in task_metrics.items():
                    val_history[f"{task}_{metric}"].append(value)

            # Print progress
            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{self.config.training.epochs} - "
                  f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                  f"Learning rate: {lr:.6f}")

            # Print task metrics
            for task in ['multiclass', 'contralateral', 'ipsilateral']:
                print(f"  {task.capitalize()}: "
                      f"Train acc: {train_metrics[task]['accuracy']:.4f}, "
                      f"Val acc: {val_metrics[task]['accuracy']:.4f}, "
                      f"Val F1: {val_metrics[task]['f1']:.4f}")

            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch

                # Save checkpoint
                self._save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, 'best_model.pth')

                print(f"New best model saved (val_loss: {val_loss:.4f})")

            # Check for early stopping
            if self.early_stopping(val_loss):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best model was at epoch {best_epoch + 1}")
                break

        print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")

        # Save training history
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            history = {
                'train': {k: [float(v) for v in vals] for k, vals in train_history.items()},
                'val': {k: [float(v) for v in vals] for k, vals in val_history.items()}
            }
            json.dump(history, f, indent=4)

        # Load best model for evaluation
        self._load_checkpoint('best_model.pth')

        # Evaluate on test set
        test_loss, test_metrics, test_predictions, test_targets = self.evaluate()

        # Save test results
        results = {
            'test_loss': float(test_loss),
            'test_metrics': {
                task: {k: float(v) for k, v in metrics.items()}
                for task, metrics in test_metrics.items()
            }
        }

        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Test results saved to {os.path.join(self.output_dir, 'test_results.json')}")

        return test_predictions, test_targets

    def _train_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            tuple: (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0
        task_losses = defaultdict(float)

        # Initialize containers for outputs and targets per task
        all_outputs = {
            'multiclass': [],
            'contralateral': [],
            'ipsilateral': []
        }
        all_targets = {
            'multiclass': [],
            'contralateral': [],
            'ipsilateral': []
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} (Train)")

        for batch_idx, batch in enumerate(pbar):
            # Get inputs and targets
            inputs, targets_dict = batch
            inputs = inputs.to(self.device)
            targets_dict = {k: v.to(self.device) for k, v in targets_dict.items()}

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs_dict = self.model(inputs)

            # Calculate loss
            loss, individual_losses = self.criterion(outputs_dict, targets_dict)

            # Store individual task losses
            for task, task_loss in individual_losses.items():
                task_losses[task] += task_loss

            # Backward pass and optimization
            loss.backward()

            # Clip gradients as mentioned in the paper
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.gradient_clip_val
            )

            self.optimizer.step()

            # Update scheduler for hybrid model (applied per step)
            if self.model_type == 'hybrid':
                self.scheduler.step()

            # Store outputs and targets for metrics
            for task in ['multiclass', 'contralateral', 'ipsilateral']:
                all_outputs[task].append(outputs_dict[task].detach().cpu())
                all_targets[task].append(targets_dict[task].cpu())

            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f"{avg_loss:.4f}"})

        # Calculate average loss
        avg_loss = total_loss / len(self.train_loader)
        avg_task_losses = {task: loss / len(self.train_loader)
                           for task, loss in task_losses.items()}

        # Calculate metrics for each classification task
        metrics = {}
        for task in ['multiclass', 'contralateral', 'ipsilateral']:
            # Process outputs and targets
            outputs = torch.cat(all_outputs[task])
            targets = torch.cat(all_targets[task])

            # Get predictions
            _, preds = torch.max(outputs, 1)

            # Calculate metrics
            metrics[task] = calculate_metrics(
                targets.numpy(),
                preds.numpy(),
                task
            )

        # Log to W&B
        log_dict = {
            'train/loss': avg_loss,
            'epoch': epoch + 1
        }

        for task, task_metrics in metrics.items():
            for metric, value in task_metrics.items():
                log_dict[f'train/{task}_{metric}'] = value
            log_dict[f'train/{task}_loss'] = avg_task_losses.get(task, 0)

        wandb.log(log_dict)

        return avg_loss, metrics

    def _validate_epoch(self, epoch):
        """
        Validate for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            tuple: (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0
        task_losses = defaultdict(float)

        # Initialize containers for outputs and targets per task
        all_outputs = {
            'multiclass': [],
            'contralateral': [],
            'ipsilateral': []
        }
        all_targets = {
            'multiclass': [],
            'contralateral': [],
            'ipsilateral': []
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch + 1} (Val)"):
                # Get inputs and targets
                inputs, targets_dict = batch
                inputs = inputs.to(self.device)
                targets_dict = {k: v.to(self.device) for k, v in targets_dict.items()}

                # Forward pass
                outputs_dict = self.model(inputs)

                # Calculate loss
                loss, individual_losses = self.criterion(outputs_dict, targets_dict)

                # Store individual task losses
                for task, task_loss in individual_losses.items():
                    task_losses[task] += task_loss

                # Store outputs and targets for metrics
                for task in ['multiclass', 'contralateral', 'ipsilateral']:
                    all_outputs[task].append(outputs_dict[task].detach().cpu())
                    all_targets[task].append(targets_dict[task].cpu())

                # Update total loss
                total_loss += loss.item()

        # Calculate average loss
        avg_loss = total_loss / len(self.val_loader)
        avg_task_losses = {task: loss / len(self.val_loader)
                           for task, loss in task_losses.items()}

        # Calculate metrics for each classification task
        metrics = {}
        for task in ['multiclass', 'contralateral', 'ipsilateral']:
            # Process outputs and targets
            outputs = torch.cat(all_outputs[task])
            targets = torch.cat(all_targets[task])

            # Get predictions
            _, preds = torch.max(outputs, 1)

            # Calculate metrics
            metrics[task] = calculate_metrics(
                targets.numpy(),
                preds.numpy(),
                task
            )

        # Log to W&B
        log_dict = {
            'val/loss': avg_loss,
            'epoch': epoch + 1
        }

        for task, task_metrics in metrics.items():
            for metric, value in task_metrics.items():
                log_dict[f'val/{task}_{metric}'] = value
            log_dict[f'val/{task}_loss'] = avg_task_losses.get(task, 0)

        wandb.log(log_dict)

        return avg_loss, metrics

    def evaluate(self):
        """
        Evaluate model on test set.

        Returns:
            tuple: (avg_loss, metrics_dict, predictions_dict, targets_dict)
        """
        print("\nEvaluating on test set...")

        self.model.eval()
        total_loss = 0
        task_losses = defaultdict(float)

        # Initialize containers for outputs and targets per task
        all_outputs = {
            'multiclass': [],
            'contralateral': [],
            'ipsilateral': []
        }
        all_targets = {
            'multiclass': [],
            'contralateral': [],
            'ipsilateral': []
        }

        # For hybrid model, also track neural activity
        if self.model_type == 'hybrid':
            all_outputs['neural_activity'] = []
            all_targets['neural_activity'] = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Get inputs and targets
                inputs, targets_dict = batch
                inputs = inputs.to(self.device)
                targets_dict = {k: v.to(self.device) for k, v in targets_dict.items()}

                # Forward pass
                outputs_dict = self.model(inputs)

                # Calculate loss
                loss, individual_losses = self.criterion(outputs_dict, targets_dict)

                # Store individual task losses
                for task, task_loss in individual_losses.items():
                    task_losses[task] += task_loss

                # Store outputs and targets for all tasks
                for task in outputs_dict.keys():
                    if task in all_outputs:
                        all_outputs[task].append(outputs_dict[task].detach().cpu())
                        if task in targets_dict:
                            all_targets[task].append(targets_dict[task].cpu())

                # Update total loss
                total_loss += loss.item()

        # Calculate average loss
        avg_loss = total_loss / len(self.test_loader)

        # Process outputs and targets
        processed_outputs = {}
        processed_targets = {}

        for task in all_outputs.keys():
            if task == 'neural_activity':
                # Regression task
                processed_outputs[task] = torch.cat(all_outputs[task]).numpy()
                processed_targets[task] = torch.cat(all_targets[task]).numpy()
            else:
                # Classification task
                outputs = torch.cat(all_outputs[task])
                targets = torch.cat(all_targets[task])

                # Store raw probabilities for ROC curves
                processed_outputs[f"{task}_probs"] = F.softmax(outputs, dim=1).numpy()

                # Get predictions
                _, preds = torch.max(outputs, 1)
                processed_outputs[task] = preds.numpy()
                processed_targets[task] = targets.numpy()

        # Calculate metrics for classification tasks
        metrics = {}
        for task in ['multiclass', 'contralateral', 'ipsilateral']:
            metrics[task] = calculate_metrics(
                processed_targets[task],
                processed_outputs[task],
                task
            )

        # Print results
        print(f"\nTest Results:")
        print(f"Loss: {avg_loss:.4f}")

        for task in ['multiclass', 'contralateral', 'ipsilateral']:
            print(f"\n{task.capitalize()}:")
            print(f"  Accuracy: {metrics[task]['accuracy']:.4f}")
            print(f"  Precision: {metrics[task]['precision']:.4f}")
            print(f"  Recall: {metrics[task]['recall']:.4f}")
            print(f"  F1 Score: {metrics[task]['f1']:.4f}")

        # Log to W&B
        log_dict = {'test/loss': avg_loss}

        for task, task_metrics in metrics.items():
            for metric, value in task_metrics.items():
                log_dict[f'test/{task}_{metric}'] = value

        wandb.log(log_dict)

        return avg_loss, metrics, processed_outputs, processed_targets

    def _save_checkpoint(self, state, filename):
        """Save model checkpoint."""
        torch.save(state, os.path.join(self.output_dir, filename))

    def _load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(os.path.join(self.output_dir, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

