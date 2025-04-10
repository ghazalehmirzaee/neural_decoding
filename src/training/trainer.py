# src/training/trainer.py

import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

try:
    import wandb
except ImportError:
    wandb = None

from src.utils.metrics import calculate_metrics
from src.training.early_stopping import EarlyStopping
from src.training.losses import FocalLoss, MultitaskLoss
import src.utils.visualization as viz


class Trainer:
    """
    Enhanced trainer for neural decoding models with improved regularization,
    metrics tracking, and visualization capabilities.
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

        # Calculate class weights for handling imbalance if train_loader is provided
        if train_loader is not None:
            class_counts = self._calculate_class_distribution(train_loader)
            if class_counts is not None:
                # Inverse frequency weighting with smoothing
                num_samples = sum(class_counts.values())
                self.class_weights = torch.FloatTensor([
                    num_samples / (len(class_counts) * max(count, 1) * len(class_counts))
                    for cls, count in sorted(class_counts.items())
                ]).to(device)
                print(f"Using class weights: {self.class_weights}")
            else:
                self.class_weights = None
        else:
            self.class_weights = None

        if model_type == 'lstm':
            # Adam optimizer for LSTM as specified in Table 1
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )

            # Scheduler for LSTM
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            # AdamW optimizer for hybrid model as in equation (21)
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.training.learning_rate,
                weight_decay=config.training.weight_decay
            )

            # One-cycle LR scheduler for hybrid model as in equation (22)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.training.learning_rate,
                epochs=config.training.epochs,
                steps_per_epoch=len(train_loader) if train_loader else 100,
                pct_start=0.3  # Warmup period is 30% of total training
            )

        # Set early stopping patience to 7 as specified in the paper
        self.early_stopping = EarlyStopping(
            patience=7,
            min_delta=1e-4
        )

        # Create output directory
        self.output_dir = os.path.join(config.paths.output_dir, model_type)
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize WandB
        self.run = None
        if wandb is not None and hasattr(config, 'wandb') and config.wandb.mode != 'disabled':
            try:
                print(f"Initializing WandB with entity={config.wandb.entity}, project={config.wandb.project}")
                self.run = wandb.init(
                    project=config.wandb.project,
                    entity=config.wandb.entity,
                    config=self._get_flattened_config(),
                    name=f"{model_type}_{wandb.util.generate_id()}",
                    group=config.wandb.group,
                    mode=config.wandb.mode
                )
                print("WandB initialized successfully!")
            except Exception as e:
                print(f"WandB initialization error: {e}")
                print("Detailed error information:")
                import traceback
                traceback.print_exc()
                print("Continuing without WandB logging...")
                self.run = None

    def _get_flattened_config(self):
        """Flatten the hierarchical config for WandB."""
        flattened = {}

        # Training config
        for key, value in self.config.training.items():
            flattened[f"training.{key}"] = value

        # Model config
        for key, value in self.config.model.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    flattened[f"model.{key}.{subkey}"] = subvalue
            else:
                flattened[f"model.{key}"] = value

        return flattened


    def _calculate_class_distribution(self, dataloader):
        """Calculate class distribution for the dataset."""
        class_counts = {}

        # Go through a subset of batches to estimate class distribution
        max_batches = min(len(dataloader), 10)  # Limit to 10 batches for efficiency
        for i, (_, targets_dict) in enumerate(dataloader):
            if i >= max_batches:
                break

            if 'multiclass' in targets_dict:
                labels = targets_dict['multiclass'].cpu().numpy()
                unique_labels, counts = np.unique(labels, return_counts=True)

                for label, count in zip(unique_labels, counts):
                    if label not in class_counts:
                        class_counts[label] = 0
                    class_counts[label] += count

        return class_counts if class_counts else None

    def train(self):
        """
        Train the model following the procedure described in the paper.

        Returns:
            tuple: (test_predictions, test_targets)
        """
        print(f"\nStarting training for {self.model_type} model...")
        print(f"Using device: {self.device}")

        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Track best model and training history
        best_val_loss = float('inf')
        best_epoch = 0
        train_history = defaultdict(list)
        val_history = defaultdict(list)

        # Training start time
        start_time = time.time()

        # Main training loop
        for epoch in range(self.config.training.epochs):
            epoch_start_time = time.time()

            # Train
            train_loss, train_metrics = self._train_epoch(epoch)

            # Validate
            val_loss, val_metrics = self._validate_epoch(epoch)

            # Update learning rate for LSTM model
            if self.model_type == 'lstm':
                self.scheduler.step(val_loss)

            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time

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
                  f"Time: {epoch_duration:.1f}s - "
                  f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
                  f"Learning rate: {lr:.6f}")

            # Print task metrics
            for task in ['multiclass', 'contralateral', 'ipsilateral']:
                print(f"  {task.capitalize()}: "
                      f"Train acc: {train_metrics[task]['accuracy']:.4f}, "
                      f"Val acc: {val_metrics[task]['accuracy']:.4f}, "
                      f"Val F1: {val_metrics[task]['f1']:.4f}")

            # Log metrics to WandB
            if self.run is not None:
                log_dict = {
                    'epoch': epoch + 1,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'learning_rate': lr
                }

                # Add task-specific metrics
                for task in ['multiclass', 'contralateral', 'ipsilateral']:
                    for metric, value in train_metrics[task].items():
                        log_dict[f'train/{task}_{metric}'] = value
                    for metric, value in val_metrics[task].items():
                        log_dict[f'val/{task}_{metric}'] = value

                wandb.log(log_dict)

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

        # Training complete
        total_duration = time.time() - start_time
        print(f"\nTraining completed in {total_duration:.1f}s. Best validation loss: {best_val_loss:.4f}")

        # Save training history
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            history = {
                'train': {k: [float(v) for v in vals] for k, vals in train_history.items()},
                'val': {k: [float(v) for v in vals] for k, vals in val_history.items()}
            }
            json.dump(history, f, indent=4)

        # Log final history plot to WandB
        if self.run is not None:
            self._log_history_plots(train_history, val_history)

        # Load best model for evaluation
        self._load_checkpoint('best_model.pth')

        # Evaluate on test set
        test_loss, test_metrics, test_predictions, test_targets = self.evaluate()

        # Generate and save visualizations
        viz.generate_all_visualizations(
            predictions=test_predictions,
            targets=test_targets,
            model_type=self.model_type,
            output_dir=self.config.paths.output_dir
        )

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

        # Close WandB run
        if self.run is not None:
            wandb.finish()

        return test_predictions, test_targets

    def _train_epoch(self, epoch):
        """
        Train for one epoch with improved metrics tracking and regularization.

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

        # Progress bar for better visualization
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} (Train)")

        for batch_idx, batch in enumerate(pbar):
            # Get inputs and targets
            inputs, targets_dict = batch
            inputs = inputs.to(self.device)
            targets_dict = {k: v.to(self.device) for k, v in targets_dict.items()}

            # Apply mixup data augmentation if enabled
            if hasattr(self.config.training, 'use_mixup') and self.config.training.use_mixup:
                inputs, targets_dict = self._mixup_batch(inputs, targets_dict)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs_dict = self.model(inputs)

            # Calculate loss
            loss, individual_losses = self.criterion(outputs_dict, targets_dict)

            # Store individual task losses
            for task, task_loss in individual_losses.items():
                task_losses[task] += task_loss.item()

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
                if task in outputs_dict and task in targets_dict:
                    all_outputs[task].append(outputs_dict[task].detach().cpu())
                    all_targets[task].append(targets_dict[task].cpu())

            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f"{avg_loss:.4f}"})

        # Calculate overall average loss
        avg_loss = total_loss / len(self.train_loader)

        # Calculate task-specific average losses
        task_avg_losses = {task: loss / len(self.train_loader) for task, loss in task_losses.items()}

        # Process stored outputs and targets for metrics
        metrics = {}
        for task in ['multiclass', 'contralateral', 'ipsilateral']:
            if all_outputs[task] and all_targets[task]:
                # Concatenate all outputs and targets
                outputs = torch.cat(all_outputs[task])
                targets = torch.cat(all_targets[task])

                # Get predictions
                _, preds = torch.max(outputs, dim=1)

                # Calculate metrics
                metrics[task] = calculate_metrics(targets.numpy(), preds.numpy())

        return avg_loss, metrics

    def _validate_epoch(self, epoch):
        """
        Validate for one epoch with improved metrics tracking.

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

        # No gradient tracking needed for validation
        with torch.no_grad():
            # Progress bar for better visualization
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} (Val)")

            for batch_idx, batch in enumerate(pbar):
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
                    task_losses[task] += task_loss.item()

                # Store outputs and targets for metrics
                for task in ['multiclass', 'contralateral', 'ipsilateral']:
                    if task in outputs_dict and task in targets_dict:
                        all_outputs[task].append(outputs_dict[task].detach().cpu())
                        all_targets[task].append(targets_dict[task].cpu())

                # Update progress bar
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f"{avg_loss:.4f}"})

        # Calculate overall average loss
        avg_loss = total_loss / len(self.val_loader)

        # Calculate task-specific average losses
        task_avg_losses = {task: loss / len(self.val_loader) for task, loss in task_losses.items()}

        # Process stored outputs and targets for metrics
        metrics = {}
        for task in ['multiclass', 'contralateral', 'ipsilateral']:
            if all_outputs[task] and all_targets[task]:
                # Concatenate all outputs and targets
                outputs = torch.cat(all_outputs[task])
                targets = torch.cat(all_targets[task])

                # Get predictions
                _, preds = torch.max(outputs, dim=1)

                # Calculate metrics
                metrics[task] = calculate_metrics(targets.numpy(), preds.numpy())

                # Add confusion matrix logging to wandb
                if self.run is not None:
                    try:
                        wandb.log({
                            f'val/{task}_confusion_matrix': wandb.plot.confusion_matrix(
                                probs=None,
                                y_true=targets.numpy(),
                                preds=preds.numpy(),
                                class_names=[f'Class {i}' for i in range(outputs.shape[1])]
                            )
                        })
                    except:
                        # Continue even if wandb logging fails
                        pass

        return avg_loss, metrics

    def evaluate(self):
        """
        Evaluate the model on the test set and collect predictions for visualization.
        """
        print("\nEvaluating model on test set...")
        self.model.eval()
        total_loss = 0

        # Initialize containers for all predictions and targets
        all_predictions = {
            'multiclass': [],
            'contralateral': [],
            'ipsilateral': [],
            'multiclass_probs': [],
            'contralateral_probs': [],
            'ipsilateral_probs': []
        }

        # Add neural activity prediction for hybrid model - FIXED
        all_targets = {
            'multiclass': [],
            'contralateral': [],
            'ipsilateral': []
        }

        # Sample a batch to determine available tasks instead of accessing dataset directly
        try:
            sample_inputs, sample_targets = next(iter(self.test_loader))
            if self.model_type == 'hybrid' and 'neural_activity' in sample_targets:
                all_predictions['neural_activity'] = []
                all_targets['neural_activity'] = []
        except StopIteration:
            # Handle empty test loader
            print("Warning: Test loader is empty")

        # No gradient tracking needed for evaluation
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")

            for batch_idx, batch in enumerate(pbar):
                # Get inputs and targets
                inputs, targets_dict = batch
                inputs = inputs.to(self.device)
                targets_dict = {k: v.to(self.device) for k, v in targets_dict.items()}

                # Forward pass
                outputs_dict = self.model(inputs)

                # Calculate loss
                loss, _ = self.criterion(outputs_dict, targets_dict)
                total_loss += loss.item()

                # Store predictions and targets for each task
                for task in ['multiclass', 'contralateral', 'ipsilateral']:
                    if task in outputs_dict and task in targets_dict:
                        # Get predictions
                        probabilities = F.softmax(outputs_dict[task], dim=1)
                        _, preds = torch.max(outputs_dict[task], dim=1)

                        # Store predictions, probabilities, and targets
                        all_predictions[task].append(preds.cpu().numpy())
                        all_predictions[f'{task}_probs'].append(probabilities.cpu().numpy())
                        all_targets[task].append(targets_dict[task].cpu().numpy())

                # Store neural activity predictions if available
                if 'neural_activity' in outputs_dict and 'neural_activity' in targets_dict:
                    all_predictions['neural_activity'].append(outputs_dict['neural_activity'].cpu().numpy())
                    all_targets['neural_activity'].append(targets_dict['neural_activity'].cpu().numpy())

        # Calculate average loss
        avg_loss = total_loss / len(self.test_loader)

        # Concatenate all predictions and targets
        for key in all_predictions:
            if all_predictions[key]:
                all_predictions[key] = np.concatenate(all_predictions[key])

        for key in all_targets:
            if all_targets[key]:
                all_targets[key] = np.concatenate(all_targets[key])

        # Calculate metrics for each task
        test_metrics = {}
        for task in ['multiclass', 'contralateral', 'ipsilateral']:
            if task in all_predictions and task in all_targets:
                test_metrics[task] = calculate_metrics(
                    all_targets[task], all_predictions[task]
                )

                # Print task-specific metrics
                print(f"\n{task.capitalize()} Metrics:")
                for metric, value in test_metrics[task].items():
                    print(f"  {metric}: {value:.4f}")

        # Add overall accuracy
        overall_acc = np.mean([metrics['accuracy'] for task, metrics in test_metrics.items()])
        test_metrics['overall'] = {'accuracy': overall_acc}
        print(f"\nOverall Accuracy: {overall_acc:.4f}")

        return avg_loss, test_metrics, all_predictions, all_targets

    def _save_checkpoint(self, state, filename):
        """Save model checkpoint."""
        torch.save(state, os.path.join(self.output_dir, filename))

    def _load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(os.path.join(self.output_dir, filename), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint (epoch {checkpoint['epoch'] + 1})")

    def _log_history_plots(self, train_history, val_history):
        """Generate and log training history plots to WandB."""
        try:
            # Create subplots for loss and accuracy
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))

            # Plot training and validation loss
            axes[0].plot(train_history['loss'], label='Train Loss')
            axes[0].plot(val_history['loss'], label='Validation Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Training and Validation Loss')
            axes[0].legend()
            axes[0].grid(True)

            # Plot training and validation accuracy for multiclass task
            if 'multiclass_accuracy' in train_history and 'multiclass_accuracy' in val_history:
                axes[1].plot(train_history['multiclass_accuracy'], label='Train Accuracy')
                axes[1].plot(val_history['multiclass_accuracy'], label='Validation Accuracy')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Accuracy')
                axes[1].set_title('Training and Validation Accuracy (Multiclass)')
                axes[1].legend()
                axes[1].grid(True)

            plt.tight_layout()

            # Log to WandB
            wandb.log({"training_history": wandb.Image(fig)})
            plt.close(fig)
        except Exception as e:
            print(f"Error generating history plots: {e}")

    def _mixup_batch(self, inputs, targets_dict):
        """
        Apply mixup data augmentation to batch.

        Mixup creates convex combinations of pairs of examples and their labels,
        which helps with regularization and generalization.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, input_size)
            targets_dict: Dictionary of target tensors

        Returns:
            tuple: (mixed_inputs, mixed_targets_dict)
        """
        batch_size = inputs.size(0)

        # Sample lambda from beta distribution
        alpha = 0.2  # Controls how strong the mixup is (smaller = less mixing)
        lam = np.random.beta(alpha, alpha, batch_size)
        lam = torch.tensor(lam, device=self.device).float().view(-1, 1, 1)

        # Create shuffled indices
        index = torch.randperm(batch_size, device=self.device)

        # Mix inputs
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

        # Mix targets
        mixed_targets_dict = {}
        for task, targets in targets_dict.items():
            if task == 'neural_activity':
                # For regression targets, apply the same lambda
                lam_flat = lam.view(-1, 1)
                mixed_targets_dict[task] = lam_flat * targets + (1 - lam_flat) * targets[index]
            else:
                # For classification, we'll keep the original targets but track lambda
                # This will be handled by the loss function
                mixed_targets_dict[task] = targets
                mixed_targets_dict[f"{task}_mixup_index"] = index
                mixed_targets_dict[f"{task}_mixup_lambda"] = lam.squeeze()

        return mixed_inputs, mixed_targets_dict

