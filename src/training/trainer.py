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
import wandb
from src.utils.metrics import calculate_metrics
from src.training.early_stopping import EarlyStopping
from src.training.losses import FocalLoss, MultitaskLoss
import src.utils.visualization as viz


class Trainer:
    """
    trainer for comprehensive neuronal decoding ablation study.

    Supported Models:
    - CNN: Spatial-only processing without temporal dynamics
    - LSTM: Basic temporal processing with recurrent connections
    - LSTM+Attention: Enhanced temporal processing with attention mechanisms
    - Hybrid: Combined spatial-temporal processing with advanced features
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
        self.best_model_path = os.path.join(
            config.paths.checkpoints_dir,
            model_type,
            'best_model.pth'
        )

        # Move model to device
        self.model.to(device)

        # Calculate class weights for handling imbalance if train_loader is provided
        if train_loader is not None:
            self.class_weights = self._calculate_class_weights(train_loader)
        else:
            self.class_weights = None

        # Initialize loss function with task weights from model config
        # This ensures fair comparison across different model architectures
        task_weights = getattr(config.model, 'task_weights', None)
        self.criterion = MultitaskLoss(
            task_weights=task_weights,
            class_weights=self.class_weights
        )

        # Initialize optimizer and scheduler based on model type
        # Different architectures benefit from different optimization strategies
        self._initialize_optimizer_and_scheduler()

        # Set early stopping with model-appropriate patience
        patience = getattr(config.training, 'early_stopping_patience', 7)
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=1e-4,
            mode='min',
            verbose=True
        )

        # Create output directories
        self.output_dir = os.path.join(config.paths.output_dir, model_type)
        os.makedirs(self.output_dir, exist_ok=True)

        checkpoint_dir = os.path.join(config.paths.checkpoints_dir, model_type)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize WandB if available and configured
        self._initialize_wandb()

    def _initialize_optimizer_and_scheduler(self):
        """
        Initialize optimizer and learning rate scheduler based on model type.

        Each model architecture has different optimization requirements:
        - CNN: Simple Adam with step decay
        - LSTM: Adam with plateau-based scheduling
        - LSTM+Attention: Adam with cosine annealing and warmup
        - Hybrid: AdamW with one-cycle scheduling
        """
        if self.model_type == 'cnn':
            # Simple optimization for CNN-only model
            # Using basic Adam with step decay for intentionally limited performance
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )

            # Step scheduler for CNN - simple and effective
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=getattr(self.config.training, 'step_size', 40),
                gamma=getattr(self.config.training, 'gamma', 0.8),
                verbose=True
            )
            self.warmup_epochs = 0  # No warmup for CNN

        elif self.model_type == 'lstm':
            # Adam optimizer for LSTM as specified in Table 1
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )

            # Scheduler for LSTM - ReduceLROnPlateau as in Table 1
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,  # Reduce by half when plateauing
                patience=5,  # Wait 5 epochs before reducing
                verbose=True
            )
            self.warmup_epochs = 0  # No warmup for basic LSTM

        elif self.model_type == 'lstm_attention':
            # Adam optimizer for LSTM+attention model with improved parameters
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=(0.9, 0.999),  # Standard Adam betas
                eps=1e-8
            )

            # Learning rate scheduling for LSTM+attention
            if hasattr(self.config.training, 'use_cosine_scheduler') and self.config.training.use_cosine_scheduler:
                # Cosine annealing for smooth learning rate decay
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.training.epochs,
                    eta_min=getattr(self.config.training, 'eta_min', 1e-6),
                    verbose=True
                )

                # Add warmup if specified
                if hasattr(self.config.training, 'warmup_epochs') and self.config.training.warmup_epochs > 0:
                    self.warmup_epochs = self.config.training.warmup_epochs
                    self.warmup_factor = getattr(self.config.training, 'warmup_factor', 0.1)
                    self.base_lr = self.config.training.learning_rate
                else:
                    self.warmup_epochs = 0
            else:
                # Fallback to step scheduler
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=getattr(self.config.training, 'step_size', 40),
                    gamma=getattr(self.config.training, 'gamma', 0.7),
                    verbose=True
                )
                self.warmup_epochs = 0

        else:  # hybrid
            # AdamW optimizer for hybrid model as in equation (21)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay
            )

            # One-cycle LR scheduler for hybrid model as in equation (22)
            steps_per_epoch = len(self.train_loader) if self.train_loader else 100
            total_steps = steps_per_epoch * self.config.training.epochs

            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.training.learning_rate,
                total_steps=total_steps,
                pct_start=0.3,  # Warmup period is 30% of total training as mentioned
                div_factor=25,  # Initial learning rate is max_lr/25
                final_div_factor=10000,  # Final learning rate is max_lr/10000
                anneal_strategy='cos'  # Cosine annealing as per the paper
            )
            self.warmup_epochs = 0  # OneCycleLR handles its own warmup

    def _initialize_wandb(self):
        """Initialize Weights & Biases logging if available and configured."""
        self.run = None
        if wandb is not None and hasattr(self.config, 'wandb') and self.config.wandb.mode != 'disabled':
            try:
                print(f"Initializing WandB with entity={self.config.wandb.entity}, project={self.config.wandb.project}")
                self.run = wandb.init(
                    project=self.config.wandb.project,
                    entity=self.config.wandb.entity,
                    config=self._get_flattened_config(),
                    name=f"{self.model_type}_{time.strftime('%Y%m%d_%H%M%S')}",
                    group=self.config.wandb.group,
                    mode=self.config.wandb.mode
                )
                print("WandB initialized successfully!")
            except Exception as e:
                print(f"WandB initialization error: {e}")
                print("Continuing without WandB logging...")
                self.run = None

    def _get_flattened_config(self):
        """Flatten the hierarchical config for WandB logging."""
        flattened = {}

        # Training config
        if hasattr(self.config, 'training'):
            for key, value in self.config.training.items():
                flattened[f"training.{key}"] = value

        # Model config
        if hasattr(self.config, 'model'):
            for key, value in self.config.model.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flattened[f"model.{key}.{subkey}"] = subvalue
                else:
                    flattened[f"model.{key}"] = value

        return flattened

    def _calculate_class_weights(self, dataloader):
        """
        Calculate inverse frequency class weights to handle class imbalance.
        This ensures fair comparison across all model types.
        """
        try:
            # Count classes across a subset of batches for efficiency
            class_counts = defaultdict(int)
            max_batches = min(len(dataloader), 20)  # Use at most 20 batches

            print("Calculating class weights from training data...")
            for i, (_, targets_dict) in enumerate(dataloader):
                if i >= max_batches:
                    break

                if 'multiclass' in targets_dict:
                    labels = targets_dict['multiclass'].cpu().numpy()
                    unique_labels, counts = np.unique(labels, return_counts=True)

                    for label, count in zip(unique_labels, counts):
                        class_counts[label] += count

            if not class_counts:
                return None

            # Convert to class weights using inverse frequency
            total_samples = sum(class_counts.values())
            num_classes = len(class_counts)

            # Create weights with smoothing to avoid extreme values
            weights = torch.FloatTensor([
                total_samples / (num_classes * max(count, 1) * num_classes)
                for cls, count in sorted(class_counts.items())
            ]).to(self.device)

            print(f"Class distribution: {dict(sorted(class_counts.items()))}")
            print(f"Class weights: {weights}")

            return weights

        except Exception as e:
            print(f"Error calculating class weights: {e}")
            return None

    def train(self):
        """
        Train the model following architecture-specific procedures.
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

            # Apply warmup if configured (for LSTM+attention)
            if hasattr(self, 'warmup_epochs') and self.warmup_epochs > 0 and epoch < self.warmup_epochs:
                lr_scale = self.warmup_factor + (1.0 - self.warmup_factor) * epoch / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.base_lr * lr_scale

            # Train one epoch
            train_loss, train_metrics = self._train_epoch(epoch)

            # Validate one epoch
            val_loss, val_metrics = self._validate_epoch(epoch)

            # Update learning rate scheduler based on model type
            self._update_scheduler(val_loss, epoch)

            # Calculate epoch duration
            epoch_duration = time.time() - epoch_start_time

            # Save metrics to history
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
                if task in train_metrics and task in val_metrics:
                    print(f"  {task.capitalize()}: "
                          f"Train acc: {train_metrics[task]['accuracy']:.4f}, "
                          f"Val acc: {val_metrics[task]['accuracy']:.4f}, "
                          f"Val F1: {val_metrics[task]['f1']:.4f}")

            # Log metrics to WandB
            self._log_to_wandb(epoch, train_loss, val_loss, train_metrics, val_metrics, lr)

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
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"Best model was at epoch {best_epoch + 1}")
                break

        # Training complete
        total_duration = time.time() - start_time
        print(f"\nTraining completed in {total_duration:.1f}s. Best validation loss: {best_val_loss:.4f}")

        # Save training history
        self._save_training_history(train_history, val_history)

        # Log final history plot to WandB
        if self.run is not None:
            self._log_history_plots(train_history, val_history)

        # Load best model for evaluation
        self._load_checkpoint('best_model.pth')

        # Evaluate on test set
        test_loss, test_metrics, test_predictions, test_targets = self.evaluate()

        # Generate and save visualizations
        self._generate_and_save_visualizations(test_predictions, test_targets)

        # Save test results
        self._save_test_results(test_loss, test_metrics)

        # Close WandB run
        if self.run is not None:
            wandb.finish()

        return test_predictions, test_targets

    def _update_scheduler(self, val_loss, epoch):
        """Update learning rate scheduler based on model type."""
        if self.model_type == 'lstm':
            self.scheduler.step(val_loss)
        elif self.model_type == 'hybrid':
            self.scheduler.step()  # OneCycleLR steps automatically
        elif self.model_type in ['cnn', 'lstm_attention']:
            # Skip warmup epochs for cosine annealing
            if not (hasattr(self, 'warmup_epochs') and epoch < self.warmup_epochs):
                self.scheduler.step()

    def _train_epoch(self, epoch):
        """Train for one epoch with model-specific handling."""
        self.model.train()
        total_loss = 0
        task_losses = defaultdict(float)

        # Initialize containers for predictions and targets per task
        all_outputs = {'multiclass': [], 'contralateral': [], 'ipsilateral': []}
        all_targets = {'multiclass': [], 'contralateral': [], 'ipsilateral': []}

        # Progress bar for better visualization
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} (Train)")

        for batch_idx, batch in enumerate(pbar):
            # Get inputs and targets
            inputs, targets_dict = batch
            inputs = inputs.to(self.device)
            targets_dict = {k: v.to(self.device) for k, v in targets_dict.items()}

            # Apply mixup data augmentation if enabled (only for more advanced models)
            if (hasattr(self.config.training, 'use_mixup') and
                    self.config.training.use_mixup and
                    self.model_type in ['hybrid']):  # Only hybrid uses mixup
                inputs, targets_dict = self._mixup_batch(inputs, targets_dict)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs_dict = self.model(inputs)

            # Calculate loss with task weighting
            loss, individual_losses = self.criterion(outputs_dict, targets_dict)

            # Store individual task losses
            for task, task_loss in individual_losses.items():
                task_losses[task] += task_loss.item()

            # Backward pass
            loss.backward()

            # Clip gradients to prevent explosions
            if hasattr(self.config.training, 'gradient_clip_val'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_val
                )

            # Optimization step
            self.optimizer.step()

            # Store outputs and targets for metrics calculation
            for task in ['multiclass', 'contralateral', 'ipsilateral']:
                if task in outputs_dict and task in targets_dict:
                    # Get predictions
                    _, preds = torch.max(outputs_dict[task], dim=1)

                    # Store for metrics
                    all_outputs[task].append(preds.detach().cpu())
                    all_targets[task].append(targets_dict[task].cpu())

            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f"{avg_loss:.4f}"})

        # Calculate overall average loss
        avg_loss = total_loss / len(self.train_loader)

        # Calculate task-specific metrics
        metrics = {}
        for task in ['multiclass', 'contralateral', 'ipsilateral']:
            if all_outputs[task] and all_targets[task]:
                # Concatenate all predictions and targets
                preds = torch.cat(all_outputs[task]).numpy()
                targets = torch.cat(all_targets[task]).numpy()

                # Calculate metrics
                metrics[task] = calculate_metrics(targets, preds, task)

        return avg_loss, metrics

    def _validate_epoch(self, epoch):
        """Validate for one epoch with improved metrics tracking."""
        self.model.eval()
        total_loss = 0
        task_losses = defaultdict(float)

        # Initialize containers for predictions and targets per task
        all_outputs = {'multiclass': [], 'contralateral': [], 'ipsilateral': []}
        all_targets = {'multiclass': [], 'contralateral': [], 'ipsilateral': []}

        # No gradient tracking for validation
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

                # Calculate loss with task weighting
                loss, individual_losses = self.criterion(outputs_dict, targets_dict)

                # Store individual task losses
                for task, task_loss in individual_losses.items():
                    task_losses[task] += task_loss.item()

                # Store outputs and targets for metrics calculation
                for task in ['multiclass', 'contralateral', 'ipsilateral']:
                    if task in outputs_dict and task in targets_dict:
                        # Get predictions
                        _, preds = torch.max(outputs_dict[task], dim=1)

                        # Store for metrics
                        all_outputs[task].append(preds.detach().cpu())
                        all_targets[task].append(targets_dict[task].cpu())

                # Update progress bar
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({'loss': f"{avg_loss:.4f}"})

        # Calculate overall average loss
        avg_loss = total_loss / len(self.val_loader)

        # Calculate task-specific metrics
        metrics = {}
        for task in ['multiclass', 'contralateral', 'ipsilateral']:
            if all_outputs[task] and all_targets[task]:
                # Concatenate all predictions and targets
                preds = torch.cat(all_outputs[task]).numpy()
                targets = torch.cat(all_targets[task]).numpy()

                # Calculate metrics
                metrics[task] = calculate_metrics(targets, preds, task)

        return avg_loss, metrics

    def evaluate(self):
        """Evaluate the model on the test set with model-specific handling."""
        print("\nEvaluating model on test set...")
        self.model.eval()
        total_loss = 0

        # Initialize containers for predictions and targets
        all_predictions = {
            'multiclass': [], 'contralateral': [], 'ipsilateral': [],
            'multiclass_probs': [], 'contralateral_probs': [], 'ipsilateral_probs': []
        }

        all_targets = {'multiclass': [], 'contralateral': [], 'ipsilateral': []}

        # Check if neural activity prediction is available (only for hybrid model)
        sample_batch = next(iter(self.test_loader))
        if 'neural_activity' in sample_batch[1] and self.model_type == 'hybrid':
            all_predictions['neural_activity'] = []
            all_targets['neural_activity'] = []

        # No gradient tracking for evaluation
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")

            for batch_idx, batch in enumerate(pbar):
                # Get inputs and targets
                inputs, targets_dict = batch
                inputs = inputs.to(self.device)
                targets_dict = {k: v.to(self.device) for k, v in targets_dict.items()}

                # Forward pass
                outputs_dict = self.model(inputs)

                # Calculate loss with task weighting
                loss, individual_losses = self.criterion(outputs_dict, targets_dict)
                total_loss += loss.item()

                # Store predictions and targets for each task
                for task in ['multiclass', 'contralateral', 'ipsilateral']:
                    if task in outputs_dict and task in targets_dict:
                        # Get predictions and probabilities
                        probs = F.softmax(outputs_dict[task], dim=1)
                        _, preds = torch.max(outputs_dict[task], dim=1)

                        # Store predictions, probabilities, and targets
                        all_predictions[task].append(preds.cpu().numpy())
                        all_predictions[f'{task}_probs'].append(probs.cpu().numpy())
                        all_targets[task].append(targets_dict[task].cpu().numpy())

                # Store neural activity predictions if available (hybrid model only)
                if ('neural_activity' in outputs_dict and
                        'neural_activity' in targets_dict and
                        self.model_type == 'hybrid'):
                    all_predictions['neural_activity'].append(
                        outputs_dict['neural_activity'].cpu().numpy())
                    all_targets['neural_activity'].append(
                        targets_dict['neural_activity'].cpu().numpy())

                # Update progress bar
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

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
                    all_targets[task], all_predictions[task], task)

                # Print task-specific metrics
                print(f"\n{task.capitalize()} Test Metrics:")
                for metric, value in test_metrics[task].items():
                    print(f"  {metric}: {value:.4f}")

        # Add overall accuracy
        overall_acc = np.mean([metrics['accuracy'] for task, metrics in test_metrics.items()])
        test_metrics['overall'] = {'accuracy': overall_acc}
        print(f"\nOverall Test Accuracy: {overall_acc:.4f}")

        return avg_loss, test_metrics, all_predictions, all_targets

    def _log_to_wandb(self, epoch, train_loss, val_loss, train_metrics, val_metrics, lr):
        """Log metrics to WandB if available."""
        if self.run is not None:
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'learning_rate': lr
            }

            # Add task-specific metrics
            for task in ['multiclass', 'contralateral', 'ipsilateral']:
                if task in train_metrics:
                    for metric, value in train_metrics[task].items():
                        log_dict[f'train/{task}_{metric}'] = value
                if task in val_metrics:
                    for metric, value in val_metrics[task].items():
                        log_dict[f'val/{task}_{metric}'] = value

            wandb.log(log_dict)

    def _save_training_history(self, train_history, val_history):
        """Save training history to JSON file."""
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            history = {
                'train': {k: [float(v) for v in vals] for k, vals in train_history.items()},
                'val': {k: [float(v) for v in vals] for k, vals in val_history.items()}
            }
            json.dump(history, f, indent=4)

    def _generate_and_save_visualizations(self, predictions, targets):
        """Generate and save all visualizations."""
        viz_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)

        # Define class names for visualization
        class_names = {
            'multiclass': ['No Footstep', 'Contralateral', 'Ipsilateral'],
            'contralateral': ['No Footstep', 'Contralateral'],
            'ipsilateral': ['No Footstep', 'Ipsilateral']
        }

        # Generate confusion matrices
        for task in ['multiclass', 'contralateral', 'ipsilateral']:
            if task in predictions and task in targets:
                viz.plot_confusion_matrix(
                    true_labels=targets[task],
                    predicted_labels=predictions[task],
                    class_names=class_names[task],
                    include_percentages=True,
                    title=f"{self.model_type.upper()} Model - {task.capitalize()} Confusion Matrix",
                    save_path=os.path.join(viz_dir, f'{task}_confusion_matrix.png')
                )

        # Generate ROC curves
        for task in ['multiclass', 'contralateral', 'ipsilateral']:
            if f'{task}_probs' in predictions and task in targets:
                viz.plot_roc_curves(
                    true_labels=targets[task],
                    predicted_probs=predictions[f'{task}_probs'],
                    class_names=class_names[task],
                    title=f"{self.model_type.upper()} Model - {task.capitalize()} ROC Curves",
                    save_path=os.path.join(viz_dir, f'{task}_roc_curves.png')
                )

        # Generate neural activity visualization for hybrid model only
        if self.model_type == 'hybrid' and 'neural_activity' in predictions:
            viz.plot_neural_activity_comparison(
                time_points=np.arange(len(targets['multiclass'])),
                true_neural=targets['neural_activity'],
                pred_neural=predictions['neural_activity'],
                true_behaviors=targets['multiclass'],
                pred_behaviors=predictions['multiclass'],
                save_path=os.path.join(viz_dir, 'neural_activity_figure4.png')
            )

    def _save_test_results(self, test_loss, test_metrics):
        """Save test results to JSON file."""
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

    def _save_checkpoint(self, state, filename):
        """Save model checkpoint."""
        save_path = os.path.join(self.config.paths.checkpoints_dir, self.model_type, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(state, save_path)

    def _load_checkpoint(self, filename):
        """Load model checkpoint."""
        load_path = os.path.join(self.config.paths.checkpoints_dir, self.model_type, filename)
        if os.path.exists(load_path):
            checkpoint = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint (epoch {checkpoint['epoch'] + 1})")
        else:
            print(f"No checkpoint found at {load_path}")

    def _log_history_plots(self, train_history, val_history):
        """Generate and log training history plots to WandB."""
        try:
            # Create subplot for loss
            fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
            ax_loss.plot(train_history['loss'], label='Train Loss')
            ax_loss.plot(val_history['loss'], label='Validation Loss')
            ax_loss.set_xlabel('Epoch')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_title('Training and Validation Loss')
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3)

            # Log loss plot
            if self.run:
                wandb.log({"loss_history": wandb.Image(fig_loss)})
            plt.close(fig_loss)

        except Exception as e:
            print(f"Error generating history plots: {e}")

    def _mixup_batch(self, inputs, targets_dict):
        """Apply mixup data augmentation (only used for hybrid model)."""
        batch_size = inputs.size(0)

        # Sample lambda from beta distribution
        alpha = getattr(self.config.training, 'mixup_alpha', 0.2)
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
                # For classification tasks, keep original targets but track lambda
                mixed_targets_dict[task] = targets
                mixed_targets_dict[f"{task}_mixup_index"] = index
                mixed_targets_dict[f"{task}_mixup_lambda"] = lam.squeeze()

        return mixed_inputs, mixed_targets_dict

