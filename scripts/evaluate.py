# scripts/evaluate.py

import os
import sys
import hydra
import torch
import json
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.lstm_model import LSTMModel
from src.models.hybrid_model import HybridCNNBiLSTM
from src.data.dataset import NeuralDataset, create_data_loaders
from src.training.trainer import Trainer
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_neural_activity
)


@hydra.main(config_path="../config", config_name="config")
def main(config: DictConfig):
    """
    Main evaluation function.

    This evaluates either the LSTM or Hybrid model based on the configuration.
    """
    print(f"Configuration: \n{OmegaConf.to_yaml(config)}")

    # Determine device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(config.paths.output_dir, exist_ok=True)

    # Create dataset
    print("\nCreating dataset...")
    dataset = NeuralDataset(
        data_path=config.dataset.path,
        sequence_length=config.dataset.sequence_length,
        apply_pca=config.dataset.apply_pca,
        n_components=config.dataset.n_components,
        normalize=config.dataset.normalize
    )

    # Create data loaders
    print("\nCreating data loaders...")
    _, _, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=config.dataset.batch_size,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        seed=config.seed
    )

    # Create model
    print(f"\nCreating {config.model.type} model...")
    if config.model.type == 'lstm':
        model = LSTMModel(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout
        )
    else:
        model = HybridCNNBiLSTM(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout
        )

    # Load pretrained weights
    checkpoint_path = os.path.join(
        config.paths.checkpoints_dir,
        config.model.type,
        'best_model.pth'
    )

    if os.path.exists(checkpoint_path):
        print(f"\nLoading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"\nNo checkpoint found at {checkpoint_path}.")
        print("Please train the model first.")
        return

    # Move model to device
    model.to(device)

    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        model_type=config.model.type,
        train_loader=None,
        val_loader=None,
        test_loader=test_loader,
        config=config,
        device=device
    )

    # Evaluate model
    loss, metrics, predictions, targets = trainer.evaluate()

    # Print results
    print("\nEvaluation Results:")
    print(f"Loss: {loss:.4f}")

    for task in ['multiclass', 'contralateral', 'ipsilateral']:
        print(f"\n{task.capitalize()}:")
        for metric, value in metrics[task].items():
            print(f"  {metric}: {value:.4f}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualization_dir = os.path.join(
        config.paths.visualizations_dir,
        config.model.type
    )
    os.makedirs(visualization_dir, exist_ok=True)

    # Define class names for visualization
    class_names = {
        'multiclass': ['No Footstep', 'Contralateral', 'Ipsilateral'],
        'contralateral': ['Negative', 'Positive'],
        'ipsilateral': ['Negative', 'Positive']
    }

    # Generate confusion matrices for all classification tasks
    for task in ['multiclass', 'contralateral', 'ipsilateral']:
        plot_confusion_matrix(
            true_labels=targets[task],
            predicted_labels=predictions[task],
            class_names=class_names[task],
            include_percentages=True,
            title=f"{config.model.type.upper()} Model - {task.capitalize()} Confusion Matrix",
            save_path=os.path.join(visualization_dir, f'{task}_confusion_matrix.png')
        )

    # Generate ROC curves for all classification tasks
    for task in ['multiclass', 'contralateral', 'ipsilateral']:
        if f"{task}_probs" in predictions:
            plot_roc_curves(
                true_labels=targets[task],
                predicted_probs=predictions[f"{task}_probs"],
                class_names=class_names[task],
                title=f"{config.model.type.upper()} Model - {task.capitalize()} ROC Curves",
                save_path=os.path.join(visualization_dir, f'{task}_roc_curves.png')
            )

    # Generate neural activity plots for hybrid model
    if config.model.type == 'hybrid' and 'neural_activity' in predictions:
        n_samples = len(targets['multiclass'])
        window_size = 200

        for start_idx in range(0, n_samples, window_size):
            end_idx = min(start_idx + window_size, n_samples)

            plot_neural_activity(
                time_points=np.arange(start_idx, end_idx),
                true_neural=targets['neural_activity'][start_idx:end_idx],
                pred_neural=predictions['neural_activity'][start_idx:end_idx],
                true_behaviors=targets['multiclass'][start_idx:end_idx],
                pred_behaviors=predictions['multiclass'][start_idx:end_idx],
                behavior_names=class_names['multiclass'],
                window_size=window_size,
                title=f"{config.model.type.upper()} Model - Neural Activity and Behavior (Window {start_idx}-{end_idx})",
                save_path=os.path.join(visualization_dir, f'neural_activity_{start_idx}_{end_idx}.png')
            )

    print(f"\nEvaluation completed. Results and visualizations saved to {config.paths.output_dir}")


if __name__ == "__main__":
    main()

