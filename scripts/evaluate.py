# scripts/evaluate.py

import os
import sys
import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.lstm_model import LSTMModel
from src.models.hybrid_model import HybridCNNBiLSTM
from src.data.dataset import NeuralDataset, create_data_loaders
from src.training.trainer import Trainer
import src.utils.visualization as viz


@hydra.main(config_path="../config", config_name="config")
def main(config: DictConfig):
    """
    Main evaluation function.

    Evaluates a trained LSTM or Hybrid CNN-BiLSTM model on the test set.
    Generates comprehensive visualizations and metrics.
    """
    # Print configuration for reference
    print(f"Configuration: \n{OmegaConf.to_yaml(config)}")

    # Determine device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(config.paths.output_dir, exist_ok=True)
    os.makedirs(config.paths.visualizations_dir, exist_ok=True)

    # Set sequence length based on model type (10 for LSTM, 32 for Hybrid)
    sequence_length = 10 if config.model.type == 'lstm' else 32
    print(f"Using model-specific sequence length: {sequence_length}")

    # Create dataset with appropriate sequence length
    print("\nCreating dataset...")
    dataset = NeuralDataset(
        data_path=config.dataset.path,
        sequence_length=sequence_length,
        apply_pca=config.dataset.apply_pca,
        n_components=config.dataset.n_components,
        normalize=config.dataset.normalize
    )

    # Create data loaders with appropriate batch size
    batch_size = 64 if config.model.type == 'lstm' else 32  # From Tables 1 & 2
    print(f"\nCreating data loaders with batch size {batch_size}...")
    _, _, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=batch_size,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        seed=config.seed
    )

    # Create model architecture with the same configuration as training
    print(f"\nCreating {config.model.type} model architecture...")
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
            dropout=config.model.dropout,
            use_skip_connection=getattr(config.model, 'use_skip_connection', True),
            use_attention=getattr(config.model, 'use_attention', True),
            num_attention_heads=getattr(config.model, 'num_attention_heads', 8)
        )

    # Move model to device
    model.to(device)

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
        print(f"Loaded model from epoch {checkpoint['epoch'] + 1}")
    else:
        print(f"\nNo checkpoint found at {checkpoint_path}.")
        print("Please train the model first.")
        return

    # Create trainer for evaluation only
    trainer = Trainer(
        model=model,
        model_type=config.model.type,
        train_loader=None,  # Not needed for evaluation
        val_loader=None,  # Not needed for evaluation
        test_loader=test_loader,
        config=config,
        device=device
    )

    # Evaluate model
    print("\nEvaluating model on test set...")
    loss, metrics, predictions, targets = trainer.evaluate()

    # Print summary of results
    print("\nEvaluation Results Summary:")
    print(f"Loss: {loss:.4f}")

    for task in ['multiclass', 'contralateral', 'ipsilateral']:
        if task in metrics:
            print(f"\n{task.capitalize()} Task:")
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

    # Generate confusion matrices
    for task in ['multiclass', 'contralateral', 'ipsilateral']:
        if task in predictions and task in targets:
            viz.plot_confusion_matrix(
                true_labels=targets[task],
                predicted_labels=predictions[task],
                class_names=class_names[task],
                include_percentages=True,
                title=f"{config.model.type.upper()} Model - {task.capitalize()} Confusion Matrix",
                save_path=os.path.join(visualization_dir, f'{task}_confusion_matrix.png')
            )

    # Generate ROC curves
    for task in ['multiclass', 'contralateral', 'ipsilateral']:
        if f'{task}_probs' in predictions and task in targets:
            viz.plot_roc_curves(
                true_labels=targets[task],
                predicted_probs=predictions[f'{task}_probs'],
                class_names=class_names[task],
                title=f"{config.model.type.upper()} Model - {task.capitalize()} ROC Curves",
                save_path=os.path.join(visualization_dir, f'{task}_roc_curves.png')
            )

    # Generate neural activity plots for hybrid model
    if config.model.type == 'hybrid' and 'neural_activity' in predictions:
        # Generate Figure 4 style visualization from the paper
        viz.plot_neural_activity_comparison(
            time_points=np.arange(len(targets['multiclass'])),
            true_neural=targets['neural_activity'],
            pred_neural=predictions['neural_activity'],
            true_behaviors=targets['multiclass'],
            pred_behaviors=predictions['multiclass'],
            save_path=os.path.join(visualization_dir, 'neural_activity_figure4.png')
        )

    print(f"\nEvaluation completed. Results and visualizations saved to {visualization_dir}")


if __name__ == "__main__":
    main()

