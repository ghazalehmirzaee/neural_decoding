# scripts/train.py

import os
import sys
import hydra
import torch
import random
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.models.lstm_model import LSTMModel
from src.models.hybrid_model import HybridCNNBiLSTM
from src.data.dataset import NeuralDataset, create_data_loaders
from src.training.trainer import Trainer


@hydra.main(config_path="../config", config_name="config")
def main(config: DictConfig):
    """
    Main training function.

    Trains either the LSTM or Hybrid CNN-BiLSTM model based on the configuration.
    Implements proper sequence length handling and initialization.
    """
    # Print configuration for reference
    print(f"Configuration: \n{OmegaConf.to_yaml(config)}")

    # Set random seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Determine device
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(config.paths.data_dir, exist_ok=True)
    os.makedirs(config.paths.output_dir, exist_ok=True)
    os.makedirs(config.paths.checkpoints_dir, exist_ok=True)
    os.makedirs(config.paths.logs_dir, exist_ok=True)
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
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=batch_size,
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
            dropout=config.model.dropout,
            use_skip_connection=getattr(config.model, 'use_skip_connection', True),
            use_attention=getattr(config.model, 'use_attention', True),
            num_attention_heads=getattr(config.model, 'num_attention_heads', 8)
        )

    # Print model summary
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Create trainer
    trainer = Trainer(
        model=model,
        model_type=config.model.type,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )

    # Train model
    print("\nStarting training...")
    predictions, targets = trainer.train()

    print(f"\nTraining and evaluation completed. Results and visualizations saved to {config.paths.output_dir}")

    return model, predictions, targets


if __name__ == "__main__":
    main()

