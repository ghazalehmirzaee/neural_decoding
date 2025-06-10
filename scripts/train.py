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

# Import all model types - this is crucial for supporting the full model suite
from src.models.lstm_model import LSTMModel
from src.models.hybrid_model import HybridCNNBiLSTM
from src.models.cnn_model import CNNOnlyModel
from src.models.lstm_attention_model import LSTMAttentionModel  # This was missing!
from src.data.dataset import NeuralDataset, create_data_loaders
from src.training.trainer import Trainer


@hydra.main(config_path="../config", config_name="config")
def main(config: DictConfig):
    """
    Main training function supporting all model architectures.

    This function orchestrates the complete training pipeline for neuronal decoding models.
    1. CNN: Captures spatial patterns in neuronal populations
    2. LSTM: Models temporal dynamics in neuronal sequences
    3. LSTM+Attention: Enhances temporal modeling with attention mechanisms
    4. Hybrid CNN-BiLSTM: Combines spatial and temporal processing
    """
    # Print configuration for reference and debugging
    print(f"Configuration: \n{OmegaConf.to_yaml(config)}")

    # Set random seeds for reproducibility across all libraries
    # This ensures consistent results across different runs
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        # These settings ensure deterministic behavior on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Determine device - automatically use GPU if available
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create all necessary directories
    # This ensures the file system is ready for outputs
    os.makedirs(config.paths.data_dir, exist_ok=True)
    os.makedirs(config.paths.output_dir, exist_ok=True)
    os.makedirs(config.paths.checkpoints_dir, exist_ok=True)
    os.makedirs(config.paths.logs_dir, exist_ok=True)
    os.makedirs(config.paths.visualizations_dir, exist_ok=True)

    # Determine sequence length based on model architecture
    if config.model.type == 'lstm':
        # Basic LSTM uses shorter sequences as specified in the paper
        sequence_length = 10
        print("LSTM model: Using short sequences for basic temporal patterns")
    elif config.model.type == 'cnn':
        # CNN needs longer sequences to detect spatial-temporal patterns
        sequence_length = 32
        print("CNN model: Using longer sequences for spatial pattern detection")
    elif config.model.type == 'lstm_attention':
        # LSTM+Attention benefits from moderate sequence length
        # Balances computational cost with temporal context
        sequence_length = 16
        print("LSTM+Attention model: Using moderate sequences for attention-based processing")
    else:  # hybrid
        # Hybrid model uses full sequences as specified in Table 2
        sequence_length = 32
        print("Hybrid model: Using full sequences for complete spatial-temporal processing")

    print(f"Selected sequence length: {sequence_length}")

    # Create dataset with appropriate preprocessing
    print("\nCreating dataset...")
    dataset = NeuralDataset(
        data_path=config.dataset.path,
        sequence_length=sequence_length,
        apply_pca=config.dataset.apply_pca,
        n_components=config.dataset.n_components,
        normalize=config.dataset.normalize
    )

    # Analyze and display class distribution
    # This helps understand data imbalance and informs training strategies
    print("\nDataset class distribution:")
    unique_classes, class_counts = np.unique(dataset.y_multiclass, return_counts=True)
    total_samples = len(dataset.y_multiclass)
    for cls, count in zip(unique_classes, class_counts):
        percentage = (count / total_samples) * 100
        class_name = ['No Footstep', 'Contralateral', 'Ipsilateral'][int(cls)]
        print(f"  Class {cls} ({class_name}): {count} samples ({percentage:.1f}%)")

    # Determine batch size based on model complexity
    # Larger models need smaller batch sizes due to memory constraints
    if config.model.type == 'lstm':
        batch_size = 64
    elif config.model.type == 'lstm_attention':
        # LSTM+Attention uses moderate batch size
        batch_size = 48
    else:  # cnn or hybrid
        batch_size = 32

    print(f"\nCreating data loaders with batch size {batch_size}...")
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset=dataset,
        batch_size=batch_size,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        seed=config.seed
    )

    # Create model based on configuration
    print(f"\nCreating {config.model.type} model...")

    if config.model.type == 'lstm':
        # Basic LSTM model as described in the paper's first approach
        model = LSTMModel(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout
        )
        print("✓ LSTM model created for temporal sequence processing")

    elif config.model.type == 'cnn':
        # CNN model for spatial pattern detection
        # This serves as a baseline for spatial-only processing
        model = CNNOnlyModel(
            input_size=config.model.input_size,
            num_classes=config.model.num_classes,
            dropout=getattr(config.model, 'dropout', 0.3)
        )
        print("✓ Pure CNN model created (no attention, no LSTM)")
        print("  - Focuses on spatial patterns in neural populations")
        print("  - Uses global pooling to aggregate temporal information")

    elif config.model.type == 'lstm_attention':
        # LSTM with attention mechanism for enhanced temporal processing
        # This model bridges the gap between basic LSTM and full hybrid
        model = LSTMAttentionModel(
            input_size=config.model.input_size,
            hidden_size=getattr(config.model, 'hidden_size', 96),
            num_layers=config.model.num_layers,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout,
            num_attention_heads=getattr(config.model, 'num_attention_heads', 4),
            attention_dim=getattr(config.model, 'attention_dim', 48)
        )
        print("✓ LSTM+Attention model created with temporal attention mechanism")
        print("  - Enhances LSTM with multi-head attention")
        print("  - Focuses on behaviorally-relevant temporal patterns")

    else:  # hybrid
        # Full hybrid model combining CNN and BiLSTM with attention
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
        print("✓ Hybrid CNN-BiLSTM model created with full feature set")
        print("  - CNN layers for spatial feature extraction")
        print("  - BiLSTM for bidirectional temporal processing")
        print("  - Multi-head attention for focusing on relevant patterns")
        print("  - Skip connections for preserving fine-grained information")

    # Display model architecture and complexity
    print(f"\nModel architecture:")
    print(model)

    # Calculate and display parameter counts
    # This helps understand model complexity and memory requirements
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Estimate model size in MB
    param_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"Approximate model size: {param_size_mb:.2f} MB")

    # Create trainer instance with all necessary components
    trainer = Trainer(
        model=model,
        model_type=config.model.type,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )

    # Start the training process
    print(f"\nStarting training for {config.model.type} model...")
    print(f"Training will run for up to {config.training.epochs} epochs")
    print(f"Early stopping patience: {config.training.early_stopping_patience} epochs")

    # Train model and get final predictions
    predictions, targets = trainer.train()

    return model, predictions, targets


if __name__ == "__main__":
    main()

