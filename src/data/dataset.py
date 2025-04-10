# src/data/dataset.py (Updated sequence_length handling)

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class NeuralDataset(Dataset):
    """
    Dataset for neural activity data with support for all three classification tasks.
    """

    def __init__(
            self,
            data_path,
            sequence_length=32,  # Default for hybrid model, LSTM should use 10
            apply_pca=True,
            n_components=3,
            normalize=True
    ):
        self.sequence_length = sequence_length

        # Load data
        data = pd.read_csv(data_path)
        print(f"Loaded data shape: {data.shape}")

        # Extract features and labels
        self.frame_indices = data.iloc[:, 0].values
        self.behavioral_labels = data.iloc[:, -1].values

        # Extract neural data
        self.features = data.iloc[:, 1:-1].values

        # Store original features and neuron count
        self.original_features = self.features.copy()
        self.num_neurons = self.features.shape[1]

        print(f"Number of neurons: {self.num_neurons}")
        print(f"Sequence length: {sequence_length}")

        # Apply normalization
        if normalize:
            print("Normalizing features...")
            scaler = StandardScaler()
            self.features = scaler.fit_transform(self.features)

        # Apply PCA for visualization
        if apply_pca:
            print("Applying PCA for neural activity visualization...")
            self.pca = PCA(n_components=n_components)
            self.pca_features = self.pca.fit_transform(self.features)
            print(f"Explained variance ratios: {self.pca.explained_variance_ratio_}")
        else:
            self.pca = None
            self.pca_features = None

        # Create sequences
        self._create_sequences()

        # Print class distribution
        print("Class distribution:")
        for task in ['multiclass', 'contralateral', 'ipsilateral']:
            labels = getattr(self, f'y_{task}')
            unique, counts = np.unique(labels, return_counts=True)
            print(f"  {task}: {dict(zip(unique, counts))}")

    def _create_sequences(self):
        """
        Create sequences for all three classification tasks.
        """
        # Initialize lists
        self.X = []
        self.y_multiclass = []
        self.y_contralateral = []
        self.y_ipsilateral = []
        self.y_neural = []
        self.sequence_indices = []

        # Create sequences
        for i in range(len(self.features) - self.sequence_length + 1):
            # Extract sequence
            sequence = self.features[i:i + self.sequence_length]

            # Extract label for the last timestep
            behavioral_label = self.behavioral_labels[i + self.sequence_length - 1]

            # Convert to binary labels for specific tasks
            # 0: no footstep, 1: contralateral, 2: ipsilateral
            contralateral_label = 1 if behavioral_label == 1 else 0
            ipsilateral_label = 1 if behavioral_label == 2 else 0

            # Extract neural activity
            if self.pca is not None:
                neural_activity = self.pca_features[i + self.sequence_length - 1, 0]
            else:
                neural_activity = np.mean(self.original_features[i + self.sequence_length - 1])

            # Store sequence and labels
            self.X.append(sequence)
            self.y_multiclass.append(behavioral_label)
            self.y_contralateral.append(contralateral_label)
            self.y_ipsilateral.append(ipsilateral_label)
            self.y_neural.append(neural_activity)
            self.sequence_indices.append(i + self.sequence_length - 1)

        # Convert to numpy arrays
        self.X = np.array(self.X)
        self.y_multiclass = np.array(self.y_multiclass)
        self.y_contralateral = np.array(self.y_contralateral)
        self.y_ipsilateral = np.array(self.y_ipsilateral)
        self.y_neural = np.array(self.y_neural)
        self.sequence_indices = np.array(self.sequence_indices)

        print(f"Created {len(self.X)} sequences of length {self.sequence_length}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get sequence and labels for the given index.
        """
        x = torch.FloatTensor(self.X[idx])

        targets = {
            'multiclass': torch.LongTensor([self.y_multiclass[idx]]).squeeze(),
            'contralateral': torch.LongTensor([self.y_contralateral[idx]]).squeeze(),
            'ipsilateral': torch.LongTensor([self.y_ipsilateral[idx]]).squeeze(),
            'neural_activity': torch.FloatTensor([self.y_neural[idx]]).squeeze()
        }

        return x, targets


def create_data_loaders(dataset, batch_size, train_ratio=0.7, val_ratio=0.15, seed=42, model_type='hybrid'):
    """
    Create data loaders with model-specific parameters.

    Args:
        model_type: Either 'lstm' (batch_size=64) or 'hybrid' (batch_size=32)
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Calculate dataset sizes
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Set batch size based on model type
    if model_type == 'lstm':
        actual_batch_size = 64  # As specified in Table 1
    else:
        actual_batch_size = 32  # As specified in Table 2

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Using batch size {actual_batch_size} for {model_type} model")

    return train_loader, val_loader, test_loader

