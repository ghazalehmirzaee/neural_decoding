# src/data/dataset.py

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class NeuralDataset(Dataset):
    """
    Dataset for neural activity data with optimized sequence handling for different models.
    """

    def __init__(
            self,
            data_path,
            sequence_length=32,  # Default for hybrid model
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

        # Extract neural data (columns between frame index and behavioral label)
        self.features = data.iloc[:, 1:-1].values

        # Store original features and neuron count
        self.original_features = self.features.copy()
        self.num_neurons = self.features.shape[1]

        print(f"Number of neurons: {self.num_neurons}")
        print(f"Using sequence length: {sequence_length}")

        # Apply normalization to handle varying calcium signal amplitudes
        if normalize:
            print("Normalizing neural signals...")
            scaler = StandardScaler()
            self.features = scaler.fit_transform(self.features)

        # Apply PCA for visualization and dimensionality reduction
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
        Create optimized sequences for all classification tasks with proper overlap.
        """
        # Initialize lists
        self.X = []  # Neural activity sequences
        self.y_multiclass = []  # 3-class labels (0: no footstep, 1: contralateral, 2: ipsilateral)
        self.y_contralateral = []  # Binary labels for contralateral footstep
        self.y_ipsilateral = []  # Binary labels for ipsilateral footstep
        self.y_neural = []  # Neural activity target for regularization
        self.sequence_indices = []  # Indices for each sequence (for debugging)

        # Create sequences with proper consideration for behavioral state
        valid_count = 0
        for i in range(len(self.features) - self.sequence_length + 1):
            # Extract sequence
            sequence = self.features[i:i + self.sequence_length]

            # Get label from the end of the sequence (predicting the current state)
            # This aligns with the paper's approach of predicting the current movement state
            behavioral_label = self.behavioral_labels[i + self.sequence_length - 1]

            # Create binary labels for specific tasks as mentioned in the paper
            # 0: no footstep, 1: contralateral, 2: ipsilateral
            contralateral_label = 1 if behavioral_label == 1 else 0
            ipsilateral_label = 1 if behavioral_label == 2 else 0

            # Extract neural activity for the prediction target (used in hybrid model)
            if self.pca is not None:
                # Use first principal component as representative neural activity
                neural_activity = self.pca_features[i + self.sequence_length - 1, 0]
            else:
                # Use mean activity across neurons when PCA not available
                neural_activity = np.mean(self.original_features[i + self.sequence_length - 1])

            # Store sequence and labels
            self.X.append(sequence)
            self.y_multiclass.append(behavioral_label)
            self.y_contralateral.append(contralateral_label)
            self.y_ipsilateral.append(ipsilateral_label)
            self.y_neural.append(neural_activity)
            self.sequence_indices.append(i + self.sequence_length - 1)
            valid_count += 1

        # Convert to numpy arrays
        self.X = np.array(self.X)
        self.y_multiclass = np.array(self.y_multiclass)
        self.y_contralateral = np.array(self.y_contralateral)
        self.y_ipsilateral = np.array(self.y_ipsilateral)
        self.y_neural = np.array(self.y_neural)
        self.sequence_indices = np.array(self.sequence_indices)

        print(f"Created {len(self.X)} sequences of length {self.sequence_length}")
        print(f"Sequence shape: {self.X.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get sequence and labels for all tasks.
        Returns a tuple of (input_sequence, target_dict).
        """
        x = torch.FloatTensor(self.X[idx])

        # Create dictionary of targets for all tasks
        targets = {
            'multiclass': torch.LongTensor([self.y_multiclass[idx]]).squeeze(),
            'contralateral': torch.LongTensor([self.y_contralateral[idx]]).squeeze(),
            'ipsilateral': torch.LongTensor([self.y_ipsilateral[idx]]).squeeze(),
            'neural_activity': torch.FloatTensor([self.y_neural[idx]]).squeeze()
        }

        return x, targets


def create_data_loaders(dataset, batch_size, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Create data loaders with appropriate batch sizes for each model type.
    Ensures proper stratification to maintain class distribution.
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Calculate dataset sizes
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Stratified split to maintain class balance
    # First, get indices for each class
    indices = np.arange(dataset_size)
    multiclass_labels = dataset.y_multiclass

    class_indices = {c: np.where(multiclass_labels == c)[0] for c in np.unique(multiclass_labels)}

    # Now create stratified splits for each class
    train_indices = []
    val_indices = []
    test_indices = []

    for c, c_indices in class_indices.items():
        np.random.shuffle(c_indices)

        c_train_size = int(train_ratio * len(c_indices))
        c_val_size = int(val_ratio * len(c_indices))

        train_indices.extend(c_indices[:c_train_size])
        val_indices.extend(c_indices[c_train_size:c_train_size + c_val_size])
        test_indices.extend(c_indices[c_train_size + c_val_size:])

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create data loaders with optimal parameters for training
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches for more stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    print(f"Batch size: {batch_size}")

    # Verify class distribution across splits
    print("Verifying class distribution:")
    for name, dataset_split in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        labels = [dataset.y_multiclass[i] for i in dataset_split.indices]
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  {name}: {dict(zip(unique, counts))}")

    return train_loader, val_loader, test_loader


