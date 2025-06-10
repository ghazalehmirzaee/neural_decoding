# src/models/cnn_only_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNOnlyModel(nn.Module):
    """
    CNN-only model for neuronal decoding - NO attention mechanisms.

    This model serves as a baseline to test spatial feature extraction alone.

    Key design principles:
    1. ONLY convolutional layers for feature extraction
    2. NO attention mechanisms
    """

    def __init__(
            self,
            input_size,  # Number of neurons
            num_classes=3,
            dropout=0.3  # Lower dropout for simpler model
    ):
        super(CNNOnlyModel, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        print(f"Initializing Pure CNN-only model (no attention) with {input_size} neurons")

        # Simple temporal CNN architecture
        # Layer 1: Detect immediate neural responses
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        # Layer 2: Detect short-term patterns
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        # Layer 3: Detect longer temporal patterns
        self.conv3 = nn.Conv1d(128, 256, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(256)

        # Layer 4: High-level feature extraction
        self.conv4 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)

        # Dropout layers
        self.dropout = nn.Dropout(dropout)

        # Global pooling to aggregate temporal information
        # This is the CNN's way of reducing sequence to single prediction
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Combine avg and max pooling
        combined_features = 256 * 2  # avg + max

        # Classifier head
        self.feature_reducer = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Task-specific heads (same as other models)
        self.multiclass_head = nn.Linear(64, num_classes)
        self.contralateral_head = nn.Linear(64, 2)
        self.ipsilateral_head = nn.Linear(64, 2)

        # Initialize weights
        self._initialize_weights()

        # Print architecture summary
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Pure CNN model: {total_params:,} parameters")

    def _initialize_weights(self):
        """Initialize with careful attention to avoid vanishing/exploding gradients."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Use He initialization for ReLU networks
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Smaller initialization for final layers to avoid saturation
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through pure CNN architecture.
        """
        batch_size, seq_len, input_size = x.shape

        # Transpose for conv1d: [batch, seq_len, neurons] -> [batch, neurons, seq_len]
        x = x.transpose(1, 2)  # [batch, neurons, seq_len]

        # Sequential CNN layers with residual-like connections
        # Layer 1: Basic temporal patterns
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.dropout(x1)

        # Layer 2: Combine with skip connection
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = self.dropout(x2)

        # Layer 3: Higher-level patterns
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = self.dropout(x3)

        # Layer 4: Final feature extraction
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x4 = self.dropout(x4)

        # Global pooling to compress temporal dimension
        # This is critical: CNN must reduce sequence to single vector
        avg_pooled = self.global_avg_pool(x4).squeeze(-1)  # [batch, 256]
        max_pooled = self.global_max_pool(x4).squeeze(-1)  # [batch, 256]

        # Combine pooled features
        combined = torch.cat([avg_pooled, max_pooled], dim=1)  # [batch, 512]

        # Feature reduction and classification
        features = self.feature_reducer(combined)  # [batch, 64]

        # Multi-task outputs
        outputs = {
            'multiclass': self.multiclass_head(features),
            'contralateral': self.contralateral_head(features),
            'ipsilateral': self.ipsilateral_head(features)
        }

        return outputs


# function for integration
def create_cnn_model(config):
    """Create CNN model that integrates with existing training pipeline."""
    return CNNOnlyModel(
        input_size=config.model.input_size,
        num_classes=config.model.num_classes,
        dropout=getattr(config.model, 'dropout', 0.3)
    )
