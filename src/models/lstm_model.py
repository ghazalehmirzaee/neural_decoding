# src/models/lstm_model.py

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM model for neural decoding with fully connected layers.

    This model implements the architecture described in the paper:
    - LSTM layers with configurable hidden size
    - Batch normalization
    - Fully connected layers (64→32)
    - Multiple output heads for different classification tasks
    """

    def __init__(
            self,
            input_size,
            hidden_size=64,
            num_layers=2,
            num_classes=3,
            dropout=0.5
    ):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer with dropout
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Batch normalization layer
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Fully connected layers as described in the paper
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Task-specific output heads
        # 1. Multiclass classification (no footstep, contralateral, ipsilateral)
        self.multiclass_head = nn.Linear(32, num_classes)

        # 2. Binary classification for contralateral footstep
        self.contralateral_head = nn.Linear(32, 2)

        # 3. Binary classification for ipsilateral footstep
        self.ipsilateral_head = nn.Linear(32, 2)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Orthogonal initialization as in Table 1."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name and len(param.shape) >= 2:
                    # Orthogonal initialization for LSTM weights
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    # Xavier initialization for linear layers
                    nn.init.xavier_uniform_(param)
                else:
                    # Normal initialization for 1D weights
                    nn.init.normal_(param, mean=0.0, std=0.1)
            elif 'bias' in name:
                # Zero initialization for biases
                nn.init.zeros_(param)

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Dictionary containing outputs for all three classification tasks
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = out[:, -1, :]

        # Apply batch normalization
        out = self.batch_norm(out)

        # Fully connected layers
        out = self.relu1(self.fc1(out))
        out = self.dropout1(out)
        out = self.relu2(self.fc2(out))
        out = self.dropout2(out)

        # Apply task-specific heads
        multiclass_output = self.multiclass_head(out)
        contralateral_output = self.contralateral_head(out)
        ipsilateral_output = self.ipsilateral_head(out)

        # Return all outputs for the three tasks
        return {
            'multiclass': multiclass_output,
            'contralateral': contralateral_output,
            'ipsilateral': ipsilateral_output
        }

