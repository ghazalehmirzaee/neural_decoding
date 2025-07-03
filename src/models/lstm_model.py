# src/models/lstm_model.py

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    LSTM model for neural decoding with fully connected layers as specified in Table 1.
    Implements the first model approach described in the paper.
    """

    def __init__(
            self,
            input_size,
            hidden_size=64,  # As specified in Table 1
            num_layers=2,  # As specified in Table 1
            num_classes=3,  # 3 classes: no footstep, contralateral, ipsilateral
            dropout=0.5  # As specified in Table 1
    ):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer with dropout as specified in Table 1
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Unidirectional LSTM as specified
        )

        # Batch normalization after LSTM as described in the paper
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Fully connected layers as specified in Table 1: hidden_size -> 64 -> 32
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Task-specific output heads for multi-task learning
        self.multiclass_head = nn.Linear(32, num_classes)
        self.contralateral_head = nn.Linear(32, 2)  # Binary classification
        self.ipsilateral_head = nn.Linear(32, 2)  # Binary classification

        # Initialize weights using methods specified in Table 1
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Orthogonal as specified in Table 1."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name and len(param.shape) >= 2:
                    # Orthogonal initialization for LSTM weights (better for RNNs)
                    nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    # Xavier uniform for linear layer weights
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
        """
        batch_size = x.size(0)

        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Use only the last time step output as described in the paper
        out = lstm_out[:, -1, :]

        # Apply batch normalization
        out = self.batch_norm(out)

        # Apply fully connected layers as specified in Table 1
        out = self.relu1(self.fc1(out))
        out = self.dropout1(out)
        out = self.relu2(self.fc2(out))
        out = self.dropout2(out)

        # Apply task-specific heads
        return {
            'multiclass': self.multiclass_head(out),
            'contralateral': self.contralateral_head(out),
            'ipsilateral': self.ipsilateral_head(out)
        }

