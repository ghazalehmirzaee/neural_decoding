# src/models/lstm_attention_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head temporal attention mechanism designed specifically for sequential neural data.
    """

    def __init__(self, hidden_size, num_heads=4, attention_dim=48):
        super(MultiHeadTemporalAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.head_dim = attention_dim // num_heads

        # Ensure attention_dim is divisible by num_heads
        assert attention_dim % num_heads == 0, "attention_dim must be divisible by num_heads"

        # Multi-head attention components
        # Using smaller dimensions than the full hybrid model for computational efficiency
        self.query = nn.Linear(hidden_size, attention_dim, bias=False)
        self.key = nn.Linear(hidden_size, attention_dim, bias=False)
        self.value = nn.Linear(hidden_size, attention_dim, bias=False)
        self.output_proj = nn.Linear(attention_dim, hidden_size)

        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)  # Light dropout in attention

        # Scaling factor for attention scores
        self.scale = math.sqrt(self.head_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize attention weights using scaled initialization for stability."""
        # Use smaller initialization scale for attention to prevent early saturation
        for module in [self.query, self.key, self.value]:
            nn.init.xavier_uniform_(module.weight, gain=0.6)

        # Output projection initialized normally
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, lstm_output):
        """
        Apply multi-head temporal attention to LSTM output sequence.
        """
        batch_size, seq_len, hidden_size = lstm_output.shape

        # Generate query, key, value projections
        Q = self.query(lstm_output)  # [batch_size, seq_len, attention_dim]
        K = self.key(lstm_output)  # [batch_size, seq_len, attention_dim]
        V = self.value(lstm_output)  # [batch_size, seq_len, attention_dim]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape after transpose: [batch_size, num_heads, seq_len, head_dim]

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        # attended shape: [batch_size, num_heads, seq_len, head_dim]

        # Concatenate heads and project back
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.attention_dim
        )
        attended_output = self.output_proj(attended)

        # Global average pooling across time dimension
        # This aggregates the temporal information while preserving the attended features
        pooled_output = torch.mean(attended_output, dim=1)  # [batch_size, hidden_size]

        # Residual connection with layer normalization
        # Use the last timestep of original LSTM output for residual connection
        residual = lstm_output[:, -1, :]  # [batch_size, hidden_size]
        output = self.layer_norm(pooled_output + residual)

        return output


class LSTMAttentionModel(nn.Module):
    """
    LSTM+Attention model designed to bridge the performance gap between basic LSTM and hybrid models.
    """

    def __init__(
            self,
            input_size,
            hidden_size=96,  # Increased from basic LSTM (64) but less than hybrid (128)
            num_layers=2,
            num_classes=3,
            dropout=0.4,  # Balanced dropout
            num_attention_heads=4,  # Multi-head attention
            attention_dim=48  # Attention dimension
    ):
        super(LSTMAttentionModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # Multi-head temporal attention mechanism
        self.attention = MultiHeadTemporalAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            attention_dim=attention_dim
        )

        # First FC layer with batch normalization
        self.fc1 = nn.Linear(hidden_size, 80)  # Bigger than basic LSTM
        self.bn1 = nn.BatchNorm1d(80)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout * 0.8)  # Slightly less dropout in early layers

        # Second FC layer
        self.fc2 = nn.Linear(80, 56)  # Intermediate layer
        self.bn2 = nn.BatchNorm1d(56)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout * 0.6)  # Progressive dropout reduction

        # Third FC layer for more representation power
        self.fc3 = nn.Linear(56, 40)
        self.bn3 = nn.BatchNorm1d(40)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout * 0.4)  # Minimal dropout in deeper layers

        # Task-specific output heads
        # Using more sophisticated heads than basic LSTM
        self.multiclass_head = nn.Sequential(
            nn.Linear(40, 24),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(24, num_classes)
        )

        self.contralateral_head = nn.Sequential(
            nn.Linear(40, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )

        self.ipsilateral_head = nn.Sequential(
            nn.Linear(40, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 2)
        )

        # Initialize weights properly for stable training
        self._initialize_weights()

    def _initialize_weights(self):
        """Careful weight initialization to ensure stable training and good performance."""
        for name, param in self.named_parameters():
            if 'lstm' in name:
                if 'weight' in name and len(param.shape) >= 2:
                    # Orthogonal initialization for LSTM weights
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1 to help with long-term dependencies
                    if 'bias_hh' in name:
                        n = param.size(0)
                        param.data[n // 4:n // 2].fill_(1.0)
            elif isinstance(param, nn.Linear):
                if len(param.shape) >= 2:
                    # Xavier initialization for linear layers
                    nn.init.xavier_uniform_(param, gain=0.8)
            elif 'bias' in name and 'head' not in name:
                nn.init.zeros_(param)
            elif 'head' in name and 'bias' in name:
                # Small positive bias for output heads to help initial learning
                nn.init.constant_(param, 0.01)

    def forward(self, x):
        """
        Forward pass with attention mechanism and enhanced feature processing.
        """
        batch_size = x.size(0)

        # Initialize LSTM hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))  # [batch_size, seq_len, hidden_size]

        # Apply multi-head temporal attention
        # This is the key enhancement over basic LSTM
        attended_features = self.attention(lstm_out)  # [batch_size, hidden_size]

        # Enhanced fully connected processing
        # More sophisticated feature transformation than basic LSTM
        out = self.relu1(self.bn1(self.fc1(attended_features)))
        out = self.dropout1(out)

        out = self.relu2(self.bn2(self.fc2(out)))
        out = self.dropout2(out)

        out = self.relu3(self.bn3(self.fc3(out)))
        out = self.dropout3(out)

        # Task-specific outputs through enhanced heads
        return {
            'multiclass': self.multiclass_head(out),
            'contralateral': self.contralateral_head(out),
            'ipsilateral': self.ipsilateral_head(out)
        }

