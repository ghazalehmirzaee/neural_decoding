# src/models/hybrid_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedAttention(nn.Module):
    """
    Multi-head self-attention mechanism as described in the paper.
    Uses query-key-value formulation with scaling factor.
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(EnhancedAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.scaling = float(self.head_dim) ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass implementing equation (16)-(18) from the paper.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)

        Returns:
            Attention output of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project inputs to queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention calculation
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention scores with scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling

        # Apply softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, num_heads, seq_len, head_dim)

        # Reshape and project back to embed_dim
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        attn_output = self.out_proj(attn_output)

        return attn_output


class DynamicNormalization(nn.Module):
    """
    Implements the hierarchical normalization strategy described in the paper.
    Combines batch, layer, and group normalization.
    """

    def __init__(self, num_features, groups=8):
        super(DynamicNormalization, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.group_norm = nn.GroupNorm(groups, num_features)

    def forward(self, x):
        """
        Forward pass implementing equations (3)-(5) from the paper.

        Args:
            x: Input tensor (batch_size, channels, seq_len) for CNN
               or (batch_size, seq_len, channels) for BiLSTM

        Returns:
            Normalized tensor with the same shape as input
        """
        if len(x.shape) == 3 and x.shape[1] != self.batch_norm.num_features:
            # Input is (batch, seq_len, channels) - transpose for BatchNorm1d
            x_t = x.transpose(1, 2)
            x_bn = self.batch_norm(x_t).transpose(1, 2)

            # For GroupNorm, reshape then apply
            batch_size, seq_len, channels = x.shape
            x_reshaped = x.reshape(-1, channels)
            x_gn = self.group_norm(x_reshaped.unsqueeze(2)).squeeze(2)
            x_gn = x_gn.reshape(batch_size, seq_len, channels)

            # Average the normalizations
            return (x_bn + x_gn) / 2
        else:
            # Input is (batch, channels, seq_len) - direct application
            x_bn = self.batch_norm(x)
            x_gn = self.group_norm(x)

            # Average the normalizations
            return (x_bn + x_gn) / 2


class HybridCNNBiLSTM(nn.Module):
    """
    Hybrid CNN-BiLSTM model with attention mechanisms for neural decoding.

    This model implements the architecture described in the paper:
    - CNN layers for spatial feature extraction
    - Skip connections with 1×1 convolution
    - BiLSTM for temporal processing
    - Multi-head self-attention
    - Multiple output heads for different tasks
    """

    def __init__(
            self,
            input_size,
            hidden_size=128,
            num_layers=2,
            num_classes=3,
            dropout=0.5
    ):
        super(HybridCNNBiLSTM, self).__init__()

        # CNN feature extractor as described in equation (6)
        self.cnn_layers = nn.ModuleList([
            # First CNN layer: input_size -> 64
            nn.Sequential(
                nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
                DynamicNormalization(64),
                nn.ReLU(),
                nn.Dropout(dropout / 2)
            ),
            # Second CNN layer: 64 -> 128
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                DynamicNormalization(128),
                nn.ReLU(),
                nn.Dropout(dropout / 2)
            ),
            # Third CNN layer: 128 -> 256
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                DynamicNormalization(256),
                nn.ReLU()
            )
        ])

        # Skip connection as described in equation (7)
        self.skip_connection = nn.Conv1d(input_size, 256, kernel_size=1)

        # Group normalization for combined output as in equation (8)
        self.group_norm = nn.GroupNorm(8, 256)

        # BiLSTM as described in equation (9)
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism as described in equations (16)-(18)
        self.attention = EnhancedAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout / 2
        )

        # Task-specific heads
        # 1. Multiclass classification head
        self.multiclass_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

        # 2. Binary classification head for contralateral footstep
        self.contralateral_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

        # 3. Binary classification head for ipsilateral footstep
        self.ipsilateral_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

        # 4. Neural activity prediction head (for regularization)
        self.neural_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for CNN and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward pass through the hybrid model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Dictionary containing outputs for all tasks
        """
        # Input shape: (batch, seq_len, input_size)
        batch_size, seq_len, input_size = x.shape

        # Transform for CNN layers: (batch, input_size, seq_len)
        x_cnn = x.transpose(1, 2)

        # CNN feature extraction (equation 6)
        h1 = self.cnn_layers[0](x_cnn)
        h2 = self.cnn_layers[1](h1)
        h3 = self.cnn_layers[2](h2)

        # Skip connection (equation 7)
        h_skip = self.skip_connection(x_cnn)

        # Combine with group normalization (equation 8)
        h_combined = self.group_norm(h3 + h_skip)

        # Prepare for BiLSTM: (batch, seq_len, channels)
        lstm_in = h_combined.transpose(1, 2)

        # Apply BiLSTM (equations 9-15)
        lstm_out, _ = self.bilstm(lstm_in)

        # Apply attention (equations 16-18)
        attn_out = self.attention(lstm_out)

        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)

        # Apply task-specific heads
        return {
            'multiclass': self.multiclass_head(pooled),
            'contralateral': self.contralateral_head(pooled),
            'ipsilateral': self.ipsilateral_head(pooled),
            'neural_activity': self.neural_head(pooled)
        }

