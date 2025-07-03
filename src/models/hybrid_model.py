# src/models/hybrid_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism as described in equations (16)-(18) of the paper.
    Enables focusing on the most behaviorally-relevant neuronal patterns.
    """

    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V as in equation (16)
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection W_O from equation (18)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # Scaling factor as in equation (17)
        self.scaling = self.head_dim ** -0.5

        self._init_weights()

    def _init_weights(self):
        """Initialize attention weights with careful scaling for stability."""
        # Use Xavier initialization as mentioned in the paper
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)
        nn.init.zeros_(self.W_o.bias)

    def forward(self, x):
        """
        Apply multi-head attention as described in equations (16)-(18).
        """
        batch_size, seq_len, embed_dim = x.shape

        # Linear projections - equation (16)
        q = self.W_q(x)  # [batch_size, seq_len, embed_dim]
        k = self.W_k(x)  # [batch_size, seq_len, embed_dim]
        v = self.W_v(x)  # [batch_size, seq_len, embed_dim]

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention - equation (17)
        # (QK^T)/sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        # Apply softmax normalization
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and concat heads - equation (18)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Apply output projection
        output = self.W_o(attn_output)

        return output


class DynamicNormalization(nn.Module):
    """
    Hierarchical normalization strategy as described in equations (3)-(5).
    Combines batch, group, and layer normalization for optimal handling of neuronal data.
    """

    def __init__(self, num_features, groups=8):
        super(DynamicNormalization, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.group_norm = nn.GroupNorm(groups, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

        # Learnable weights for combining different normalizations
        self.weights = nn.Parameter(torch.ones(3) / 3)
        self.num_features = num_features

    def forward(self, x):
        """
        Apply combined normalization strategy.
        """
        # Apply softmax to ensure weights sum to 1
        weights = F.softmax(self.weights, dim=0)

        # Handle different input formats based on dimensions
        if x.dim() == 3:
            if x.size(1) == self.num_features:
                # Input is [batch, channels, seq_len]
                x_bn = self.batch_norm(x)
                x_gn = self.group_norm(x)

                # For layer norm, we need to transpose
                x_t = x.transpose(1, 2)  # [batch, seq_len, channels]
                x_ln = self.layer_norm(x_t).transpose(1, 2)  # back to [batch, channels, seq_len]

                # Weighted combination according to equation (3-5)
                return weights[0] * x_bn + weights[1] * x_gn + weights[2] * x_ln
            else:
                # Input is [batch, seq_len, channels]
                # Transpose for batch and group norm
                x_t = x.transpose(1, 2)  # [batch, channels, seq_len]
                x_bn = self.batch_norm(x_t).transpose(1, 2)
                x_gn = self.group_norm(x_t).transpose(1, 2)

                # Layer norm works directly
                x_ln = self.layer_norm(x)

                # Weighted combination
                return weights[0] * x_bn + weights[1] * x_gn + weights[2] * x_ln
        else:
            # For 2D input [batch, features]
            x_bn = self.batch_norm(x.unsqueeze(-1)).squeeze(-1)
            x_gn = self.group_norm(x.unsqueeze(-1)).squeeze(-1)
            x_ln = self.layer_norm(x)

            # Weighted combination
            return weights[0] * x_bn + weights[1] * x_gn + weights[2] * x_ln


class HybridCNNBiLSTM(nn.Module):
    """
    Hybrid CNN-BiLSTM model as described in the paper with attention mechanism.
    Integrates spatial and temporal processing for improved neural decoding.
    """

    def __init__(
            self,
            input_size,
            hidden_size=128,  # As specified in Table 2
            num_layers=2,  # As specified in Table 2
            num_classes=3,  # 3 classes: no footstep, contralateral, ipsilateral
            dropout=0.5,  # As specified in Table 2
            use_skip_connection=True,
            use_attention=True,
            num_attention_heads=8
    ):
        super(HybridCNNBiLSTM, self).__init__()

        self.use_skip_connection = use_skip_connection
        self.use_attention = use_attention

        # CNN layers for spatial feature extraction as in equation (6)
        # First CNN layer: input_size -> 64
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.norm1 = DynamicNormalization(64)
        self.dropout1 = nn.Dropout(dropout / 2)  # Lower dropout in early layers

        # Second CNN layer: 64 -> 128
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.norm2 = DynamicNormalization(128)
        self.dropout2 = nn.Dropout(dropout / 2)

        # Third CNN layer: 128 -> 256
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.norm3 = DynamicNormalization(256)

        # Skip connection using 1×1 convolution as in equation (7)
        if use_skip_connection:
            self.skip_connection = nn.Conv1d(input_size, 256, kernel_size=1)
            self.group_norm = nn.GroupNorm(8, 256)  # As specified in equation (8)

        # BiLSTM for temporal processing as in equation (9)
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # Bidirectional as specified
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-head attention as in equations (16)-(18)
        if use_attention:
            self.attention = MultiHeadAttention(
                embed_dim=hidden_size * 2,  # *2 for bidirectional
                num_heads=num_attention_heads
            )

        # Common features before task-specific heads
        bilstm_out_dim = hidden_size * 2  # *2 for bidirectional

        # Task-specific heads with layer normalization (5)
        # 1. Multiclass classification head (no footstep, contralateral, ipsilateral)
        self.multiclass_head = nn.Sequential(
            nn.Linear(bilstm_out_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

        # 2. Binary classification for contralateral footstep
        self.contralateral_head = nn.Sequential(
            nn.Linear(bilstm_out_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

        # 3. Binary classification for ipsilateral footstep
        self.ipsilateral_head = nn.Sequential(
            nn.Linear(bilstm_out_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

        # 4. Neural activity prediction head for regularization
        self.neural_head = nn.Sequential(
            nn.Linear(bilstm_out_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights properly as described in the paper."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Kaiming initialization for convolutional layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                # Standard normalization initialization
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Xavier uniform for linear layers
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                # Orthogonal initialization for LSTM weights
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        """
        Forward pass through the hybrid model.

        Args:
            x: Input tensor [batch_size, seq_len, input_size]

        Returns:
            Dictionary of task outputs
        """
        # Reshape for CNN: [batch, seq_len, input_size] -> [batch, input_size, seq_len]
        batch_size, seq_len, input_size = x.shape
        x_cnn = x.transpose(1, 2)  # Now [batch, input_size, seq_len]

        # CNN feature extraction with hierarchy as in equation (6)
        # First CNN layer
        h1 = self.conv1(x_cnn)
        h1 = self.norm1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout1(h1)

        # Second CNN layer
        h2 = self.conv2(h1)
        h2 = self.norm2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout2(h2)

        # Third CNN layer
        h3 = self.conv3(h2)
        h3 = self.norm3(h3)
        h3 = F.relu(h3)

        # Skip connection as in equation (7)
        if self.use_skip_connection:
            h_skip = self.skip_connection(x_cnn)
            # Combine with group normalization as in equation (8)
            h_combined = self.group_norm(h3 + h_skip)
        else:
            h_combined = h3

        # Reshape for BiLSTM: [batch, channels, seq_len] -> [batch, seq_len, channels]
        lstm_in = h_combined.transpose(1, 2)  # Now [batch, seq_len, 256]

        # Apply BiLSTM as in equation (9)
        lstm_out, _ = self.bilstm(lstm_in)  # [batch, seq_len, 2*hidden_size]

        # Apply multi-head attention if enabled
        if self.use_attention:
            attn_out = self.attention(lstm_out)
        else:
            attn_out = lstm_out

        # Global average pooling for sequence aggregation
        pooled = torch.mean(attn_out, dim=1)  # [batch, 2*hidden_size]

        # Apply task-specific heads
        return {
            'multiclass': self.multiclass_head(pooled),
            'contralateral': self.contralateral_head(pooled),
            'ipsilateral': self.ipsilateral_head(pooled),
            'neural_activity': self.neural_head(pooled)
        }

