# src/models/hybrid_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism as described in equations (16)-(18) of the paper.
    """

    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

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
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.zeros_(self.W_k.bias)
        nn.init.zeros_(self.W_v.bias)
        nn.init.zeros_(self.W_o.bias)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        # Linear projections
        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention as in equation (17)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)

        # Concat heads and apply output projection as in equation (18)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.W_o(attn_output)

        return output


class DynamicNormalization(nn.Module):
    """
    Hierarchical normalization strategy as described in equations (3)-(5).
    """

    def __init__(self, num_features, groups=8):
        super(DynamicNormalization, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.group_norm = nn.GroupNorm(groups, num_features)
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, x):
        # Handle different input formats
        if len(x.shape) == 3 and x.shape[1] != self.batch_norm.num_features:
            # Input is (batch, seq_len, channels)
            x_t = x.transpose(1, 2)
            x_bn = self.batch_norm(x_t).transpose(1, 2)

            batch_size, seq_len, channels = x.shape
            x_reshaped = x.reshape(-1, channels)
            x_gn = self.group_norm(x_reshaped.unsqueeze(2)).squeeze(2)
            x_gn = x_gn.reshape(batch_size, seq_len, channels)

            x_ln = self.layer_norm(x)

            return (x_bn + x_gn + x_ln) / 3
        else:
            # Input is (batch, channels, seq_len)
            x_bn = self.batch_norm(x)
            x_gn = self.group_norm(x)

            x_t = x.transpose(1, 2)
            x_ln = self.layer_norm(x_t).transpose(1, 2)

            return (x_bn + x_gn + x_ln) / 3


class HybridCNNBiLSTM(nn.Module):
    """
    Hybrid CNN-BiLSTM model as described in the paper with attention mechanism.
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

        # CNN layers for spatial feature extraction as in equation (6)
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

        # Skip connection using 1×1 convolution as in equation (7)
        self.skip_connection = nn.Conv1d(input_size, 256, kernel_size=1)

        # Group normalization for skip connection outputs as in equation (8)
        self.group_norm = nn.GroupNorm(8, 256)

        # BiLSTM for temporal processing as in equation (9)
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-head attention as in equations (16)-(18)
        self.attention = MultiHeadAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8
        )

        # Task-specific heads with layer normalization (5)
        # 1. Multiclass classification head
        self.multiclass_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

        # 2. Binary classification for contralateral footstep
        self.contralateral_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

        # 3. Binary classification for ipsilateral footstep
        self.ipsilateral_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

        # 4. Neural activity prediction head for regularization
        self.neural_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
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
        x_cnn = x.transpose(1, 2)

        # CNN feature extraction with hierarchy as in equation (6)
        h1 = self.cnn_layers[0](x_cnn)
        h2 = self.cnn_layers[1](h1)
        h3 = self.cnn_layers[2](h2)

        # Skip connection as in equation (7)
        h_skip = self.skip_connection(x_cnn)

        # Combine with group normalization as in equation (8)
        h_combined = self.group_norm(h3 + h_skip)

        # Reshape for BiLSTM: [batch, channels, seq_len] -> [batch, seq_len, channels]
        lstm_in = h_combined.transpose(1, 2)

        # Apply BiLSTM as in equation (9)
        lstm_out, _ = self.bilstm(lstm_in)

        # Apply multi-head attention
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

