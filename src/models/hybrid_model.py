# src/models/hybrid_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedAttention(nn.Module):
    """
    Multi-head self-attention mechanism with improved scaling and dropout.
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.15):
        super(EnhancedAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Query, Key, Value projections with improved initialization
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Improved scaling factor for stable gradients
        self.scaling = float(self.head_dim) ** -0.5

        # Increased dropout for stronger regularization
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        # Improved initialization for attention mechanisms
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        """
        Forward pass implementing equations (16)-(18) from the paper with
        improved numerical stability.
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

        # Apply softmax with improved numerical stability
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply dropout for regularization
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
    Enhanced hierarchical normalization strategy combining batch, layer, and group norms.
    """

    def __init__(self, num_features, groups=8):
        super(DynamicNormalization, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.group_norm = nn.GroupNorm(groups, num_features)
        self.layer_norm = nn.LayerNorm(num_features)  # Added layer norm for better stability

    def forward(self, x):
        """
        Forward pass implementing enhanced normalization with better stability.
        """
        if len(x.shape) == 3 and x.shape[1] != self.batch_norm.num_features:
            # Input is (batch, seq_len, channels) - for BiLSTM
            x_t = x.transpose(1, 2)
            x_bn = self.batch_norm(x_t).transpose(1, 2)

            # For GroupNorm, reshape then apply
            batch_size, seq_len, channels = x.shape
            x_reshaped = x.reshape(-1, channels)
            x_gn = self.group_norm(x_reshaped.unsqueeze(2)).squeeze(2)
            x_gn = x_gn.reshape(batch_size, seq_len, channels)

            # Apply layer norm directly
            x_ln = self.layer_norm(x)

            # Combine all normalizations with weighted average
            return (x_bn + x_gn + x_ln) / 3
        else:
            # Input is (batch, channels, seq_len) - for CNN
            x_bn = self.batch_norm(x)
            x_gn = self.group_norm(x)

            # Apply layer norm with reshape
            x_t = x.transpose(1, 2)  # (batch, seq_len, channels)
            x_ln = self.layer_norm(x_t).transpose(1, 2)  # back to (batch, channels, seq_len)

            # Combine all normalizations with weighted average
            return (x_bn + x_gn + x_ln) / 3


class HybridCNNBiLSTM(nn.Module):
    """
    Enhanced Hybrid CNN-BiLSTM model with optimized attention mechanisms and regularization.
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

        # Enhanced CNN feature extractor
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

        # Improved skip connection with gating mechanism
        self.skip_connection = nn.Conv1d(input_size, 256, kernel_size=1)
        self.skip_gate = nn.Sequential(
            nn.Conv1d(input_size, 256, kernel_size=1),
            nn.Sigmoid()
        )

        # Enhanced group normalization
        self.group_norm = nn.GroupNorm(8, 256)

        # Bidirectional LSTM with improved dropout
        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Enhanced attention mechanism
        self.attention = EnhancedAttention(
            embed_dim=hidden_size * 2,  # *2 for bidirectional
            num_heads=8,
            dropout=dropout / 2
        )

        # Task-specific heads with layer normalization
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
        """Enhanced weight initialization for better convergence."""
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
        Forward pass through the enhanced hybrid model with improved feature extraction.
        """
        # Input shape: (batch, seq_len, input_size)
        batch_size, seq_len, input_size = x.shape

        # Transform for CNN layers: (batch, input_size, seq_len)
        x_cnn = x.transpose(1, 2)

        # CNN feature extraction with residual connections
        h1 = self.cnn_layers[0](x_cnn)
        h2 = self.cnn_layers[1](h1)
        h3 = self.cnn_layers[2](h2)

        # Enhanced skip connection with gating
        h_skip = self.skip_connection(x_cnn)
        skip_gate = self.skip_gate(x_cnn)
        h_skip = h_skip * skip_gate  # Apply gate

        # Combine with group normalization
        h_combined = self.group_norm(h3 + h_skip)

        # Prepare for BiLSTM: (batch, seq_len, channels)
        lstm_in = h_combined.transpose(1, 2)

        # Apply BiLSTM with both hidden states and cell states
        lstm_out, (hidden, cell) = self.bilstm(lstm_in)

        # Apply enhanced attention
        attn_out = self.attention(lstm_out)

        # Global average pooling with residual connection
        pooled = torch.mean(attn_out, dim=1)

        # Apply task-specific heads
        return {
            'multiclass': self.multiclass_head(pooled),
            'contralateral': self.contralateral_head(pooled),
            'ipsilateral': self.ipsilateral_head(pooled),
            'neural_activity': self.neural_head(pooled)
        }

