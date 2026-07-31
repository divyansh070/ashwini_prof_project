#!/usr/bin/env python3
"""
Patch Time Series Transformer (PatchTST) Architecture for Battery SOH/RUL Estimation.
Implements:
  1. PatchEmbedding: Linearly projects segmented dQ/du patches into d_model token space
     with learnable positional encodings (Nie et al., ICLR 2023).
  2. Multi-Head Transformer Encoder: 4-layer Self-Attention blocks with RevIN / LayerNorm.
  3. Regression Head: Aggregates token embeddings across cycles and patches to predict log10(Cycle Life).
  4. Encoder Freezing & Unfreezing methods for Multi-Dataset Transfer Learning.
"""

import torch
import torch.nn as nn
import numpy as np


class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series distribution shift robustness.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super(RevIN, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine_weight = nn.Parameter(torch.ones(1, num_channels, 1))
        self.affine_bias = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        # x shape: (batch_size, num_channels, seq_len)
        if mode == "norm":
            mean = torch.mean(x, dim=2, keepdim=True).detach()
            std = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - mean) / std
            x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == "denorm":
            return x


class PatchEmbedding(nn.Module):
    """
    Linear projection of time series patches into d_model token space + Positional Encoding.
    """
    def __init__(self, patch_len: int = 16, stride: int = 8, d_model: int = 64, num_patches: int = 24):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_patches = num_patches
        
        self.projection = nn.Linear(patch_len, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x_patches: torch.Tensor) -> torch.Tensor:
        """
        x_patches shape: (batch_size * num_channels, num_patches, patch_len)
        returns shape  : (batch_size * num_channels, num_patches, d_model)
        """
        tokens = self.projection(x_patches) + self.positional_encoding
        return self.dropout(tokens)


class BatteryPatchTST(nn.Module):
    """
    Full PatchTST architecture with RUL Regression Head and Transfer Learning encoder freeze controls.
    """
    def __init__(
        self,
        num_channels: int = 46,     # Early cycles 10 through 100
        seq_len: int = 200,         # SOD normalized grid points
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super(BatteryPatchTST, self).__init__()
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1
        self.d_model = d_model

        # Reversible Instance Normalization
        self.revin = RevIN(num_channels)

        # Patch Tokenizer
        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            num_patches=self.num_patches
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # Multi-Channel Token Aggregation -> RUL Regression Head
        self.flatten_dim = num_channels * d_model
        self.regression_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape : (batch_size, num_channels=46, seq_len=200)
        Output shape  : (batch_size, 1) -> log10(Cycle Life)
        """
        batch_size = x.size(0)

        # 1. Instance Normalization across channels
        x_norm = self.revin(x, mode="norm")

        # 2. Extract overlapping patches: shape (batch_size, num_channels, num_patches, patch_len)
        x_patches = x_norm.unfold(dimension=2, size=self.patch_len, step=self.stride)

        # 3. Reshape for token projection: (batch_size * num_channels, num_patches, patch_len)
        x_patches_flat = x_patches.reshape(batch_size * self.num_channels, self.num_patches, self.patch_len)

        # 4. Embed patches to tokens: (batch_size * num_channels, num_patches, d_model)
        tokens = self.patch_embedding(x_patches_flat)

        # 5. Transformer Encoder Attention: (batch_size * num_channels, num_patches, d_model)
        encoded_tokens = self.transformer_encoder(tokens)

        # 6. Mean pooling across patch tokens: (batch_size * num_channels, d_model)
        channel_embeddings = torch.mean(encoded_tokens, dim=1)

        # 7. Reshape to aggregate channels: (batch_size, num_channels * d_model)
        aggregated = channel_embeddings.reshape(batch_size, self.num_channels * self.d_model)

        # 8. Regression Head -> predict log10(Cycle Life)
        log_eol_pred = self.regression_head(aggregated)
        return log_eol_pred

    def freeze_encoder(self):
        """
        Freezes the PatchEmbedding and TransformerEncoder layers for multi-dataset transfer learning.
        Only the Regression Head remains trainable.
        """
        for param in self.revin.parameters():
            param.requires_grad = False
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
        for param in self.regression_head.parameters():
            param.requires_grad = True

    def unfreeze_encoder(self):
        """
        Unfreezes all encoder layers for end-to-end training.
        """
        for param in self.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test model shape consistency
    model = BatteryPatchTST(num_channels=46, seq_len=200, d_model=64, nhead=4, num_layers=4)
    dummy_input = torch.randn(8, 46, 200)
    out = model(dummy_input)
    print(f"PatchTST test forward successful! Output shape: {out.shape}")
    model.freeze_encoder()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen encoder trainable head params: {trainable_params:,}")
