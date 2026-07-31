#!/usr/bin/env python3
"""
Physics-Informed Koopman Neural Operator (KNO) with Self-Attention & Domain Adversarial Adaptation.
Implements:
  1. KoopmanEncoder: Maps non-linear dQ/d(SOC) curves into a linear invariant latent subspace.
  2. KoopmanOperatorLayer: Learns transition matrix K in R^(D x D) enforcing z_{k+1} = K z_k
     and computes Physics-Informed Koopman Linearity Loss L_KNO.
  3. SelfAttentionModule: Multi-head temporal attention across Koopman trajectory embeddings.
  4. GradientReversalLayer (GRL) & DomainDiscriminator: Explicit Domain-Adversarial Neural Network (DANN)
     adaptation head to align Stanford LFP (Source) and Oxford LCO / CALCE NMC (Target) latent distributions.
  5. RULRegressionHead: Predicts log10(Cycle Life) from Koopman-attended representations.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) for explicit Domain Adversarial Adaptation (DANN).
    Forward: identity transformation.
    Backward: multiplies incoming gradients by -alpha.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, alpha)


class KoopmanEncoder(nn.Module):
    """
    Maps 1D SOC curves (L=200) into a D-dimensional Koopman latent embedding space.
    """
    def __init__(self, in_features: int = 200, d_model: int = 64):
        super(KoopmanEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_cycles, in_features)
        return self.encoder(x)


class KoopmanOperatorLayer(nn.Module):
    """
    Learns linear Koopman operator matrix K in R^(D x D) such that z_{k+1} = K z_k.
    Calculates the Physics-Informed Koopman Linearity Loss.
    """
    def __init__(self, d_model: int = 64):
        super(KoopmanOperatorLayer, self).__init__()
        # Learnable transition matrix initialized close to identity
        self.K = nn.Parameter(torch.eye(d_model) + torch.randn(d_model, d_model) * 0.01)

    def forward(self, z_seq: torch.Tensor):
        """
        z_seq shape: (batch_size, num_cycles, d_model)
        returns    : koompan_loss scalar, evolved_z tensor
        """
        batch_size, num_cycles, d_model = z_seq.size()
        if num_cycles < 2:
            return torch.tensor(0.0, device=z_seq.device), z_seq

        z_current = z_seq[:, :-1, :]  # (batch_size, T-1, D)
        z_next_true = z_seq[:, 1:, :] # (batch_size, T-1, D)

        # Apply Koopman transition matrix K: z_{k+1}_pred = z_k K^T
        z_next_pred = torch.matmul(z_current, self.K.t())

        # Physics-Informed Koopman Linearity Loss
        koopman_loss = torch.mean((z_next_true - z_next_pred) ** 2)

        return koopman_loss, z_next_pred


class SelfAttentionModule(nn.Module):
    """
    Multi-Head Temporal Self-Attention across Koopman trajectory embeddings.
    """
    def __init__(self, d_model: int = 64, nhead: int = 4, dropout: float = 0.1):
        super(SelfAttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq shape: (batch_size, num_cycles, d_model)
        attn_out, _ = self.attention(z_seq, z_seq, z_seq)
        out = self.norm(z_seq + self.dropout(attn_out))
        return out


class DomainDiscriminator(nn.Module):
    """
    Domain adversarial classification head predicting source (0) vs. target (1) domain.
    """
    def __init__(self, d_model: int = 64):
        super(DomainDiscriminator, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

    def forward(self, z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        # Apply Gradient Reversal Layer (GRL)
        z_rev = grad_reverse(z, alpha=alpha)
        logits = self.classifier(z_rev)
        return logits


class BatteryKoopmanDANN(nn.Module):
    """
    Full Physics-Informed Koopman Neural Operator + Self-Attention + DANN Model.
    """
    def __init__(
        self,
        in_features: int = 200,    # SOC grid points
        num_cycles: int = 46,      # Early cycles
        d_model: int = 64,
        nhead: int = 4,
        dropout: float = 0.1
    ):
        super(BatteryKoopmanDANN, self).__init__()
        self.num_cycles = num_cycles
        self.d_model = d_model

        self.encoder = KoopmanEncoder(in_features=in_features, d_model=d_model)
        self.koopman_layer = KoopmanOperatorLayer(d_model=d_model)
        self.attention = SelfAttentionModule(d_model=d_model, nhead=nhead, dropout=dropout)

        # RUL Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        # Domain Discriminator (DANN)
        self.domain_discriminator = DomainDiscriminator(d_model=d_model)

    def extract_features(self, x: torch.Tensor):
        """
        Extracts Koopman-attended global embedding (pooled across cycles).
        """
        z_seq = self.encoder(x)                            # (batch_size, num_cycles, d_model)
        koopman_loss, _ = self.koopman_layer(z_seq)        # Scalar loss
        z_attn = self.attention(z_seq)                     # (batch_size, num_cycles, d_model)
        z_global = torch.mean(z_attn, dim=1)               # (batch_size, d_model)
        return z_global, koopman_loss

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        """
        Input x shape : (batch_size, num_cycles=46, in_features=200)
        Returns:
          1. log_eol_pred  : predicted log10(Cycle Life) shape (batch_size, 1)
          2. domain_logits : DANN domain classification logits shape (batch_size, 2)
          3. koopman_loss  : scalar Physics-Informed linearity loss
        """
        z_global, koopman_loss = self.extract_features(x)

        # 1. RUL Prediction
        log_eol_pred = self.regression_head(z_global)

        # 2. Domain Discriminator (via GRL)
        domain_logits = self.domain_discriminator(z_global, alpha=alpha)

        return log_eol_pred, domain_logits, koopman_loss


if __name__ == "__main__":
    # Test Koopman DANN shape consistency
    model = BatteryKoopmanDANN(in_features=200, num_cycles=46, d_model=64)
    dummy_input = torch.randn(8, 46, 200)
    pred_log, dom_logits, kno_loss = model(dummy_input, alpha=0.5)
    print(f"Koopman DANN test forward successful!")
    print(f"  RUL Pred shape   : {pred_log.shape}")
    print(f"  Domain Logits    : {dom_logits.shape}")
    print(f"  KNO Linearity Loss: {kno_loss.item():.4f}")
