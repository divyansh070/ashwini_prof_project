#!/usr/bin/env python3
"""
Physics-Informed Koopman Neural Operator & Domain-Adversarial Architecture (src/koopman/koopman_model.py).
Implements:
  1. KoopmanEncoder: Maps 2D SOC-normalized dQ/d(SOC) curves (num_cycles=46, L=200) into a linear-latent space Z.
  2. KoopmanOperatorLayer: Learns transition matrix K in R^{D x D} such that z_{k+1} = K z_k,
     penalized by a Physics-Informed Linearity Loss (L_KNO) and a Thermodynamic Monotonicity Loss (L_mono)
     to prevent unphysical capacity rebound across cycle progression.
  3. DomainDiscriminator: A Gradient Reversal Layer (GRL) classifier for Domain-Adversarial
     Neural Network (DANN) transfer learning across battery chemistries (LFP, LCO, NMC).
  4. Multi-Task Prediction Head: Simultaneously estimates Remaining Useful Life (EOL) and
     Knee Onset Cycle (C_knee).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) from Ganin et al. (2016).
    Forward pass is an identity mapping; backward pass scales and reverses the gradient by -alpha.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class KoopmanEncoder(nn.Module):
    """
    1D-CNN + Self-Attention feature extractor mapping 200-pt SOC-normalized dQ/d(SOC)
    vectors across early cycles into a D-dimensional Koopman latent embedding.
    """
    def __init__(self, in_features: int = 200, d_model: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, d_model, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(d_model)
        
        self.proj = nn.Linear(d_model * (in_features // 4), d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_cycles, in_features)
        B, T, L = x.shape
        x_flat = x.view(B * T, 1, L)
        
        h = F.relu(self.bn1(self.conv1(x_flat)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = h.view(B * T, -1)
        z = F.gelu(self.proj(h))
        z = self.layer_norm(z).view(B, T, -1)
        
        # Self-attention over temporal cycle dimension
        z_attn, _ = self.attn(z, z, z)
        return z_attn


class KoopmanOperatorLayer(nn.Module):
    """
    Learns an invariant linear transition matrix K in R^{D x D} such that:
       z_{k+1} = K z_k
    Computes two physics-informed regularizers:
       1. Linearity Loss (L_KNO): || z_{k+1} - K z_k ||_2^2
       2. Monotonicity Loss (L_mono): Penalizes positive increments in latent trajectory norm
          to enforce thermodynamic non-rebounding capacity fade.
    """
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.K = nn.Parameter(torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model))

    def forward(self, z_seq: torch.Tensor):
        # z_seq shape: (B, T, D)
        B, T, D = z_seq.shape
        
        z_curr = z_seq[:, :-1, :]  # (B, T-1, D)
        z_next = z_seq[:, 1:, :]   # (B, T-1, D)
        
        # Linear Koopman evolution prediction: z_{k+1}^{pred} = z_k K^T
        z_pred_next = torch.matmul(z_curr, self.K.t())
        
        # 1. Koopman Linearity Loss (L_KNO)
        kno_loss = F.mse_loss(z_pred_next, z_next)
        
        # 2. Thermodynamic Monotonicity Loss (L_mono)
        # In an irreversible degradation process, latent state magnitude should monotonically decay
        # Any unphysical positive jump in ||z_{k+1}|| - ||z_k|| is penalized
        norm_curr = torch.norm(z_curr, p=2, dim=-1)
        norm_next = torch.norm(z_next, p=2, dim=-1)
        mono_violation = F.relu(norm_next - norm_curr)
        mono_loss = torch.mean(mono_violation ** 2)
        
        return kno_loss, mono_loss


class DomainDiscriminator(nn.Module):
    """
    Gradient Reversal Layer (GRL) Domain Discriminator for DANN adversarial training.
    """
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # Binary classification: 0=Source (LFP), 1=Target (LCO/NMC)
        )

    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        feat_rev = GradientReversalFunction.apply(x, alpha)
        return self.net(feat_rev)


class BatteryKoopmanDANN(nn.Module):
    """
    Complete Multi-Task Physics-Informed Koopman Neural Operator & DANN Architecture.
    Simultaneously outputs:
      1. pred_log_eol: Log10 predicted End-of-Life (remaining cycle life).
      2. pred_log_knee: Log10 predicted Knee Onset Cycle (C_knee).
      3. domain_logits: Source vs. Target domain discriminator logits.
      4. kno_loss: Koopman linearity regularizer.
      5. mono_loss: Thermodynamic monotonicity regularizer.
    """
    def __init__(self, in_features: int = 200, num_cycles: int = 46, d_model: int = 64):
        super().__init__()
        self.encoder = KoopmanEncoder(in_features=in_features, d_model=d_model)
        self.koopman = KoopmanOperatorLayer(d_model=d_model)
        self.domain_classifier = DomainDiscriminator(d_model=d_model)
        
        # Shared pooling and feature bottleneck
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.shared_fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.15)
        )
        
        # Primary Task Head: Log10 EOL prediction
        self.fc_eol = nn.Linear(64, 1)
        
        # Auxiliary Task Head: Log10 Knee Onset Cycle (C_knee) prediction
        self.fc_knee = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, alpha: float = 0.0):
        # x: (B, T, L)
        z_seq = self.encoder(x)  # (B, T, D)
        
        # Compute Koopman Physics-Informed losses
        kno_loss, mono_loss = self.koopman(z_seq)
        
        # Temporal pooling over early cycles
        z_pool = z_seq.transpose(1, 2)  # (B, D, T)
        z_global = self.pool(z_pool).squeeze(-1)  # (B, D)
        
        feat = self.shared_fc(z_global)
        
        # Multi-task predictions
        pred_log_eol = self.fc_eol(feat)
        pred_log_knee = self.fc_knee(feat)
        
        # Domain Adversarial Discriminator logits
        domain_logits = self.domain_classifier(z_global, alpha=alpha)
        
        return pred_log_eol, pred_log_knee, domain_logits, kno_loss, mono_loss
