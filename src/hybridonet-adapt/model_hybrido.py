import torch
import torch.nn as nn
from typing import Tuple, Optional


class ODEFunc(nn.Module):
    """
    Derivative function dz/dt = f(z, t).
    Faithful to paper: f is realized as a single linear layer: dz/dt = W*z + b.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


class NeuralODEBlock(nn.Module):
    """
    Modular Neural Ordinary Differential Equation (NODE) Block.
    Integrates hidden state trajectory over continuous time using Runge-Kutta 4 (RK4).
    Evaluates at continuous integration step (t=2 / 2 integration steps).
    """
    def __init__(self, hidden_dim: int = 64, num_steps: int = 2):
        super().__init__()
        self.ode_func = ODEFunc(hidden_dim)
        self.num_steps = num_steps

    def _rk4_step(self, f: nn.Module, t: float, z: torch.Tensor, dt: float) -> torch.Tensor:
        k1 = f(t, z)
        k2 = f(t + dt / 2.0, z + dt / 2.0 * k1)
        k3 = f(t + dt / 2.0, z + dt / 2.0 * k2)
        k4 = f(t + dt, z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        dt = 1.0 / max(1, self.num_steps)
        t = 0.0
        for _ in range(self.num_steps):
            z = self._rk4_step(self.ode_func, t, z, dt)
            t += dt
        return z


class FeatureExtractor(nn.Module):
    """
    HybridoNet-Adapt Feature Extractor (Tran et al., 2025):
    1. 2-layer LSTM (input_dim=18 -> hidden_dim=64)
    2. Multihead Attention (embed_dim=64)
    3. Second-to-last attention timestep selection (h_{t=-2})
    4. Neural ODE (NODE) continuous dynamics block
    """
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        ode_block: Optional[nn.Module] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Two-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0
        )

        # 2. Multihead Attention (Scaled Dot-Product)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 3. Modular Neural ODE Block
        self.node = ode_block if ode_block is not None else NeuralODEBlock(hidden_dim=hidden_dim, num_steps=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (Batch, Seq_Len=10, Channels=3, Features=6) -> (Batch, 10, 18)
        Output: Latent state z (Batch, 64)
        """
        if x.dim() == 4:
            b, s, c, f = x.shape
            x = x.view(b, s, c * f)

        # LSTM temporal feature encoding -> (B, S=10, 64)
        lstm_out, _ = self.lstm(x)

        # Multihead Attention with residual connection & LayerNorm
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)
        h = self.layer_norm(lstm_out + attn_out)  # (B, S=10, 64)

        # Paper-faithful: Select second-to-last attention timestep (-2)
        h_selected = h[:, -2, :]  # (B, 64)

        # Continuous state evolution via Neural ODE
        z = self.node(h_selected)  # (B, 64)
        return z


class Predictor(nn.Module):
    """
    RUL Predictor Network:
    Three linear layers [128, 64, 32, 1] with BatchNorm1d, Dropout(0.1), ReLU,
    ending strictly with Sigmoid() activation.
    """
    def __init__(self, in_features: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class HybridoNetAdapt(nn.Module):
    """
    HybridoNet-Adapt (Tran et al., 2025)
    Complete Domain Adaptation Architecture for Battery RUL.
    
    Target prediction formula:
        Y_hat_T = theta_S * G_Y^S(G_F(X)) + theta_T * G_Y^T(G_F(X))
    """
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        ode_block: Optional[nn.Module] = None
    ):
        super().__init__()
        # Shared Feature Extractor G_F
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
            num_heads=num_heads,
            dropout=dropout,
            ode_block=ode_block
        )

        # Dual Predictors G_Y^S and G_Y^T
        self.source_predictor = Predictor(in_features=hidden_dim, dropout=dropout)
        self.target_predictor = Predictor(in_features=hidden_dim, dropout=dropout)

        # Trainable Trade-Off Parameters
        self.theta_s = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.theta_t = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 64-dimensional feature embeddings z."""
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            y_hat_comb: Combined target prediction = theta_s * y_hat_s + theta_t * y_hat_t
            y_hat_s: Source predictor output
            y_hat_t: Target predictor output
            z: Latent feature embedding
        """
        z = self.extract_features(x)
        y_hat_s = self.source_predictor(z)
        y_hat_t = self.target_predictor(z)
        
        # Direct sum weighting as specified in Eq. (11)
        y_hat_comb = self.theta_s * y_hat_s + self.theta_t * y_hat_t
        
        return y_hat_comb, y_hat_s, y_hat_t, z
