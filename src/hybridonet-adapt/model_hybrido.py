import torch
import torch.nn as nn
from typing import Tuple, Optional


class ODEFunc(nn.Module):
    """
    Derivative function dz/dt = f(z, t) parameterized by a neural network.
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, t: float, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class NeuralODEBlock(nn.Module):
    """
    Modular Neural Ordinary Differential Equation (NODE) Block.
    Integrates hidden state trajectory over continuous time [0, 1] using Runge-Kutta 4 (RK4).
    Can be replaced directly with a Koopman Operator block.
    """
    def __init__(self, hidden_dim: int = 64, num_steps: int = 4):
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
        # z shape: (Batch, Seq_Len, Hidden_Dim) or (Batch, Hidden_Dim)
        dt = 1.0 / self.num_steps
        t = 0.0
        for _ in range(self.num_steps):
            z = self._rk4_step(self.ode_func, t, z, dt)
            t += dt
        return z


class FeatureExtractor(nn.Module):
    """
    HybridoNet Temporal Feature Extractor:
    1. 2-layer LSTM (input_dim=18 -> hidden_dim=64)
    2. Multihead Attention (embed_dim=64)
    3. Neural ODE (NODE) block (modular & swappable)
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

        # 3. Modular Neural ODE Block (or custom operator)
        self.node = ode_block if ode_block is not None else NeuralODEBlock(hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: (Batch, Seq_Len=10, Channels=3, Features=6) -> flattened to (Batch, 10, 18)
        Output: (Batch, hidden_dim=64)
        """
        if x.dim() == 4:
            b, s, c, f = x.shape
            x = x.view(b, s, c * f)

        # LSTM temporal feature encoding
        lstm_out, _ = self.lstm(x)  # (B, S, 64)

        # Multihead Attention with residual & LayerNorm
        attn_out, _ = self.mha(lstm_out, lstm_out, lstm_out)  # (B, S, 64)
        h = self.layer_norm(lstm_out + attn_out)

        # Temporal aggregation (mean pooling over sequence steps)
        h_pooled = h.mean(dim=1)  # (B, 64)

        # Continuous state evolution via Neural ODE
        z = self.node(h_pooled)  # (B, 64)
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
    
    Structure:
    - Feature Extractor (LSTM + MHA + NODE) -> feature embedding z (64-D)
    - Source Predictor P^S -> y_hat_s
    - Target Predictor P^T -> y_hat_t
    - Combined Target Prediction: Y_hat_T = theta_s * y_hat_s + theta_t * y_hat_t
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
        # Shared Feature Extractor
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_lstm_layers=num_lstm_layers,
            num_heads=num_heads,
            dropout=dropout,
            ode_block=ode_block
        )

        # Dual Predictors
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
            z: Latent feature embedding (for MMD loss computation)
        """
        z = self.extract_features(x)
        y_hat_s = self.source_predictor(z)
        y_hat_t = self.target_predictor(z)
        
        # Direct weighted sum without denominator as defined in paper
        y_hat_comb = self.theta_s * y_hat_s + self.theta_t * y_hat_t
        
        return y_hat_comb, y_hat_s, y_hat_t, z
