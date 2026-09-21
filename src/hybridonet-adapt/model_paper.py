#!/usr/bin/env python3
"""
HybridoNet-Adapt — paper-faithful model (Tran et al., 2025, arXiv:2503.21392v2).

Every default below follows the paper text. Where the paper does not specify a
detail, the choice is exposed as an argument and marked [UNSPECIFIED] so it can
be varied and reported.

What the paper specifies (Sections 3.2, 4.1, 4.6):
  * Feature extractor G_F = LSTM -> Multihead Attention -> NODE
  * LSTM: 2 layers, hidden 64; Multihead Attention hidden 64
  * Attention output taken at the SECOND-TO-LAST time step (Fig. 8)
  * NODE: dh/dt = f(h,t,theta), f = a single linear layer, t0 = 0, t1 = 1,
    "NODE output time step of 2" (Fig. 8)
  * Predictors G_Y^S, G_Y^T share an architecture: three linear layers with
    dims [128, 64, 32, 1]; each hidden linear layer followed by ReLU,
    1D BatchNorm, Dropout(0.1); sigmoid at the end
  * Eq. (1): Y_T = theta_S * G_Y^S(G_F(X)) + theta_T * G_Y^T(G_F(X))
  * Eq. (2): Y_S = G_Y^S(G_F(X))           <- source head ALONE
  * theta_S, theta_T are "learnable trade-off parameters" (no constraint stated)

Reconstruction note on the NODE output (INFERENCE, see node_output):
  The extractor hidden size is 64, yet the predictors start at 128
  ("three linear layers ... [128, 64, 32, 1]"). The paper also says the NODE
  output uses 2 discrete time points. Evaluating the ODE trajectory at the two
  points t = {0, 1} and concatenating them gives exactly 2 x 64 = 128, which
  reconciles both statements. That is the default here. 'last' (use only h(t1),
  64-D) is available if you want to test the alternative reading.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ODEFunc(nn.Module):
    """f(h, t) = W h + b  — 'a single linear layer' (Sec. 3.2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, t: float, h: torch.Tensor) -> torch.Tensor:
        return self.linear(h)


class NeuralODE(nn.Module):
    """
    Integrates dh/dt = f(h) from t0 to t1 and returns the trajectory evaluated at
    `num_points` equally spaced time points (paper: t0=0, t1=1, 2 points).

    Solver [UNSPECIFIED in paper]: fixed-step RK4 with `substeps` steps between
    consecutive output points. With a linear f the exact flow is a matrix
    exponential, and RK4 approximates it closely.
    """

    def __init__(self, dim: int, t0: float = 0.0, t1: float = 1.0,
                 num_points: int = 2, substeps: int = 4):
        super().__init__()
        if num_points < 2:
            raise ValueError("num_points must be >= 2 (paper uses 2)")
        self.func = ODEFunc(dim)
        self.t0, self.t1 = float(t0), float(t1)
        self.num_points = int(num_points)
        self.substeps = int(substeps)

    def _rk4(self, h: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        f = self.func
        k1 = f(t, h)
        k2 = f(t + dt / 2, h + dt / 2 * k1)
        k3 = f(t + dt / 2, h + dt / 2 * k2)
        k4 = f(t + dt, h + dt * k3)
        return h + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, h0: torch.Tensor) -> torch.Tensor:
        """Returns (num_points, B, dim); index 0 is h(t0)."""
        span = (self.t1 - self.t0) / (self.num_points - 1)
        dt = span / self.substeps
        h, t = h0, self.t0
        traj = [h0]
        for _ in range(self.num_points - 1):
            for _ in range(self.substeps):
                h = self._rk4(h, t, dt)
                t += dt
            traj.append(h)
        return torch.stack(traj, dim=0)


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,           # paper: 64
        num_lstm_layers: int = 2,       # paper: 2
        num_heads: int = 4,             # [UNSPECIFIED]
        dropout: float = 0.1,           # paper: 0.1
        attn_timestep: int = -2,        # paper: second-to-last (Fig. 8)
        node_points: int = 2,           # paper: 2 (Fig. 8)
        node_output: str = "concat",    # INFERENCE — see module docstring
        attn_residual_norm: bool = False,  # [UNSPECIFIED] not in paper; off
        post_node_norm: bool = False,      # [UNSPECIFIED] not in paper; off
    ):
        super().__init__()
        if node_output not in ("concat", "last"):
            raise ValueError("node_output must be 'concat' or 'last'")
        self.attn_timestep = attn_timestep
        self.node_output = node_output
        self.attn_residual_norm = attn_residual_norm

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_lstm_layers,
                            batch_first=True,
                            dropout=dropout if num_lstm_layers > 1 else 0.0)
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout,
                                         batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim) if attn_residual_norm else nn.Identity()
        self.node = NeuralODE(hidden_dim, 0.0, 1.0, num_points=node_points)

        self.out_dim = hidden_dim * node_points if node_output == "concat" else hidden_dim
        self.post_norm = nn.LayerNorm(self.out_dim) if post_node_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:                       # (B, 10, 3, 6) -> (B, 10, 18)
            b, s, c, f = x.shape
            x = x.reshape(b, s, c * f)
        h, _ = self.lstm(x)                    # (B, 10, H)
        a, _ = self.mha(h, h, h)               # (B, 10, H)
        if self.attn_residual_norm:
            a = self.attn_norm(h + a)
        h_sel = a[:, self.attn_timestep, :]    # (B, H)
        traj = self.node(h_sel)                # (P, B, H)
        if self.node_output == "concat":
            z = traj.permute(1, 0, 2).reshape(traj.shape[1], -1)   # (B, P*H)
        else:
            z = traj[-1]
        return self.post_norm(z)


class Predictor(nn.Module):
    """
    Three linear layers: in -> 64 -> 32 -> 1 (paper dims [128, 64, 32, 1]).
    Hidden linear layers are each followed by ReLU, BatchNorm1d, Dropout
    ("respectively", Sec. 3.2); sigmoid at the end.
    """

    def __init__(self, in_features: int = 128, dropout: float = 0.1,
                 norm: str = "batchnorm"):
        super().__init__()
        def N(d):
            if norm == "batchnorm":
                return nn.BatchNorm1d(d)
            elif norm == "layernorm":
                return nn.LayerNorm(d)
            return nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(), N(64), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), N(32), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class HybridoNetAdapt(nn.Module):
    def __init__(
        self,
        input_dim: int = 18,
        hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_timestep: int = -2,
        node_points: int = 2,
        node_output: str = "concat",
        attn_residual_norm: bool = False,
        post_node_norm: bool = False,
        predictor_norm: str = "batchnorm",     # paper: 1D BatchNorm
        theta_mode: str = "free",              # paper: unconstrained learnable
        theta_init: float = 0.5,               # [UNSPECIFIED]
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_dim, hidden_dim, num_lstm_layers, num_heads, dropout,
            attn_timestep, node_points, node_output,
            attn_residual_norm, post_node_norm,
        )
        d = self.feature_extractor.out_dim
        self.source_predictor = Predictor(d, dropout, predictor_norm)
        self.target_predictor = Predictor(d, dropout, predictor_norm)

        if theta_mode not in ("free", "softmax"):
            raise ValueError("theta_mode must be 'free' or 'softmax'")
        self.theta_mode = theta_mode
        if theta_mode == "free":
            self.theta_raw = nn.Parameter(torch.tensor([theta_init, 1.0 - theta_init]))
        else:
            self.theta_raw = nn.Parameter(torch.zeros(2))

    @property
    def thetas(self) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.theta_raw, 0) if self.theta_mode == "softmax" else self.theta_raw
        return w[0], w[1]

    @property
    def theta_s(self) -> torch.Tensor:
        return self.thetas[0]

    @property
    def theta_t(self) -> torch.Tensor:
        return self.thetas[1]

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor):
        """
        Returns (y_T, y_S, y_Thead, z):
          y_T     Eq. (1): theta_S * G_Y^S + theta_T * G_Y^T   (target prediction)
          y_S     Eq. (2): G_Y^S alone                          (source prediction)
          y_Thead G_Y^T alone (diagnostics)
          z       latent features used by the MMD loss
        """
        z = self.feature_extractor(x)
        y_s = self.source_predictor(z)
        y_t = self.target_predictor(z)
        ts, tt = self.thetas
        return ts * y_s + tt * y_t, y_s, y_t, z
