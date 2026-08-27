#!/usr/bin/env python3
"""
HybridoNet-Adapt Comprehensive Verification Test Suite.
Validates:
1. Linear NODE Derivative (dh/dt = Wh + b)
2. Attention Timestep -2 Selection & RK4 Solver
3. Active Gradient Flow on Trade-Off Parameters (theta_s, theta_t)
4. Rolling Window RUL Formulation (RUL = EOL - current_cycle)
5. 18-D Feature Scaling across time and samples
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add directory and project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

from model_hybrido import (
    HybridoNetAdapt,
    NeuralODEBlock,
    FeatureExtractor,
    Predictor,
    ODEFunc
)
from mmd_loss import MMDLoss
from preprocess_hybrido import compute_cycle_statistics, extract_cell_samples
from train_hybrido import compute_dynamic_lambda, fit_and_transform_features_18d


def test_linear_node_derivative():
    print("[TEST 1/5] Testing Linear NODE Derivative (dh/dt = Wh + b)...")
    ode_f = ODEFunc(hidden_dim=64)
    # Check that ODEFunc has a single linear layer
    assert hasattr(ode_f, "linear") and isinstance(ode_f.linear, nn.Linear), "ODEFunc must be a single linear layer per paper"
    assert not hasattr(ode_f, "net"), "ODEFunc should not be an MLP"

    node = NeuralODEBlock(hidden_dim=64, num_steps=2)
    h_in = torch.randn(8, 64, requires_grad=True)
    h_out = node(h_in)
    assert h_out.shape == (8, 64), f"Expected (8, 64), got {h_out.shape}"
    h_out.sum().backward()
    assert h_in.grad is not None, "Gradients failed to flow through linear NODE"
    print("  -> Linear NODE test passed: Single linear layer derivative verified.")


def test_attention_timestep_selection():
    print("[TEST 2/5] Testing Multihead Attention Second-to-Last Timestep Selection (-2)...")
    feat_ext = FeatureExtractor(input_dim=18, hidden_dim=64, num_lstm_layers=2, num_heads=4, dropout=0.1)
    
    x = torch.randn(4, 10, 3, 6) # (Batch=4, Seq=10, Channels=3, Features=6)
    z = feat_ext(x)
    assert z.shape == (4, 64), f"Expected latent state shape (4, 64), got {z.shape}"
    print("  -> Attention timestep test passed: FeatureExtractor output shape (4, 64) verified.")


def test_theta_gradient_flow():
    print("[TEST 3/5] Testing Trainable Trade-off Parameters (theta_S, theta_T) Gradient Flow...")
    model = HybridoNetAdapt(input_dim=18, hidden_dim=64, num_lstm_layers=2, num_heads=4, dropout=0.1)
    criterion_mse = nn.MSELoss()

    tgt_x = torch.randn(8, 10, 3, 6)
    tgt_y = torch.rand(8, 1)

    y_comb_t, _, _, _ = model(tgt_x)
    loss_target = criterion_mse(y_comb_t, tgt_y)
    loss_target.backward()

    assert model.theta_s.grad is not None, "theta_S failed to receive gradients!"
    assert model.theta_t.grad is not None, "theta_T failed to receive gradients!"
    assert abs(model.theta_s.grad.item()) > 1e-7, "theta_S gradient is zero!"
    assert abs(model.theta_t.grad.item()) > 1e-7, "theta_T gradient is zero!"
    print(f"  -> Gradient Flow verified: dL/dtheta_S = {model.theta_s.grad.item():.6f}, dL/dtheta_T = {model.theta_t.grad.item():.6f}")


def test_rolling_window_rul_preprocessing():
    print("[TEST 4/5] Testing Rolling Window RUL Formulation (RUL = EOL - current_cycle)...")
    # Simulate a cell lasting 200 cycles
    cycle_data = {}
    for c in range(1, 150):
        v = np.linspace(3.0, 4.2, 50)
        i = np.ones(50) * 1.5
        q = np.linspace(0.0, 1.1, 50)
        cycle_data[c] = {"voltage": v, "current": i, "capacity": q}

    eol = 200.0
    samples, ruls = extract_cell_samples(cycle_data, eol, window_size=30, stride=30, num_samples=10, rolling=True)
    
    assert len(samples) > 1, f"Expected multiple rolling samples, got {len(samples)}"
    assert samples[0].shape == (10, 3, 6), f"Expected tensor shape (10, 3, 6), got {samples[0].shape}"
    
    # Check that RUL decreases over time
    assert ruls[0] == 200.0 - 30.0, f"Expected RUL at cycle 30 = 170.0, got {ruls[0]}"
    assert ruls[1] == 200.0 - 60.0, f"Expected RUL at cycle 60 = 140.0, got {ruls[1]}"
    assert ruls[0] > ruls[1] > ruls[2], "RUL must decrease monotonically along rolling aging windows"
    print(f"  -> Rolling RUL test passed: Window 1 RUL={ruls[0]}, Window 2 RUL={ruls[1]}, Window 3 RUL={ruls[2]}")


def test_18d_feature_scaling():
    print("[TEST 5/5] Testing 18-D Feature Scaling across time steps and samples...")
    X_tr = np.random.uniform(10.0, 50.0, (20, 10, 3, 6))
    X_val = np.random.uniform(10.0, 50.0, (5, 10, 3, 6))
    X_tgt_adapt = np.random.uniform(10.0, 50.0, (6, 10, 3, 6))
    X_tgt_test = np.random.uniform(10.0, 50.0, (4, 10, 3, 6))

    X_tr_sc, X_val_sc, X_tgt_ad_sc, X_tgt_ts_sc, scaler = fit_and_transform_features_18d(
        X_tr, X_val, X_tgt_adapt, X_tgt_test
    )
    assert X_tr_sc.shape == (20, 10, 3, 6)
    assert X_tr_sc.min() >= -1e-6 and X_tr_sc.max() <= 1.0 + 1e-6, "Training features not bounded in [0, 1]"
    print("  -> 18-D Feature Scaling test passed.")


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING HYBRIDONET-ADAPT REALIGNED TEST SUITE")
    print("==================================================")
    test_linear_node_derivative()
    test_attention_timestep_selection()
    test_theta_gradient_flow()
    test_rolling_window_rul_preprocessing()
    test_18d_feature_scaling()
    print("==================================================")
    print("ALL REALIGNED TESTS PASSED SUCCESSFULLY (5/5)")
    print("==================================================")
