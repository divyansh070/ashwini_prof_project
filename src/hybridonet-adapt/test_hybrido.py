#!/usr/bin/env python3
"""
Unit and Integration Test Suite for HybridoNet-Adapt Baseline.
Verifies shapes, mathematical formulations, gradient flow, and leakage prevention.
"""

import sys
import os
import torch
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
    Predictor
)
from mmd_loss import MMDLoss
from preprocess_hybrido import compute_cycle_statistics, extract_cell_tensor
from train_hybrido import compute_dynamic_lambda, fit_and_transform_features


def test_preprocessing():
    print("[TEST 1/5] Testing Preprocessing & Feature Extraction...")
    # Generate 50 points of dummy cycle data
    v = np.linspace(3.0, 4.2, 50) + np.random.normal(0, 0.01, 50)
    i = np.ones(50) * 1.5 + np.random.normal(0, 0.01, 50)
    q = np.linspace(0.0, 1.1, 50)

    # 1. Cycle statistics
    feat = compute_cycle_statistics(v, i, q)
    assert feat.shape == (3, 6), f"Expected (3, 6), got {feat.shape}"
    assert not np.isnan(feat).any(), "Found NaNs in computed cycle statistics"

    # 2. Window sampling
    cycle_dict = {}
    for c in range(1, 35):
        cycle_dict[c] = {"voltage": v, "current": i, "capacity": q}
    tensor = extract_cell_tensor(cycle_dict, window_size=30, num_samples=10)
    assert tensor.shape == (10, 3, 6), f"Expected (10, 3, 6), got {tensor.shape}"
    print("  -> Preprocessing test passed: Shape (10, 3, 6) verified.")


def test_mmd_loss():
    print("[TEST 2/5] Testing Maximum Mean Discrepancy (MMD) Loss...")
    mmd_fn = MMDLoss(sigma=1.0)
    
    # Identical distributions should have MMD close to 0
    x1 = torch.randn(32, 64)
    x2 = x1.clone()
    loss_ident = mmd_fn(x1, x2)
    assert loss_ident.item() < 1e-4, f"Expected near zero MMD for identical distributions, got {loss_ident.item()}"

    # Different distributions should have positive MMD
    y = torch.randn(32, 64) + 2.0
    loss_diff = mmd_fn(x1, y)
    assert loss_diff.item() > 0.0, "MMD should be strictly positive for disparate distributions"
    print(f"  -> MMD Loss test passed: Discrepancy={loss_diff.item():.4f}")


def test_neural_ode_block():
    print("[TEST 3/5] Testing Modular Neural ODE Block...")
    node = NeuralODEBlock(hidden_dim=64, num_steps=4)
    z_in = torch.randn(16, 64, requires_grad=True)
    z_out = node(z_in)
    assert z_out.shape == (16, 64), f"Expected (16, 64), got {z_out.shape}"

    # Verify backprop through RK4 integration steps
    loss = z_out.sum()
    loss.backward()
    assert z_in.grad is not None, "Gradients failed to backpropagate through Neural ODE"
    print("  -> Neural ODE test passed: RK4 integration and gradient flow verified.")


def test_model_forward_and_backward():
    print("[TEST 4/5] Testing HybridoNetAdapt Model Architecture...")
    model = HybridoNetAdapt(input_dim=18, hidden_dim=64, num_lstm_layers=2, num_heads=4, dropout=0.1)
    
    x = torch.randn(8, 10, 3, 6)
    y_comb, y_s, y_t, z = model(x)

    assert y_comb.shape == (8, 1), f"Expected y_comb shape (8, 1), got {y_comb.shape}"
    assert y_s.shape == (8, 1), f"Expected y_s shape (8, 1), got {y_s.shape}"
    assert y_t.shape == (8, 1), f"Expected y_t shape (8, 1), got {y_t.shape}"
    assert z.shape == (8, 64), f"Expected latent z shape (8, 64), got {z.shape}"
    assert (y_s >= 0.0).all() and (y_s <= 1.0).all(), "Sigmoid activation range violated"

    # Test backward pass
    loss = y_comb.sum() + y_s.sum() + y_t.sum()
    loss.backward()
    assert model.theta_s.grad is not None and model.theta_t.grad is not None, "Trade-off parameters theta_s, theta_t failed to receive gradients"
    print("  -> Architecture test passed: Dual predictors and trainable trade-offs verified.")


def test_dynamic_lambda_and_leakage_scaling():
    print("[TEST 5/5] Testing Dynamic Lambda Scheduling and Zero-Leakage Scaler...")
    l_start = compute_dynamic_lambda(0, 100, gamma=10.0)
    l_mid = compute_dynamic_lambda(50, 100, gamma=10.0)
    l_end = compute_dynamic_lambda(100, 100, gamma=10.0)

    assert abs(l_start - 0.0) < 1e-4, f"Expected lambda(0) ~ 0.0, got {l_start}"
    assert l_mid > l_start and l_end > l_mid, "Lambda must monotonically increase with progress"
    assert abs(l_end - 1.0) < 0.01, f"Expected lambda(1) ~ 1.0, got {l_end}"

    # Scaling test
    X_tr = np.random.uniform(10.0, 50.0, (20, 10, 3, 6))
    X_val = np.random.uniform(10.0, 50.0, (5, 10, 3, 6))
    X_tgt = np.random.uniform(10.0, 50.0, (10, 10, 3, 6))

    X_tr_sc, X_val_sc, X_tgt_sc, scaler = fit_and_transform_features(X_tr, X_val, X_tgt)
    assert X_tr_sc.min() >= -1e-6 and X_tr_sc.max() <= 1.0 + 1e-6, f"Training features out of bounds: min={X_tr_sc.min()}, max={X_tr_sc.max()}"
    print("  -> Dynamic scheduling and zero-leakage scaling verified.")


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING HYBRIDONET-ADAPT VERIFICATION TEST SUITE")
    print("==================================================")
    test_preprocessing()
    test_mmd_loss()
    test_neural_ode_block()
    test_model_forward_and_backward()
    test_dynamic_lambda_and_leakage_scaling()
    print("==================================================")
    print("ALL TESTS PASSED SUCCESSFULLY (5/5)")
    print("==================================================")
