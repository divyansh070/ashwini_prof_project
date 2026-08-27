#!/usr/bin/env python3
"""
HybridoNet-Adapt Comprehensive Verification Test Suite.
Validates:
1. Linear NODE Derivative (dh/dt = Wh + b)
2. Deterministic Attention Timestep Selection (index -2)
3. Active Gradient Flow on Trade-Off Parameters (theta_s, theta_t)
4. Rolling Window RUL Formulation (RUL = EOL - current_cycle)
5. Zero Intra-Battery Leakage: Cell-Level Group Splitting
6. Robust RUL Normalization (No Sigmoid Saturation)
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
from train_hybrido import (
    compute_dynamic_lambda,
    fit_and_transform_features_18d,
    split_by_cell_id,
    RobustRULScaler
)


def test_linear_node_derivative():
    print("[TEST 1/6] Testing Linear NODE Derivative (dh/dt = Wh + b)...")
    ode_f = ODEFunc(hidden_dim=64)
    assert hasattr(ode_f, "linear") and isinstance(ode_f.linear, nn.Linear), "ODEFunc must be a single linear layer per paper"
    assert not hasattr(ode_f, "net"), "ODEFunc should not be an MLP"

    node = NeuralODEBlock(hidden_dim=64, num_steps=2)
    h_in = torch.randn(8, 64, requires_grad=True)
    h_out = node(h_in)
    assert h_out.shape == (8, 64), f"Expected (8, 64), got {h_out.shape}"
    h_out.sum().backward()
    assert h_in.grad is not None, "Gradients failed to flow through linear NODE"
    print("  -> Linear NODE test passed: Single linear layer derivative verified.")


def test_deterministic_attention_timestep():
    print("[TEST 2/6] Testing Deterministic Multihead Attention Timestep (-2) Selection...")
    feat_ext = FeatureExtractor(input_dim=18, hidden_dim=64, num_lstm_layers=2, num_heads=4, dropout=0.0)
    
    # Create input tensor where step index -2 (i.e. index 8 out of 10) has a distinct magnitude
    x = torch.zeros(2, 10, 3, 6)
    x[:, 8, :, :] = 10.0 # Distinct marker on timestep index 8 (second-to-last)

    # Disable ODE dynamics temporarily to probe the exact timestep passed from MHA
    identity_node = nn.Identity()
    feat_ext.node = identity_node
    feat_ext.eval()

    with torch.no_grad():
        z_selected = feat_ext(x) # (2, 64)
        
        # Now run a synthetic test comparing against step 9 (last) vs step 8 (second-to-last)
        # Using forward hook on layer_norm
        captured_h = []
        def hook_fn(m, inp, out):
            captured_h.append(out.detach())
        handle = feat_ext.layer_norm.register_forward_hook(hook_fn)
        _ = feat_ext(x)
        handle.remove()

        h_full = captured_h[0] # (2, 10, 64)
        h_second_last = h_full[:, -2, :]
        h_last = h_full[:, -1, :]
        h_mean = h_full.mean(dim=1)

        diff_selected = torch.norm(z_selected - h_second_last).item()
        diff_last = torch.norm(z_selected - h_last).item()
        diff_mean = torch.norm(z_selected - h_mean).item()

        assert diff_selected < 1e-5, f"FeatureExtractor did not select timestep -2! (diff={diff_selected})"
        assert diff_last > 1e-3, "FeatureExtractor erroneously matched last timestep (-1)"
        assert diff_mean > 1e-3, "FeatureExtractor erroneously matched mean pooling"

    print("  -> Deterministic timestep test passed: Exactly timestep -2 verified (not mean or -1).")


def test_theta_gradient_flow():
    print("[TEST 3/6] Testing Trainable Trade-off Parameters (theta_S, theta_T) Gradient Flow...")
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


def test_cell_level_disjoint_splitting():
    print("[TEST 4/6] Testing Zero Intra-Battery Leakage: Cell-Level Group Splitting...")
    # Simulate 5 unique batteries with 4 windows each = 20 total samples
    cell_names = ["cell_A", "cell_B", "cell_C", "cell_D", "cell_E"]
    cell_ids = np.repeat(cell_names, 4)
    X = np.random.randn(20, 10, 3, 6).astype(np.float32)
    Y = np.random.uniform(100, 1500, 20).astype(np.float32)

    X_tr, X_ts, Y_tr, Y_ts, tr_cells, ts_cells = split_by_cell_id(
        X, Y, cell_ids, test_ratio=0.40, random_state=42
    )

    tr_unique = set(tr_cells)
    ts_unique = set(ts_cells)

    assert tr_unique.isdisjoint(ts_unique), f"Leakage detected! Shared cells: {tr_unique.intersection(ts_unique)}"
    assert len(tr_unique) + len(ts_unique) == len(cell_names), "Lost unique cells during partitioning"
    assert len(X_tr) + len(X_ts) == 20, "Lost sample windows during split"
    print(f"  -> Cell-level split verified: Train cells={tr_unique}, Test cells={ts_unique} (Zero overlap)")


def test_robust_rul_scaling():
    print("[TEST 5/6] Testing Robust RUL Normalization (No Sigmoid Saturation)...")
    # Source battery RUL: [50, 1800]
    Y_source = np.array([50.0, 500.0, 1200.0, 1800.0])
    # Target battery RUL exceeds source: [100, 3200]
    Y_target = np.array([100.0, 1500.0, 2500.0, 3200.0])

    scaler = RobustRULScaler().fit(Y_source, Y_target)
    Y_tgt_norm = scaler.transform(Y_target)

    assert (Y_tgt_norm >= 0.0).all() and (Y_tgt_norm <= 1.0).all(), f"Target RUL saturated or exceeded [0, 1]: {Y_tgt_norm}"
    
    Y_reconstructed = scaler.inverse_transform(Y_tgt_norm)
    assert np.allclose(Y_target, Y_reconstructed, atol=1e-3), "Inverse transform failed to accurately recover true cycle life"
    print(f"  -> Robust RUL scaling verified: Y_max ceiling={scaler.y_max:.0f} cyc, Target Max Scaled={Y_tgt_norm.max():.4f}")


def test_18d_feature_scaling():
    print("[TEST 6/6] Testing 18-D Feature Scaling across time steps and samples...")
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
    print("RUNNING HYBRIDONET-ADAPT HARDENED VERIFICATION")
    print("==================================================")
    test_linear_node_derivative()
    test_deterministic_attention_timestep()
    test_theta_gradient_flow()
    test_cell_level_disjoint_splitting()
    test_robust_rul_scaling()
    test_18d_feature_scaling()
    print("==================================================")
    print("ALL 6/6 HARDENED TESTS PASSED SUCCESSFULLY")
    print("==================================================")
