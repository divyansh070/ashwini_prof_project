#!/usr/bin/env python3
"""
HybridoNet-Adapt Comprehensive Verification Test Suite (8/8 Tests).
Validates:
1. Linear NODE Derivative (dh/dt = Wh + b, Table 2 hidden=128)
2. Deterministic Multihead Attention Last Timestep Selection (index -1)
3. Active Gradient Flow on Trade-Off Parameters (theta_s, theta_t)
4. Rolling Window RUL Formulation (RUL = EOL - current_cycle)
5. Zero Intra-Battery Leakage: Cell-Level Group Splitting & ValueError on Single Cell
6. Fixed Physical RUL Normalization Ceiling (5000 cyc) & ValueError on Overflow
7. 18-D Feature Scaling across time and samples
8. Full End-to-End Pipeline Smoke Test
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
    RobustRULScaler,
    BatteryDataset,
    train_hybrido_session,
    DEFAULT_RUL_MAX_CEILING
)


def test_linear_node_derivative():
    print("[TEST 1/8] Testing Linear NODE Derivative (dh/dt = Wh + b, hidden=128)...")
    ode_f = ODEFunc(hidden_dim=128)
    assert hasattr(ode_f, "linear") and isinstance(ode_f.linear, nn.Linear), "ODEFunc must be a single linear layer per paper"
    assert not hasattr(ode_f, "net"), "ODEFunc should not be an MLP"

    node = NeuralODEBlock(hidden_dim=128, num_steps=2)
    h_in = torch.randn(8, 128, requires_grad=True)
    h_out = node(h_in)
    assert h_out.shape == (8, 128), f"Expected (8, 128), got {h_out.shape}"
    h_out.sum().backward()
    assert h_in.grad is not None, "Gradients failed to flow through linear NODE"
    print("  -> Linear NODE test passed: Single linear layer derivative (128-D) verified.")


def test_deterministic_attention_timestep():
    print("[TEST 2/8] Testing Deterministic Multihead Attention Last Timestep (-1) Selection...")
    feat_ext = FeatureExtractor(input_dim=18, hidden_dim=128, num_lstm_layers=2, num_heads=4, dropout=0.0)
    
    # Create input tensor where step index -1 (i.e. index 9 out of 10) has a distinct marker
    x = torch.zeros(2, 10, 3, 6)
    x[:, 9, :, :] = 10.0 # Distinct marker on last timestep index 9

    identity_node = nn.Identity()
    feat_ext.node = identity_node
    feat_ext.eval()

    with torch.no_grad():
        z_selected = feat_ext(x) # (2, 128)
        
        captured_h = []
        def hook_fn(m, inp, out):
            captured_h.append(out.detach())
        handle = feat_ext.layer_norm.register_forward_hook(hook_fn)
        _ = feat_ext(x)
        handle.remove()

        h_full = captured_h[0] # (2, 10, 128)
        h_last = h_full[:, -1, :]
        h_second_last = h_full[:, -2, :]
        h_mean = h_full.mean(dim=1)

        diff_selected = torch.norm(z_selected - h_last).item()
        diff_second_last = torch.norm(z_selected - h_second_last).item()
        diff_mean = torch.norm(z_selected - h_mean).item()

        assert diff_selected < 1e-5, f"FeatureExtractor did not select last timestep -1! (diff={diff_selected})"
        assert diff_second_last > 1e-3, "FeatureExtractor erroneously matched second-to-last timestep (-2)"
        assert diff_mean > 1e-3, "FeatureExtractor erroneously matched mean pooling"

    print("  -> Deterministic timestep test passed: Exactly last timestep (-1) verified.")


def test_theta_gradient_flow():
    print("[TEST 3/8] Testing Trainable Trade-off Parameters (theta_S, theta_T) Gradient Flow...")
    model = HybridoNetAdapt(input_dim=18, hidden_dim=128, num_lstm_layers=2, num_heads=4, dropout=0.1)
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
    print("[TEST 4/8] Testing Rolling Window RUL Formulation (RUL = EOL - current_cycle)...")
    cycle_data = {}
    for c in range(1, 150):
        v = np.linspace(3.0, 4.2, 50)
        i = np.ones(50) * 1.5
        q = np.linspace(0.0, 1.1, 50)
        cycle_data[c] = {"voltage": v, "current": i, "capacity": q}

    eol = 200.0
    samples, ruls = extract_cell_samples(cycle_data, eol, window_size=30, stride=30, num_samples=10, rolling=True)
    
    assert len(samples) >= 3, f"Expected at least 3 rolling windows, got {len(samples)}"
    assert samples[0].shape == (10, 3, 6), f"Expected tensor shape (10, 3, 6), got {samples[0].shape}"
    
    # Check exact mathematical RUL values:
    assert ruls[0] == 170.0, f"Expected Window 1 RUL=170.0, got {ruls[0]}"
    assert ruls[1] == 140.0, f"Expected Window 2 RUL=140.0, got {ruls[1]}"
    assert ruls[2] == 110.0, f"Expected Window 3 RUL=110.0, got {ruls[2]}"
    print(f"  -> Rolling RUL test passed: Window 1 (cycle 30) RUL={ruls[0]}, Window 2 (cycle 60) RUL={ruls[1]}, Window 3 (cycle 90) RUL={ruls[2]}")


def test_cell_level_disjoint_splitting():
    print("[TEST 5/8] Testing Zero Intra-Battery Leakage: Cell-Level Group Splitting...")
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

    # Test that single cell dataset raises ValueError rather than falling back to random window split
    single_cell_ids = np.repeat(["cell_solo"], 10)
    try:
        _ = split_by_cell_id(X[:10], Y[:10], single_cell_ids, test_ratio=0.2)
        assert False, "Failed to raise ValueError on single-cell dataset!"
    except ValueError:
        pass # Expected

    print("  -> Cell-level split test passed: Zero window overlap and ValueError on single cell verified.")


def test_fixed_ceiling_rul_scaling():
    print("[TEST 6/8] Testing Fixed Physical RUL Normalization Ceiling (5000 cyc) & Overflow Error...")
    scaler = RobustRULScaler(y_max=DEFAULT_RUL_MAX_CEILING)
    
    Y_valid = np.array([50.0, 1200.0, 3200.0, 4800.0])
    Y_scaled = scaler.transform(Y_valid)

    assert (Y_scaled >= 0.0).all() and (Y_scaled <= 1.0).all(), f"RUL out of [0, 1] bounds: {Y_scaled}"
    assert Y_scaled.max() < 1.0, f"Max value hit saturation: {Y_scaled.max()}"

    Y_recovered = scaler.inverse_transform(Y_scaled)
    assert np.allclose(Y_valid, Y_recovered, atol=1e-3), "Inverse transform failed to accurately recover true cycle life"

    # Verify that exceeding the ceiling raises a loud ValueError instead of silent corruption
    Y_overflow = np.array([500.0, 6000.0])
    try:
        _ = scaler.transform(Y_overflow)
        assert False, "Failed to raise ValueError on RUL > 5000!"
    except ValueError:
        pass # Expected

    print("  -> Fixed ceiling test passed: 5000-cycle normalization and loud overflow error verified.")


def test_18d_feature_scaling():
    print("[TEST 7/8] Testing 18-D Feature Scaling across time steps and samples...")
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


def test_end_to_end_smoke():
    print("[TEST 8/8] Testing Full End-to-End Pipeline Smoke Run...")
    # Generate 4 source cells, 4 target cells (3 windows each)
    src_cells = np.repeat([f"src_c{i}" for i in range(4)], 3)
    tgt_cells = np.repeat([f"tgt_c{i}" for i in range(4)], 3)

    X_src = np.random.randn(12, 10, 3, 6).astype(np.float32)
    Y_src = np.random.uniform(200, 2500, 12).astype(np.float32)

    X_tgt = np.random.randn(12, 10, 3, 6).astype(np.float32)
    Y_tgt = np.random.uniform(200, 3500, 12).astype(np.float32)

    # 1. Cell-level splitting
    X_tr, X_val, Y_tr, Y_val, _, _ = split_by_cell_id(X_src, Y_src, src_cells, test_ratio=0.25, random_state=42)
    X_ad, X_ts, Y_ad, Y_ts, _, _ = split_by_cell_id(X_tgt, Y_tgt, tgt_cells, test_ratio=0.50, random_state=42)

    # 2. Scaling
    X_tr_sc, X_val_sc, X_ad_sc, X_ts_sc, _ = fit_and_transform_features_18d(X_tr, X_val, X_ad, X_ts)
    scaler_y = RobustRULScaler(y_max=DEFAULT_RUL_MAX_CEILING).fit()
    Y_tr_sc = scaler_y.transform(Y_tr)
    Y_val_sc = scaler_y.transform(Y_val)
    Y_ad_sc = scaler_y.transform(Y_ad)
    Y_ts_sc = scaler_y.transform(Y_ts)

    src_loader = DataLoader(BatteryDataset(X_tr_sc, Y_tr_sc), batch_size=4, shuffle=True, drop_last=(len(X_tr_sc) > 4))
    val_loader = DataLoader(BatteryDataset(X_val_sc, Y_val_sc), batch_size=4, shuffle=False)
    tgt_loader = DataLoader(BatteryDataset(X_ad_sc, Y_ad_sc), batch_size=4, shuffle=True, drop_last=(len(X_ad_sc) > 4))
    tgt_test_loader = DataLoader(BatteryDataset(X_ts_sc, Y_ts_sc), batch_size=4, shuffle=False)

    model = HybridoNetAdapt(input_dim=18, hidden_dim=128, num_lstm_layers=2, num_heads=4, dropout=0.1)

    results = train_hybrido_session(
        model=model,
        source_loader=src_loader,
        target_loader=tgt_loader,
        val_loader=val_loader,
        target_test_loader=tgt_test_loader,
        target_y_test_raw=Y_ts,
        val_y_raw=Y_val,
        scaler_y=scaler_y,
        epochs=1,
        lr=0.0005,
        device="cpu"
    )

    assert np.isfinite(results["test_rmse"]), "Test RMSE is not finite!"
    assert np.isfinite(results["test_mape"]), "Test MAPE is not finite!"
    assert np.isfinite(results["final_theta_s"]) and np.isfinite(results["final_theta_t"]), "Theta weights are not finite!"
    print(f"  -> End-to-End smoke test passed: Test RMSE={results['test_rmse']:.2f} cyc, theta_S={results['final_theta_s']:.3f}, theta_T={results['final_theta_t']:.3f}")


if __name__ == "__main__":
    print("==================================================")
    print("RUNNING HYBRIDONET-ADAPT HARDENED VERIFICATION (8/8)")
    print("==================================================")
    test_linear_node_derivative()
    test_deterministic_attention_timestep()
    test_theta_gradient_flow()
    test_rolling_window_rul_preprocessing()
    test_cell_level_disjoint_splitting()
    test_fixed_ceiling_rul_scaling()
    test_18d_feature_scaling()
    test_end_to_end_smoke()
    print("==================================================")
    print("ALL 8/8 HARDENED TESTS PASSED SUCCESSFULLY")
    print("==================================================")
