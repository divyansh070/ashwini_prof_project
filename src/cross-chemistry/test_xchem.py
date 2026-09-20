#!/usr/bin/env python3
"""
Stage 0 Cross-Chemistry Verification Test Suite.

Runs fast static and synthetic unit tests in CPU-seconds:
1. Differential rate feature extraction & shapes (18-D, 30-D, 36-D).
2. Signal dropping: confirms unselected signals (V) are dropped, not zero-padded.
3. Model architecture: confirms input_dim is required (no default) and supports arbitrary dimensions.
4. RobustRULScaler bug fix: confirms ceiling is locked and immutable during transform().
5. Dynamic feature scaler: handles 3D and 4D tensors with [-0.5, 1.5] clipping.
6. End-to-end synthetic smoke run with canary logging.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
hybridonet_dir = os.path.join(os.path.dirname(current_dir), "hybridonet-adapt")

for p in [current_dir, hybridonet_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from preprocess_xchem import (
    extract_window_tensor,
    parse_delta_signals,
    build_feature_names,
    compute_cycle_statistics
)
from model_xchem import HybridoNetAdapt, FeatureExtractor
from train_xchem import (
    RobustRULScaler,
    fit_and_transform_features,
    BatteryDataset,
    train_hybrido_session,
    DEFAULT_RUL_MAX_CEILING
)


def test_delta_computation_and_shapes():
    print("\n[TEST 1/6] Testing Delta Feature Extraction & Tensor Shapes...")
    # Synthetic cycle data
    cycle_data = {}
    for c in range(1, 31):
        cycle_data[c] = {
            "voltage": np.full(50, 3.5 + 0.01 * c),
            "current": np.full(50, -1.0 - 0.02 * c),
            "capacity": np.full(50, 1.2 - 0.005 * c)
        }
    window = list(range(1, 31))

    # 1. Base 18-D features (no deltas)
    t_base = extract_window_tensor(cycle_data, window, num_samples=10, use_deltas=False)
    assert t_base.shape == (10, 18), f"Expected (10, 18), got {t_base.shape}"

    # 2. Arm B: 30-D features (Q, I deltas only)
    t_30 = extract_window_tensor(cycle_data, window, num_samples=10, use_deltas=True, active_delta_signals=["Q", "I"])
    assert t_30.shape == (10, 30), f"Expected (10, 30), got {t_30.shape}"

    # Verify timestep 0 delta is zero
    assert np.allclose(t_30[0, 18:], 0.0), "Timestep 0 deltas must be exactly zero"
    # Verify timestep 9 delta is non-zero (capacity decayed)
    assert not np.allclose(t_30[9, 18:], 0.0), "Later timesteps must have non-zero deltas"

    # 3. Arm C: 36-D features (all deltas)
    t_36 = extract_window_tensor(cycle_data, window, num_samples=10, use_deltas=True, active_delta_signals=["V", "I", "Q"])
    assert t_36.shape == (10, 36), f"Expected (10, 36), got {t_36.shape}"

    print(" -> Passed: Shapes verified (18-D, 30-D, 36-D).")


def test_unselected_deltas_are_dropped():
    print("\n[TEST 2/6] Testing that Unselected Signals are Dropped (Not Zero-Padded)...")
    signals = parse_delta_signals("Q,I")
    assert signals == ["I", "Q"], f"Expected canonical order ['I', 'Q'], got {signals}"

    names_30 = build_feature_names(use_deltas=True, active_delta_signals=signals)
    assert len(names_30) == 30, f"Expected 30 feature names, got {len(names_30)}"

    # Confirm voltage deltas do NOT appear anywhere in the names
    v_deltas = [n for n in names_30 if n.startswith("dV_")]
    assert len(v_deltas) == 0, f"Arm B must NOT have dV features, but found: {v_deltas}"

    # Confirm I and Q deltas DO appear
    i_deltas = [n for n in names_30 if n.startswith("dI_")]
    q_deltas = [n for n in names_30 if n.startswith("dQ_")]
    assert len(i_deltas) == 6, f"Expected 6 dI features, got {len(i_deltas)}"
    assert len(q_deltas) == 6, f"Expected 6 dQ features, got {len(q_deltas)}"

    print(" -> Passed: Arm B is genuinely 30-D with V deltas completely dropped.")


def test_input_dim_required_no_default():
    print("\n[TEST 3/6] Testing input_dim Requirement (No Default Value)...")
    # Verify calling without input_dim raises TypeError
    try:
        _ = HybridoNetAdapt()
        assert False, "HybridoNetAdapt should have raised TypeError when input_dim is missing!"
    except TypeError:
        pass

    try:
        _ = FeatureExtractor()
        assert False, "FeatureExtractor should have raised TypeError when input_dim is missing!"
    except TypeError:
        pass

    # Verify instantiating with explicit dimensions works seamlessly
    for dim in [18, 30, 36]:
        model = HybridoNetAdapt(input_dim=dim)
        x = torch.randn(4, 10, dim)
        y_comb, ys, yt, z = model(x)
        assert y_comb.shape == (4, 1), f"Expected (4, 1), got {y_comb.shape}"
        assert z.shape == (4, 128), f"Expected (4, 128), got {z.shape}"
        assert torch.isclose(model.theta_s + model.theta_t, torch.tensor(1.0)), "Theta weights must sum to 1.0"

    print(" -> Passed: input_dim is strictly required and supports 18-D, 30-D, 36-D.")


def test_robust_rul_scaler_bug_fix():
    print("\n[TEST 4/6] Testing RobustRULScaler Immutability Bug Fix...")
    scaler = RobustRULScaler(y_max=2500)

    # Calling transform before fit raises RuntimeError
    try:
        scaler.transform(np.array([100.0]))
        assert False, "Transform before fit should raise RuntimeError"
    except RuntimeError:
        pass

    # Fit on splits with max label 2100 (below 2500)
    y_tr = np.array([100.0, 1500.0, 2100.0])
    y_val = np.array([200.0, 800.0])
    y_tgt = np.array([50.0, 1900.0])
    scaler.fit(y_tr, y_val, y_tgt)

    assert scaler.y_max == 2500.0, f"Expected 2500.0, got {scaler.y_max}"

    # Repeated transforms must NOT mutate y_max
    sc1 = scaler.transform(y_tr)
    assert scaler.y_max == 2500.0
    sc2 = scaler.transform(y_val)
    assert scaler.y_max == 2500.0
    sc3 = scaler.transform(y_tgt)
    assert scaler.y_max == 2500.0

    # Auto-expansion during fit when max exceeds ceiling
    scaler_auto = RobustRULScaler(y_max=DEFAULT_RUL_MAX_CEILING)
    y_large = np.array([100.0, 3150.0])  # Exceeds 2500
    scaler_auto.fit(y_tr, y_large)
    locked_max = scaler_auto.y_max
    assert locked_max >= 3150.0, "Ceiling should auto-expand during fit"

    # Any unexpected overflow after fit must raise ValueError, NOT silently mutate
    try:
        scaler.transform(np.array([2600.0]))
        assert False, "Expected ValueError on unexpected label overflow in transform()"
    except ValueError:
        pass

    print(" -> Passed: RobustRULScaler locks ceiling during fit() and remains strictly immutable.")


def test_dynamic_feature_scaler():
    print("\n[TEST 5/6] Testing Dynamic Feature Scaler on 30-D Tensors...")
    X_tr = np.random.randn(20, 10, 30).astype(np.float32)
    X_val = np.random.randn(5, 10, 30).astype(np.float32)
    X_ad = np.random.randn(10, 10, 30).astype(np.float32)
    X_ts = np.random.randn(8, 10, 30).astype(np.float32)

    X_tr_s, X_val_s, X_ad_s, X_ts_s, sc = fit_and_transform_features(X_tr, X_val, X_ad, X_ts)

    assert X_tr_s.shape == (20, 10, 30)
    assert X_val_s.shape == (5, 10, 30)
    assert X_ad_s.shape == (10, 10, 30)
    assert X_ts_s.shape == (8, 10, 30)

    # Test clipping [-0.5, 1.5]
    assert np.all(X_ts_s >= -0.5), "Values must be clipped at -0.5"
    assert np.all(X_ts_s <= 1.5), "Values must be clipped at 1.5"

    print(" -> Passed: Dynamic scaling and [-0.5, 1.5] clipping verified.")


def test_synthetic_smoke_run():
    print("\n[TEST 6/6] Testing End-to-End Smoke Training Session on 30-D Tensors...")
    input_dim = 30
    model = HybridoNetAdapt(input_dim=input_dim, hidden_dim=64, norm_type="layernorm")

    # Small synthetic datasets
    X_src = np.random.randn(32, 10, input_dim).astype(np.float32)
    Y_src = np.random.uniform(100, 2000, size=(32,)).astype(np.float32)

    X_tgt = np.random.randn(32, 10, input_dim).astype(np.float32)
    Y_tgt = np.random.uniform(100, 2000, size=(32,)).astype(np.float32)

    scaler_y = RobustRULScaler(y_max=2500).fit(Y_src, Y_tgt)
    Y_src_sc = scaler_y.transform(Y_src)
    Y_tgt_sc = scaler_y.transform(Y_tgt)

    src_loader = DataLoader(BatteryDataset(X_src, Y_src_sc), batch_size=16)
    tgt_loader = DataLoader(BatteryDataset(X_tgt, Y_tgt_sc), batch_size=16)
    val_loader = DataLoader(BatteryDataset(X_src[:16], Y_src_sc[:16]), batch_size=16)
    test_loader = DataLoader(BatteryDataset(X_tgt[:16], Y_tgt_sc[:16]), batch_size=16)

    res = train_hybrido_session(
        model=model,
        source_loader=src_loader,
        target_loader=tgt_loader,
        val_loader=val_loader,
        target_test_loader=test_loader,
        target_y_test_raw=Y_tgt[:16],
        val_y_raw=Y_src[:16],
        scaler_y=scaler_y,
        epochs=1,
        lr=0.001,
        early_stop_patience=None,
        checkpoint_path=None,
        device="cpu"
    )

    assert "test_rmse" in res
    assert "test_r2" in res
    assert "test_paper_mape" in res
    assert 0.0 <= res["final_theta_s"] <= 1.0
    assert 0.0 <= res["final_theta_t"] <= 1.0

    print(f" -> Smoke run passed: Test RMSE={res['test_rmse']:.2f} cyc, theta_S={res['final_theta_s']:.3f}, theta_T={res['final_theta_t']:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING CROSS-CHEMISTRY (STAGE 0) VERIFICATION TEST SUITE")
    print("=" * 60)
    test_delta_computation_and_shapes()
    test_unselected_deltas_are_dropped()
    test_input_dim_required_no_default()
    test_robust_rul_scaler_bug_fix()
    test_dynamic_feature_scaler()
    test_synthetic_smoke_run()
    print("=" * 60)
    print("ALL 6/6 STAGE 0 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
