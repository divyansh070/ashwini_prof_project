#!/usr/bin/env python3
"""
HybridoNet-Adapt Diagnostic Tool: Feature Scaler & Aging Signal Audit.

Performs:
1. Scaler Distribution Audit: Evaluates feature-by-feature min, max, and std of Target (HUST)
   when scaled by a MinMaxScaler fitted strictly on Source (MATR). Identifies out-of-bound
   values (< 0.0 or > 1.0) and crushed variance (std ~ 0).
2. Aging Signal Audit: Evaluates whether capacity, voltage, and current features actually fade
   with decreasing RUL across cells, computing Pearson correlation r(RUL, feature).
"""

import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr


FEATURE_NAMES = [
    # Row 0: Voltage signals
    "V_mean", "V_std", "V_min", "V_max", "V_var", "V_med",
    # Row 1: Current signals
    "I_mean", "I_std", "I_min", "I_max", "I_var", "I_med",
    # Row 2: Capacity signals
    "Q_mean", "Q_std", "Q_min", "Q_max", "Q_var", "Q_med",
]


def run_diagnostics(
    source_path: str = "data/hybridonet/processed/MATR_raw_features.npz",
    target_path: str = "data/hybridonet/processed/HUST_raw_features.npz"
):
    print("=" * 80)
    print("HYBRIDONET-ADAPT DIAGNOSTIC AUDIT: SCALER DISTRIBUTION & AGING SIGNALS")
    print("=" * 80)

    if not os.path.exists(source_path) or not os.path.exists(target_path):
        print(f"Error: Missing feature files.\n  Source: {source_path}\n  Target: {target_path}")
        sys.exit(1)

    print(f"Loading Source: {source_path}")
    s_data = np.load(source_path, allow_pickle=True)
    S_X, S_Y, S_cells = s_data['X'], s_data['Y'], s_data['cell_ids']

    print(f"Loading Target: {target_path}")
    t_data = np.load(target_path, allow_pickle=True)
    T_X, T_Y, T_cells = t_data['X'], t_data['Y'], t_data['cell_ids']

    print(f"Source samples: {S_X.shape[0]} across {len(np.unique(S_cells))} cells")
    print(f"Target samples: {T_X.shape[0]} across {len(np.unique(T_cells))} cells")
    print()

    # ---------------------------------------------------------
    # DIAGNOSTIC 1: FEATURE SCALER AUDIT
    # ---------------------------------------------------------
    print("-" * 80)
    print("DIAGNOSTIC 1: SCALER AUDIT (MinMaxScaler fit on Source only vs Target transform)")
    print("-" * 80)

    S_flat = S_X.reshape(-1, 18)
    T_flat = T_X.reshape(-1, 18)

    # Scaler fit strictly on source
    sc_src = MinMaxScaler(feature_range=(0.0, 1.0)).fit(S_flat)
    T_trans_src = sc_src.transform(T_flat)

    # Combined scaler fit on source + target
    sc_comb = MinMaxScaler(feature_range=(0.0, 1.0)).fit(np.vstack([S_flat, T_flat]))
    T_trans_comb = sc_comb.transform(T_flat)

    print(f"{'Idx':<4} {'Feature Name':<10} | {'Source Min':<10} {'Source Max':<10} | {'Tgt Min':<10} {'Tgt Max':<10} | {'Tgt Sc (Src-Fit)':<22} {'Tgt Std':<8} | {'Status'}")
    print("-" * 115)

    severe_out_of_bounds = []
    crushed_variance = []

    for i in range(18):
        name = FEATURE_NAMES[i]
        s_min, s_max = float(S_flat[:, i].min()), float(S_flat[:, i].max())
        t_min, t_max = float(T_flat[:, i].min()), float(T_flat[:, i].max())
        t_sc_min, t_sc_max = float(T_trans_src[:, i].min()), float(T_trans_src[:, i].max())
        t_sc_std = float(T_trans_src[:, i].std())

        status = "OK"
        if t_sc_min < -0.2 or t_sc_max > 1.2:
            status = "OUT_OF_BOUNDS"
            severe_out_of_bounds.append((name, t_sc_min, t_sc_max))
        elif t_sc_std < 0.01:
            status = "CRUSHED_VARIANCE"
            crushed_variance.append((name, t_sc_std))

        print(f"f{i:02d}  {name:<10} | {s_min:10.4f} {s_max:10.4f} | {t_min:10.4f} {t_max:10.4f} | [{t_sc_min:7.2f}, {t_sc_max:7.2f}]       {t_sc_std:8.4f} | {status}")

    print("-" * 115)
    if severe_out_of_bounds:
        print(f"⚠️ OUT-OF-BOUNDS WARNING: {len(severe_out_of_bounds)} features have Target values far outside [0, 1] when using Source-only scaler!")
        for item in severe_out_of_bounds:
            print(f"   - {item[0]}: [{item[1]:.2f}, {item[2]:.2f}]")
    if crushed_variance:
        print(f"⚠️ CRUSHED VARIANCE WARNING: {len(crushed_variance)} features have near-zero variance on Target!")
        for item in crushed_variance:
            print(f"   - {item[0]}: std={item[1]:.4f}")
    print()

    # VERIFY COMBINED FIT (Source-train + Target-adapt)
    print("-" * 80)
    print("VERIFICATION: COMBINED FIT (Source + Target Adaptation) WITH [-0.5, 1.5] CLIPPING")
    print("-" * 80)
    T_comb_clipped = np.clip(T_trans_comb, -0.5, 1.5)
    print(f"{'Idx':<4} {'Feature Name':<10} | {'Combined Min/Max':<20} {'Std':<8} | {'Status'}")
    print("-" * 65)
    all_ok = True
    for i in range(18):
        name = FEATURE_NAMES[i]
        c_min = float(T_comb_clipped[:, i].min())
        c_max = float(T_comb_clipped[:, i].max())
        c_std = float(T_comb_clipped[:, i].std())
        stat = "OK ✅"
        if c_min < -0.2 or c_max > 1.2:
            stat = "OUT_OF_BOUNDS ⚠️"
            all_ok = False
        print(f"f{i:02d}  {name:<10} | [{c_min:7.2f}, {c_max:7.2f}]       {c_std:8.4f} | {stat}")
    print("-" * 65)
    if all_ok:
        print("✅ COMBINED FIT CONFIRMED: All 18 target features safely bounded within [0, 1]!")
    print()

    # ---------------------------------------------------------
    # DIAGNOSTIC 2: AGING SIGNAL AUDIT (Capacity & Voltage Fade)
    # ---------------------------------------------------------
    print("-" * 80)
    print("DIAGNOSTIC 2: AGING SIGNAL AUDIT (Does capacity fade with RUL across Target cells?)")
    print("-" * 80)

    unique_cells = np.unique(T_cells)
    print(f"Evaluating {min(5, len(unique_cells))} sample Target cells for degradation trajectories...\n")

    cell_correlations_q_max = []
    cell_correlations_q_mean = []
    cell_correlations_v_mean = []

    for cid in unique_cells[:5]:
        mask = (T_cells == cid)
        Xi = T_X[mask]
        Yi = T_Y[mask]

        if len(Yi) < 5:
            continue

        order = np.argsort(-Yi)  # early life (high RUL) -> late life (low RUL)
        sample_indices = np.linspace(0, len(order) - 1, min(8, len(order)), dtype=int)

        print(f"Cell: {cid} ({len(Yi)} observation windows, RUL span: {Yi.min():.0f} - {Yi.max():.0f} cyc)")
        print(f"{'Cycle RUL':<12} {'Q_mean (Ah)':<14} {'Q_max (Ah)':<14} {'V_mean (V)':<14} {'I_mean (A)':<14}")
        for idx in sample_indices:
            w = Xi[order[idx]]
            r = Yi[order[idx]]
            q_mean = float(w[:, 2, 0].mean())
            q_max = float(w[:, 2, 3].mean())
            v_mean = float(w[:, 0, 0].mean())
            i_mean = float(w[:, 1, 0].mean())
            print(f"{r:9.0f} cyc   {q_mean:10.4f}     {q_max:10.4f}     {v_mean:10.4f}     {i_mean:10.4f}")

        # Compute Pearson correlation with RUL
        # For a degrading cell, as RUL decreases, capacity should decrease (so correlation r(RUL, Q) should be strongly POSITIVE)
        q_max_series = np.array([Xi[k][:, 2, 3].mean() for k in range(len(Xi))])
        q_mean_series = np.array([Xi[k][:, 2, 0].mean() for k in range(len(Xi))])
        v_mean_series = np.array([Xi[k][:, 0, 0].mean() for k in range(len(Xi))])

        if np.std(q_max_series) > 1e-6 and np.std(Yi) > 1e-6:
            r_qmax, _ = pearsonr(Yi, q_max_series)
            cell_correlations_q_max.append(r_qmax)
        if np.std(q_mean_series) > 1e-6 and np.std(Yi) > 1e-6:
            r_qmean, _ = pearsonr(Yi, q_mean_series)
            cell_correlations_q_mean.append(r_qmean)
        if np.std(v_mean_series) > 1e-6 and np.std(Yi) > 1e-6:
            r_vmean, _ = pearsonr(Yi, v_mean_series)
            cell_correlations_v_mean.append(r_vmean)

        print(f" -> Pearson r(RUL, Q_max) = {cell_correlations_q_max[-1]:+.4f} (positive indicates capacity fades over life)")
        print()

    print("-" * 80)
    print("SUMMARY OF CORRELATIONS ACROSS ALL TARGET CELLS:")
    all_corrs_qmax = []
    for cid in unique_cells:
        mask = (T_cells == cid)
        Xi, Yi = T_X[mask], T_Y[mask]
        if len(Yi) < 5:
            continue
        q_max_series = np.array([Xi[k][:, 2, 3].mean() for k in range(len(Xi))])
        if np.std(q_max_series) > 1e-6 and np.std(Yi) > 1e-6:
            r, _ = pearsonr(Yi, q_max_series)
            all_corrs_qmax.append(r)

    avg_r = float(np.mean(all_corrs_qmax)) if all_corrs_qmax else 0.0
    print(f"Average r(RUL, Q_max) across {len(all_corrs_qmax)} cells: {avg_r:+.4f}")
    if avg_r > 0.5:
        print("✅ STRONG AGING SIGNAL: Capacity strongly and monotonically fades as RUL decreases.")
    elif avg_r > 0.1:
        print("⚠️ WEAK AGING SIGNAL: Capacity shows moderate correlation with RUL.")
    else:
        print("❌ NO AGING SIGNAL: Capacity is flat or uncorrelated with RUL! Check column extraction.")
    print("=" * 80)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/hybridonet/processed/MATR_raw_features.npz"
    tgt = sys.argv[2] if len(sys.argv) > 2 else "data/hybridonet/processed/HUST_raw_features.npz"
    run_diagnostics(src, tgt)
