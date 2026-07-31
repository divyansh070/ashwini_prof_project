#!/usr/bin/env python3
"""
Feature Engineering Module for Lithium-Ion Battery SOH & RUL Estimation.
Implements rigorous Physics-Informed Differential Capacity Analysis (dQ/dV),
Savitzky-Golay smoothing with peak amplitude preservation, and early-cycle tracking
metrics across the Stanford/MIT Fast-Charging Dataset.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FeatureEngineering")

# Strictly monotonic, uniform voltage grid for dQ/dV interpolation (2.05 V to 3.50 V, 1000 points)
# Step size dV ~ 1.45 mV
V_GRID = np.linspace(2.05, 3.50, 1000)

# Savitzky-Golay Filter parameters:
# Window size 31 points (~45 mV window) and polynomial order 3 (cubic)
# Smooths sensor noise without attenuating the sharp LFP phase-transition peaks at ~3.2V and ~3.3V
SG_WINDOW_LENGTH = 31
SG_POLYORDER = 3


def compute_dqdv_curve(df_cycle: pd.DataFrame):
    """
    Computes raw and Savitzky-Golay smoothed dQ/dV curves on a strictly monotonic voltage grid.
    Returns V_GRID, dqdv_raw, dqdv_smooth.
    """
    # Filter strictly for discharge step (where current < -0.1 A)
    if "current_A" in df_cycle.columns:
        df_dis = df_cycle[df_cycle["current_A"] < -0.1]
        if len(df_dis) >= 15:
            df_cycle = df_dis

    # Extract voltage and capacity
    v = df_cycle["voltage_V"].values
    q = df_cycle["capacity_Ah"].values

    # Remove NaN values
    valid_idx = ~(np.isnan(v) | np.isnan(q))
    v = v[valid_idx]
    q = q[valid_idx]

    if len(v) < 15:
        return V_GRID, np.zeros_like(V_GRID), np.zeros_like(V_GRID), np.zeros_like(V_GRID)

    # Ensure strictly monotonic ascending voltage for interpolation
    # Sort by voltage ascending
    sort_idx = np.argsort(v)
    v_sorted = v[sort_idx]
    q_sorted = q[sort_idx]

    # Remove duplicate voltage points to prevent zero step / division by zero
    v_unique, unique_idx = np.unique(v_sorted, return_index=True)
    q_unique = q_sorted[unique_idx]

    if len(v_unique) < 10:
        return V_GRID, np.zeros_like(V_GRID), np.zeros_like(V_GRID), np.zeros_like(V_GRID)

    # Interpolate capacity onto standard V_GRID with clamping to avoid extrapolation noise
    interp_func = interp1d(v_unique, q_unique, kind="linear", bounds_error=False, fill_value=(q_unique[0], q_unique[-1]))
    q_interp = interp_func(V_GRID)

    # Compute numerical derivative dQ/dV using central difference on uniform grid
    dqdv_raw = np.gradient(q_interp, V_GRID)

    # Apply Savitzky-Golay smoothing
    dqdv_smooth = savgol_filter(dqdv_raw, window_length=SG_WINDOW_LENGTH, polyorder=SG_POLYORDER)

    return V_GRID, q_interp, dqdv_raw, dqdv_smooth


def extract_peak_features(v_grid: np.ndarray, dqdv: np.ndarray):
    """
    Extracts LFP phase-transition peak locations and amplitudes from dQ/dV curve.
    - Peak 1 (High V LFP plateau): search region [3.28 V, 3.38 V]
    - Peak 2 (Low V LFP plateau) : search region [3.17 V, 3.27 V]
    """
    # Peak 1 search
    mask1 = (v_grid >= 3.28) & (v_grid <= 3.38)
    if np.any(mask1):
        idx1 = np.argmax(np.abs(dqdv[mask1]))
        v_peak1 = v_grid[mask1][idx1]
        h_peak1 = dqdv[mask1][idx1]
    else:
        v_peak1, h_peak1 = 3.33, 0.0

    # Peak 2 search
    mask2 = (v_grid >= 3.17) & (v_grid <= 3.27)
    if np.any(mask2):
        idx2 = np.argmax(np.abs(dqdv[mask2]))
        v_peak2 = v_grid[mask2][idx2]
        h_peak2 = dqdv[mask2][idx2]
    else:
        v_peak2, h_peak2 = 3.22, 0.0

    return {
        "v_peak1": float(v_peak1),
        "h_peak1": float(h_peak1),
        "v_peak2": float(v_peak2),
        "h_peak2": float(h_peak2)
    }


def compute_delta_dqdv_statistics(q_100: np.ndarray, q_10: np.ndarray, dqdv_100: np.ndarray, dqdv_10: np.ndarray):
    """
    Computes statistical moments and L-norms of the difference curves:
    Delta Q(V) = Q_100(V) - Q_10(V) and Delta(dQ/dV) = dQ/dV_100(V) - dQ/dV_10(V).
    """
    delta_q = q_100 - q_10
    delta = dqdv_100 - dqdv_10

    var_val = float(np.var(delta))
    min_val = float(np.min(delta))
    max_val = float(np.max(delta))
    mean_val = float(np.mean(delta))
    skew_val = float(skew(delta))
    kurt_val = float(kurtosis(delta))

    # Delta Q(V) Severson metrics
    var_q_val = float(np.var(delta_q))
    min_q_val = float(np.min(delta_q))
    log_var_q_val = float(np.log10(np.abs(var_q_val) + 1e-12))
    log_min_q_val = float(np.log10(np.abs(min_q_val) + 1e-12))

    # Integral norms over voltage grid (step size dV)
    dv = V_GRID[1] - V_GRID[0]
    l1_val = float(np.sum(np.abs(delta)) * dv)
    l2_val = float(np.sqrt(np.sum(delta**2) * dv))

    return {
        "var_dqdv_100_10": var_val,
        "min_dqdv_100_10": min_val,
        "max_dqdv_100_10": max_val,
        "mean_dqdv_100_10": mean_val,
        "skew_dqdv_100_10": skew_val,
        "kurt_dqdv_100_10": kurt_val,
        "l1_dqdv_100_10": l1_val,
        "l2_dqdv_100_10": l2_val,
        "var_delta_q_100_10": var_q_val,
        "min_delta_q_100_10": min_q_val,
        "log_var_delta_q_100_10": log_var_q_val,
        "log_min_delta_q_100_10": log_min_q_val
    }


def main():
    parser = argparse.ArgumentParser(description="Physics-Informed Feature Engineering (dQ/dV) Pipeline")
    parser.add_argument("--proc-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--ts-file", type=str, default="battery_time_series.parquet", help="Input time-series file")
    parser.add_argument("--sum-file", type=str, default="battery_summary.parquet", help="Input summary file")
    parser.add_argument("--out-features", type=str, default="engineered_features.parquet", help="Output feature file")
    parser.add_argument("--out-dqdv", type=str, default="dqdv_curves.parquet", help="Output dQ/dV curves file")
    args = parser.parse_args()

    ts_path = os.path.join(args.proc_dir, args.ts_file)
    sum_path = os.path.join(args.proc_dir, args.sum_file)

    if not os.path.exists(ts_path) or not os.path.exists(sum_path):
        logger.error(f"Input files not found: {ts_path} or {sum_path}. Run Phase 1 first.")
        sys.exit(1)

    logger.info(f"Loading time-series dataset from {ts_path}...")
    ts_df = pd.read_parquet(ts_path)
    logger.info(f"Loading summary dataset from {sum_path}...")
    sum_df = pd.read_parquet(sum_path)

    cells = sorted(sum_df["cell_id"].unique())
    logger.info(f"Extracting physics-informed dQ/dV and early-cycle features across {len(cells)} cells...")

    feature_records = []
    dqdv_records = []

    for idx, cell_id in enumerate(cells):
        cell_ts = ts_df[ts_df["cell_id"] == cell_id]
        cell_sum = sum_df[sum_df["cell_id"] == cell_id].sort_values("cycle_number")

        if len(cell_ts) == 0 or len(cell_sum) == 0:
            continue

        # Extract cycle 10 and cycle 100 time series
        c10_ts = cell_ts[cell_ts["cycle_number"] == 10]
        c100_ts = cell_ts[cell_ts["cycle_number"] == 100]

        # Compute dQ/dV curves for cycle 10, 50, 100
        _, q10_interp, dqdv10_raw, dqdv10_smooth = compute_dqdv_curve(c10_ts)
        _, q100_interp, dqdv100_raw, dqdv100_smooth = compute_dqdv_curve(c100_ts)
        
        c50_ts = cell_ts[cell_ts["cycle_number"] == 50]
        if len(c50_ts) > 0:
            _, _, dqdv50_raw, dqdv50_smooth = compute_dqdv_curve(c50_ts)
        else:
            dqdv50_raw, dqdv50_smooth = np.zeros_like(V_GRID), np.zeros_like(V_GRID)

        # Record dQ/dV curves for downstream visualization (Cycles 10, 50, 100)
        for cyc_num, raw_arr, smooth_arr in [(10, dqdv10_raw, dqdv10_smooth), 
                                             (50, dqdv50_raw, dqdv50_smooth), 
                                             (100, dqdv100_raw, dqdv100_smooth)]:
            for v_val, r_val, s_val in zip(V_GRID, raw_arr, smooth_arr):
                dqdv_records.append({
                    "cell_id": cell_id,
                    "cycle_number": int(cyc_num),
                    "voltage_V": float(v_val),
                    "dqdv_raw": float(r_val),
                    "dqdv_smooth": float(s_val)
                })

        # Extract Peak 1 and Peak 2 features for Cycle 10 and Cycle 100
        peaks_10 = extract_peak_features(V_GRID, dqdv10_smooth)
        peaks_100 = extract_peak_features(V_GRID, dqdv100_smooth)

        # Compute Delta Q(V) and Delta(dQ/dV) statistical features
        delta_stats = compute_delta_dqdv_statistics(q100_interp, q10_interp, dqdv100_smooth, dqdv10_smooth)

        # Extract capacity and temperature trajectory features from cycle summary (Cycles 10 to 100)
        c10_sum = cell_sum[cell_sum["cycle_number"] == 10]
        c100_sum = cell_sum[cell_sum["cycle_number"] == 100]

        cap_10 = float(c10_sum["discharge_capacity_Ah"].iloc[0]) if len(c10_sum) > 0 else np.nan
        cap_100 = float(c100_sum["discharge_capacity_Ah"].iloc[0]) if len(c100_sum) > 0 else np.nan
        delta_cap = cap_100 - cap_10 if (not np.isnan(cap_100) and not np.isnan(cap_10)) else np.nan
        rel_cap_loss = delta_cap / cap_10 if (not np.isnan(delta_cap) and cap_10 > 0) else np.nan

        # Linear fit of capacity fade slope over cycles 10 to 100
        valid_sum = cell_sum[(cell_sum["cycle_number"] >= 10) & (cell_sum["cycle_number"] <= 100)]
        if len(valid_sum) >= 5:
            slope, _ = np.polyfit(valid_sum["cycle_number"], valid_sum["discharge_capacity_Ah"], 1)
        else:
            slope = np.nan

        # Temperature and voltage change features
        temp_10 = float(c10_sum["avg_temperature_C"].iloc[0]) if len(c10_sum) > 0 else np.nan
        temp_100 = float(c100_sum["avg_temperature_C"].iloc[0]) if len(c100_sum) > 0 else np.nan
        delta_temp = temp_100 - temp_10 if (not np.isnan(temp_100) and not np.isnan(temp_10)) else np.nan

        v_10 = float(c10_sum["avg_voltage_V"].iloc[0]) if len(c10_sum) > 0 else np.nan
        v_100 = float(c100_sum["avg_voltage_V"].iloc[0]) if len(c100_sum) > 0 else np.nan
        delta_avg_v = v_100 - v_10 if (not np.isnan(v_100) and not np.isnan(v_10)) else np.nan

        # Target label
        cycle_life = int(cell_sum["cycle_life"].iloc[0])
        batch_name = cell_sum["batch"].iloc[0]
        c0 = float(cell_sum["initial_capacity_Ah"].iloc[0])

        # Combine all features into record
        feat = {
            "cell_id": cell_id,
            "batch": batch_name,
            "cycle_life": cycle_life,
            "log_cycle_life": float(np.log10(cycle_life)),
            "initial_capacity_Ah": c0,
            "cap_10": cap_10,
            "cap_100": cap_100,
            "delta_cap_100_10": delta_cap,
            "rel_cap_loss_100_10": rel_cap_loss,
            "cap_fade_slope": float(slope),
            "delta_temp_100_10": delta_temp,
            "delta_avg_v_100_10": delta_avg_v,
            # Peak 1 features
            "v_peak1_10": peaks_10["v_peak1"],
            "v_peak1_100": peaks_100["v_peak1"],
            "h_peak1_10": peaks_10["h_peak1"],
            "h_peak1_100": peaks_100["h_peak1"],
            "delta_v_peak1_100_10": peaks_100["v_peak1"] - peaks_10["v_peak1"],
            "delta_h_peak1_100_10": peaks_100["h_peak1"] - peaks_10["h_peak1"],
            # Peak 2 features
            "v_peak2_10": peaks_10["v_peak2"],
            "v_peak2_100": peaks_100["v_peak2"],
            "h_peak2_10": peaks_10["h_peak2"],
            "h_peak2_100": peaks_100["h_peak2"],
            "delta_v_peak2_100_10": peaks_100["v_peak2"] - peaks_10["v_peak2"],
            "delta_h_peak2_100_10": peaks_100["h_peak2"] - peaks_10["h_peak2"],
        }
        feat.update(delta_stats)
        feature_records.append(feat)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(cells):
            logger.info(f"Processed {idx+1}/{len(cells)} cells...")

    feat_df = pd.DataFrame(feature_records)
    dqdv_df = pd.DataFrame(dqdv_records)

    out_feat_path = os.path.join(args.proc_dir, args.out_features)
    out_dqdv_path = os.path.join(args.proc_dir, args.out_dqdv)

    feat_df.to_parquet(out_feat_path, index=False)
    dqdv_df.to_parquet(out_dqdv_path, index=False)

    logger.info("="*60)
    logger.info("FEATURE ENGINEERING SUMMARY:")
    logger.info(f"Total cells featurized : {len(feat_df)}")
    logger.info(f"Total features per cell: {feat_df.shape[1] - 3}")  # exclude id, batch, label
    logger.info(f"Target cycle life range: {feat_df['cycle_life'].min()} - {feat_df['cycle_life'].max()} cycles")
    logger.info(f"Key physics feature correlations with log(cycle_life):")
    for col in ["var_dqdv_100_10", "min_dqdv_100_10", "delta_cap_100_10", "delta_h_peak1_100_10"]:
        if col in feat_df.columns:
            corr = feat_df[col].corr(feat_df["log_cycle_life"])
            logger.info(f"  {col:<22}: r = {corr:+.3f}")
    logger.info(f"Features dataset saved : {out_feat_path} ({os.path.getsize(out_feat_path)/1e6:.2f} MB)")
    logger.info(f"dQ/dV curves saved     : {out_dqdv_path} ({os.path.getsize(out_dqdv_path)/1e6:.2f} MB)")
    logger.info("="*60)
    logger.info("Feature engineering complete!")


if __name__ == "__main__":
    main()
