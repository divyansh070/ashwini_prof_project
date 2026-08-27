#!/usr/bin/env python3
"""
HybridoNet-Adapt Feature Preprocessing Pipeline (Tran et al., 2025).

Exact methodology:
1. Signal Filtering: Apply median filter (kernel=3) to Voltage (V), Current (I), and Capacity (Q) per cycle.
2. Statistical Extraction: Extract 6 features (Mean, Std, Min, Max, Var, Median) for each of the 3 signals -> 18 features/cycle.
3. Window Sampling: Uniformly sample 10 cycles from the first 30 cycles.
4. Output Tensor: Shape (10, 3, 6) per cell sample.
5. CRITICAL: Save RAW unscaled feature tensors. Min-Max scaling is fitted ONLY on training splits in train_hybrido.py.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import glob
from typing import Dict, List, Tuple, Optional

try:
    from scipy.signal import medfilt
except ImportError:
    def medfilt(x, kernel_size=3):
        # Fallback simple 1D median filter if scipy not installed
        k = kernel_size // 2
        pad_x = np.pad(x, (k, k), mode="edge")
        out = np.zeros_like(x)
        for i in range(len(x)):
            out[i] = np.median(pad_x[i:i + kernel_size])
        return out

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [HybridoPreprocess] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HybridoPreprocess")


def compute_cycle_statistics(
    voltage: np.ndarray,
    current: np.ndarray,
    capacity: np.ndarray,
    filter_kernel: int = 3
) -> np.ndarray:
    """
    Computes 6 statistical features for Voltage, Current, Capacity after median filtering.
    Features: [Mean, Std, Min, Max, Variance, Median]
    Returns:
        feature_matrix: Shape (3, 6)
    """
    signals = [voltage, current, capacity]
    feature_matrix = np.zeros((3, 6), dtype=np.float32)

    for i, sig in enumerate(signals):
        if len(sig) == 0:
            continue
        # 1. Apply median filter to remove sudden measurement spikes
        clean_sig = medfilt(sig, kernel_size=filter_kernel)
        
        # 2. Extract 6 statistical features
        mean_val = float(np.mean(clean_sig))
        std_val = float(np.std(clean_sig))
        min_val = float(np.min(clean_sig))
        max_val = float(np.max(clean_sig))
        var_val = float(np.var(clean_sig))
        med_val = float(np.median(clean_sig))

        feature_matrix[i] = [mean_val, std_val, min_val, max_val, var_val, med_val]

    return feature_matrix


def extract_cell_tensor(
    cycle_data: Dict[int, Dict[str, np.ndarray]],
    window_size: int = 30,
    num_samples: int = 10
) -> Optional[np.ndarray]:
    """
    Uniformly samples `num_samples` cycles from a `window_size` observation window.
    Returns:
        tensor: Shape (num_samples=10, channels=3, features=6)
    """
    available_cycles = sorted([c for c in cycle_data.keys() if c > 0])
    # Filter cycles within observation window (e.g., first 30 cycles)
    window_cycles = [c for c in available_cycles if c <= window_size]

    if len(window_cycles) < num_samples:
        # Fallback if 1-indexed cycles start slightly later or have gaps
        if len(available_cycles) >= num_samples:
            window_cycles = available_cycles[:window_size]
        else:
            return None

    # Uniform sampling of 10 cycles across the window
    idx_uniform = np.linspace(0, len(window_cycles) - 1, num_samples, dtype=int)
    selected_cycles = [window_cycles[i] for i in idx_uniform]

    cell_tensor = np.zeros((num_samples, 3, 6), dtype=np.float32)
    for step_idx, cyc in enumerate(selected_cycles):
        c_dict = cycle_data[cyc]
        v = np.array(c_dict.get("voltage", c_dict.get("V", [])))
        i = np.array(c_dict.get("current", c_dict.get("I", [])))
        q = np.array(c_dict.get("capacity", c_dict.get("Q", c_dict.get("Qd", []))))

        feat_3x6 = compute_cycle_statistics(v, i, q)
        cell_tensor[step_idx] = feat_3x6

    return cell_tensor


def process_parquet_dataset(
    parquet_path: str,
    domain_name: str,
    window_size: int = 30,
    num_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Processes a standardized battery parquet file and extracts raw unscaled feature tensors.
    """
    if not os.path.exists(parquet_path):
        logger.warning(f"File not found: {parquet_path}")
        return np.empty((0, 10, 3, 6)), np.empty((0,)), []

    df = pd.read_parquet(parquet_path)
    cell_ids = df["cell_id"].unique()
    
    tensors = []
    labels = []
    valid_ids = []

    for cid in cell_ids:
        cell_df = df[df["cell_id"] == cid]
        
        # Determine cycle life label (End of Life)
        if "cycle_life" in cell_df.columns and not cell_df["cycle_life"].isna().all():
            eol = float(cell_df["cycle_life"].dropna().iloc[0])
        elif "max_cycle" in cell_df.columns and not cell_df["max_cycle"].isna().all():
            eol = float(cell_df["max_cycle"].dropna().iloc[0])
        else:
            eol = float(cell_df["cycle_number"].max())

        # Collect raw time series per cycle
        cycle_data = {}
        for cyc_num, group in cell_df.groupby("cycle_number"):
            if cyc_num > window_size and len(cycle_data) >= window_size:
                break
            v = group["voltage"].values if "voltage" in group.columns else group.get("V", pd.Series()).values
            i = group["current"].values if "current" in group.columns else group.get("I", pd.Series()).values
            q = group["discharge_capacity"].values if "discharge_capacity" in group.columns else group.get("capacity", pd.Series()).values

            cycle_data[int(cyc_num)] = {"voltage": v, "current": i, "capacity": q}

        cell_tensor = extract_cell_tensor(cycle_data, window_size=window_size, num_samples=num_samples)
        if cell_tensor is not None and not np.isnan(cell_tensor).any():
            tensors.append(cell_tensor)
            labels.append(eol)
            valid_ids.append(f"{domain_name}_{cid}")

    if len(tensors) == 0:
        return np.empty((0, num_samples, 3, 6)), np.empty((0,)), []

    return np.array(tensors, dtype=np.float32), np.array(labels, dtype=np.float32), valid_ids


def main():
    parser = argparse.ArgumentParser(description="HybridoNet-Adapt Raw Feature Preprocessing")
    parser.add_argument("--data-dir", type=str, default="data/real_processed", help="Directory containing processed battery parquets")
    parser.add_argument("--output-dir", type=str, default="data/hybridonet/processed", help="Output directory for raw unscaled tensors")
    parser.add_argument("--window-size", type=int, default=30, help="Initial cycle window size")
    parser.add_argument("--num-samples", type=int, default=10, help="Uniformly sampled cycles in window")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Starting HybridoNet-Adapt raw feature extraction...")

    # Look for available battery datasets
    parquet_files = glob.glob(os.path.join(args.data_dir, "*.parquet"))
    if not parquet_files:
        logger.info(f"No parquets found in {args.data_dir}. Searching project root for available parquet files...")
        parquet_files = glob.glob("data/**/*.parquet", recursive=True)

    summary_records = []
    for p_file in parquet_files:
        domain = os.path.splitext(os.path.basename(p_file))[0]
        X, Y, cell_ids = process_parquet_dataset(p_file, domain, window_size=args.window_size, num_samples=args.num_samples)
        if len(X) > 0:
            out_file = os.path.join(args.output_dir, f"{domain}_raw_features.npz")
            np.savez_compressed(
                out_file,
                X=X, # Shape (N, 10, 3, 6) - RAW UNSCALED
                Y=Y, # Shape (N,) - RAW cycle life
                cell_ids=np.array(cell_ids)
            )
            logger.info(f"Saved {domain}: {X.shape[0]} cells, Tensor Shape: {X.shape} -> {out_file}")
            summary_records.append({"domain": domain, "cells": len(X), "tensor_shape": str(X.shape)})

    if not summary_records:
        logger.warning("No parquets processed. Ensure dataset download scripts have run.")
    else:
        logger.info("Raw feature extraction completed with zero global scaling leakage.")


if __name__ == "__main__":
    main()
