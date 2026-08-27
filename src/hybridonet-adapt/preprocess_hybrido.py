#!/usr/bin/env python3
"""
HybridoNet-Adapt Feature Preprocessing Pipeline (Tran et al., 2025).

Exact methodology:
1. Signal Filtering: 1D median filter (kernel=3) on Voltage (V), Current (I), and Capacity (Q) per cycle.
2. Statistical Extraction: 6 features (Mean, Std, Min, Max, Var, Median) for each of the 3 signals -> 18 features/cycle.
3. Observation Window: 30-cycle observation window, uniformly sampling 10 cycles.
4. Target Formulation: RUL = EOL - current_cycle (historical-data-independent RUL prediction).
5. Data Leakage Prevention: Saves RAW unscaled feature tensors and individual cell_ids for strict cell-level splitting.
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
        feature_matrix: Shape (3, 6) -> 18 features
    """
    signals = [voltage, current, capacity]
    feature_matrix = np.zeros((3, 6), dtype=np.float32)

    for i, sig in enumerate(signals):
        if len(sig) == 0:
            continue
        clean_sig = medfilt(sig, kernel_size=filter_kernel)
        
        mean_val = float(np.mean(clean_sig))
        std_val = float(np.std(clean_sig))
        min_val = float(np.min(clean_sig))
        max_val = float(np.max(clean_sig))
        var_val = float(np.var(clean_sig))
        med_val = float(np.median(clean_sig))

        feature_matrix[i] = [mean_val, std_val, min_val, max_val, var_val, med_val]

    return feature_matrix


def extract_window_tensor(
    cycle_data: Dict[int, Dict[str, np.ndarray]],
    window_cycles: List[int],
    num_samples: int = 10
) -> Optional[np.ndarray]:
    """
    Uniformly samples `num_samples` (10) cycles from a list of window cycles (e.g. 30 cycles).
    Returns:
        tensor: Shape (10, 3, 6)
    """
    if len(window_cycles) < num_samples:
        return None

    # Uniform 10-cycle sampling across the 30-cycle window
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


def extract_cell_samples(
    cycle_data: Dict[int, Dict[str, np.ndarray]],
    eol: float,
    window_size: int = 30,
    stride: int = 30,
    num_samples: int = 10,
    rolling: bool = True
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Extracts 10x3x6 tensors and true RUL targets (RUL = EOL - current_cycle).
    If rolling=True: extracts multiple 30-cycle observation windows (stride=30 gives non-overlapping rolling windows, stride<30 gives sliding windows).
    If rolling=False: extracts only the first 30-cycle window (RUL = EOL - 30).
    """
    available_cycles = sorted([c for c in cycle_data.keys() if c > 0])
    if len(available_cycles) < window_size:
        return [], []

    samples = []
    ruls = []

    if not rolling:
        # Single early-life window [1..30]
        window = [c for c in available_cycles if c <= window_size]
        if len(window) >= num_samples:
            tensor = extract_window_tensor(cycle_data, window, num_samples)
            if tensor is not None and not np.isnan(tensor).any():
                samples.append(tensor)
                ruls.append(max(0.0, float(eol - window[-1])))
        return samples, ruls

    # Rolling windows [t-window_size+1 .. t]
    for end_idx in range(window_size, len(available_cycles) + 1, stride):
        window = available_cycles[end_idx - window_size:end_idx]
        current_cycle = window[-1]
        
        # Stop window extraction once current cycle reaches or exceeds EOL (current_cycle >= eol)
        if current_cycle >= eol:
            break

        tensor = extract_window_tensor(cycle_data, window, num_samples)
        if tensor is not None and not np.isnan(tensor).any():
            true_rul = float(eol - current_cycle)
            samples.append(tensor)
            ruls.append(true_rul)

    return samples, ruls


def process_parquet_dataset(
    parquet_path: str,
    domain_name: str,
    window_size: int = 30,
    stride: int = 30,
    num_samples: int = 10,
    rolling: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Processes standardized battery parquets and extracts raw unscaled feature tensors + RUL labels + cell IDs.
    Returns:
        X: (N, 10, 3, 6)
        Y: (N,)
        sample_ids: List of window-level identifiers
        cell_ids: List of battery-level cell identifiers (for cell-level grouping)
    """
    if not os.path.exists(parquet_path):
        logger.warning(f"File not found: {parquet_path}")
        return np.empty((0, num_samples, 3, 6)), np.empty((0,)), [], []

    df = pd.read_parquet(parquet_path)
    cell_unique = df["cell_id"].unique()
    
    all_tensors = []
    all_ruls = []
    all_sample_ids = []
    all_cell_ids = []

    for cid in cell_unique:
        cell_df = df[df["cell_id"] == cid]
        
        # Determine cell End of Life (EOL)
        if "cycle_life" in cell_df.columns and not cell_df["cycle_life"].isna().all():
            eol = float(cell_df["cycle_life"].dropna().iloc[0])
        elif "max_cycle" in cell_df.columns and not cell_df["max_cycle"].isna().all():
            eol = float(cell_df["max_cycle"].dropna().iloc[0])
        else:
            eol = float(cell_df["cycle_number"].max())

        cycle_data = {}
        for cyc_num, group in cell_df.groupby("cycle_number"):
            v = group["voltage"].values if "voltage" in group.columns else group.get("V", pd.Series()).values
            i = group["current"].values if "current" in group.columns else group.get("I", pd.Series()).values
            q = group["discharge_capacity"].values if "discharge_capacity" in group.columns else group.get("capacity", pd.Series()).values
            cycle_data[int(cyc_num)] = {"voltage": v, "current": i, "capacity": q}

        tensors, ruls = extract_cell_samples(
            cycle_data, eol, window_size=window_size, stride=stride, num_samples=num_samples, rolling=rolling
        )

        cell_global_id = f"{domain_name}_{cid}"
        for s_idx, (t_mat, r_val) in enumerate(zip(tensors, ruls)):
            all_tensors.append(t_mat)
            all_ruls.append(r_val)
            all_sample_ids.append(f"{cell_global_id}_w{s_idx}")
            all_cell_ids.append(cell_global_id)

    if len(all_tensors) == 0:
        return np.empty((0, num_samples, 3, 6)), np.empty((0,)), [], []

    return np.array(all_tensors, dtype=np.float32), np.array(all_ruls, dtype=np.float32), all_sample_ids, all_cell_ids


def main():
    parser = argparse.ArgumentParser(description="HybridoNet-Adapt Rolling RUL Preprocessing")
    parser.add_argument("--data-dir", type=str, default="data/real_processed", help="Directory containing processed battery parquets")
    parser.add_argument("--output-dir", type=str, default="data/hybridonet/processed", help="Output directory for raw unscaled tensors")
    parser.add_argument("--window-size", type=int, default=30, help="Observation window size (cycles)")
    parser.add_argument("--stride", type=int, default=30, help="Window stride for rolling RUL samples (30=non-overlapping)")
    parser.add_argument("--num-samples", type=int, default=10, help="Uniformly sampled cycles in window")
    parser.add_argument("--early-only", action="store_true", help="Extract only early-cycle (first window) rather than full rolling RUL")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rolling = not args.early_only
    logger.info(f"Extracting HybridoNet features (Rolling RUL mode: {rolling}, Window: {args.window_size}, Stride: {args.stride})...")

    parquet_files = glob.glob(os.path.join(args.data_dir, "*.parquet"))
    if not parquet_files:
        parquet_files = glob.glob("data/**/*.parquet", recursive=True)

    for p_file in parquet_files:
        domain = os.path.splitext(os.path.basename(p_file))[0]
        X, Y, sample_ids, cell_ids = process_parquet_dataset(
            p_file, domain, window_size=args.window_size, stride=args.stride, num_samples=args.num_samples, rolling=rolling
        )
        if len(X) > 0:
            out_file = os.path.join(args.output_dir, f"{domain}_raw_features.npz")
            np.savez_compressed(
                out_file,
                X=X,                     # (N_samples, 10, 3, 6) - RAW UNSCALED
                Y=Y,                     # (N_samples,) - TRUE RUL (EOL - current_cycle)
                sample_ids=np.array(sample_ids),
                cell_ids=np.array(cell_ids) # Battery-level grouping for zero-leakage splits
            )
            n_cells = len(np.unique(cell_ids))
            logger.info(f"Saved {domain}: {len(X)} samples across {n_cells} unique cells, RUL range: [{Y.min():.0f}, {Y.max():.0f}] cyc -> {out_file}")

    logger.info("Raw feature extraction completed with zero global scaling leakage.")


if __name__ == "__main__":
    main()
