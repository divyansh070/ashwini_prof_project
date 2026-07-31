#!/usr/bin/env python3
"""
Multi-Chemistry Data Preprocessing & Patching Module for PatchTST (LEAKAGE-FREE).
Solves the chemistry cross-compatibility challenge by normalizing distinct voltage plateaus
into a unified Normalized State of Discharge (SOD) grid u in [0.0, 1.0].

CRITICAL LEAKAGE PREVENTIONS:
  - Preserves RAW unscaled universal dQ/du matrices.
  - Zero statistical standardization is applied across the dataset before splitting.
  - Standard deviation/mean scaling MUST occur dynamically within each Train fold.

Output tensors saved to `data/patchtst_processed/`:
  - stanford_lfp_patches.npz
  - oxford_lco_patches.npz
  - calce_nmc_patches.npz
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PatchTSTPreprocess] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PatchTSTPreprocess")

SEED = 42
np.random.seed(SEED)

# Normalized grid: L=200 uniform SOD bins
L_GRID = 200
SOD_GRID = np.linspace(0.0, 1.0, L_GRID)
CYCLES_CNN = np.arange(10, 101, 2)  # 46 early cycles

# Chemistry voltage bounds (for SOD normalization)
CHEMISTRY_BOUNDS = {
    "LFP": {"v_min": 2.05, "v_max": 3.50},
    "LCO": {"v_min": 2.70, "v_max": 4.20},
    "NMC": {"v_min": 2.70, "v_max": 4.20}
}

# Patching parameters (Nie et al., ICLR 2023 PatchTST)
PATCH_LEN = 16
STRIDE = 8


def normalize_voltage_to_sod(v_array: np.ndarray, chem: str) -> np.ndarray:
    """
    Normalizes chemistry-specific voltage array into State of Discharge (SOD) grid u in [0.0, 1.0].
    """
    b = CHEMISTRY_BOUNDS.get(chem, {"v_min": 2.50, "v_max": 4.20})
    v_min, v_max = b["v_min"], b["v_max"]
    u = (v_array - v_min) / (v_max - v_min)
    return np.clip(u, 0.0, 1.0)


def compute_normalized_dqdu(v_raw: np.ndarray, q_raw: np.ndarray, chem: str) -> np.ndarray:
    """
    Computes Savitzky-Golay smoothed dQ/du curve on uniform SOD grid u in [0.0, 1.0].
    """
    if len(v_raw) < 10:
        return np.zeros(L_GRID, dtype=np.float32)

    u_raw = normalize_voltage_to_sod(v_raw, chem)

    # Sort strictly descending for discharge curve interpolation
    sort_idx = np.argsort(u_raw)[::-1]
    u_sorted = u_raw[sort_idx]
    q_sorted = q_raw[sort_idx]

    # Ensure uniqueness
    u_uniq, uniq_idx = np.unique(u_sorted, return_index=True)
    q_uniq = q_sorted[uniq_idx]

    if len(u_uniq) < 5:
        return np.zeros(L_GRID, dtype=np.float32)

    try:
        f_q = interp1d(u_uniq, q_uniq, kind="linear", bounds_error=False, fill_value="extrapolate")
        q_interp = f_q(SOD_GRID)
    except Exception:
        return np.zeros(L_GRID, dtype=np.float32)

    # Numerical derivative dQ/du
    dq_du = np.gradient(q_interp, SOD_GRID)

    # Savitzky-Golay smoothing
    window = min(15, len(dq_du) - 2 if len(dq_du) % 2 == 1 else len(dq_du) - 3)
    if window >= 5:
        dq_du_smooth = savgol_filter(dq_du, window_length=window, polyorder=3)
    else:
        dq_du_smooth = dq_du

    return dq_du_smooth.astype(np.float32)


def patch_sequence(seq: np.ndarray, patch_len: int = PATCH_LEN, stride: int = STRIDE) -> np.ndarray:
    """
    Segments 1D sequence of length L=200 into patches of length P=16 with stride S=8.
    Returns array of shape (num_patches, patch_len).
    """
    L = len(seq)
    num_patches = (L - patch_len) // stride + 1
    patches = np.zeros((num_patches, patch_len), dtype=np.float32)
    for i in range(num_patches):
        start = i * stride
        patches[i, :] = seq[start : start + patch_len]
    return patches


def preprocess_dataset(df: pd.DataFrame, chem: str, out_path: str):
    """
    Constructs 2D matrices (num_cycles=46, L=200) and PatchTST tensor arrays for a dataset.
    No global dataset scaling is applied to prevent scaling leakage.
    """
    logger.info(f"Processing {chem} dataset (Rows: {len(df)})...")
    cells = sorted(df["cell_id"].unique())
    N = len(cells)

    matrices_2d_raw = np.zeros((N, len(CYCLES_CNN), L_GRID), dtype=np.float32)
    num_patches = (L_GRID - PATCH_LEN) // STRIDE + 1
    patches_4d_raw = np.zeros((N, len(CYCLES_CNN), num_patches, PATCH_LEN), dtype=np.float32)
    y_eol = np.zeros(N, dtype=np.float32)

    for idx, cell in enumerate(cells):
        cell_df = df[df["cell_id"] == cell]
        eol_val = cell_df["cycle_life"].iloc[0]
        y_eol[idx] = eol_val

        for c_idx, cyc in enumerate(CYCLES_CNN):
            cyc_df = cell_df[cell_df["cycle_number"] == cyc]
            if len(cyc_df) < 5:
                dqdu = np.zeros(L_GRID, dtype=np.float32)
            else:
                dqdu = compute_normalized_dqdu(
                    cyc_df["voltage_V"].values,
                    cyc_df["capacity_Ah"].values,
                    chem=chem
                )
            matrices_2d_raw[idx, c_idx, :] = dqdu
            patches_4d_raw[idx, c_idx, :, :] = patch_sequence(dqdu, PATCH_LEN, STRIDE)

    # LEAKAGE-FREE ARCHIVING: Save raw unscaled features.
    np.savez_compressed(
        out_path,
        matrices_2d=matrices_2d_raw,
        patches_4d=patches_4d_raw,
        y_eol=y_eol,
        cells=cells,
        chemistry=chem,
        sod_grid=SOD_GRID
    )
    logger.info(f"Saved raw {chem} preprocessed patches -> {out_path} (Shape: {patches_4d_raw.shape})")
    return matrices_2d_raw, patches_4d_raw, y_eol


def main():
    parser = argparse.ArgumentParser(description="Preprocess & Patch Multi-Chemistry Battery Datasets (Leakage-Free)")
    parser.add_argument("--in-dir", type=str, default="data/patchtst_raw", help="Input raw directory")
    parser.add_argument("--out-dir", type=str, default="data/patchtst_processed", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("======================================================================")
    logger.info("PREPROCESSING & PATCHING MULTI-CHEMISTRY TIME SERIES (PatchTST)")
    logger.info("======================================================================")

    lfp_file = os.path.join(args.in_dir, "stanford_lfp.parquet")
    lco_file = os.path.join(args.in_dir, "oxford_lco.parquet")
    nmc_file = os.path.join(args.in_dir, "calce_nmc.parquet")

    if os.path.exists(lfp_file):
        df_lfp = pd.read_parquet(lfp_file)
        preprocess_dataset(df_lfp, "LFP", os.path.join(args.out_dir, "stanford_lfp_patches.npz"))
    else:
        logger.warning(f"Missing {lfp_file}. Run download_datasets.py first.")

    if os.path.exists(lco_file):
        df_lco = pd.read_parquet(lco_file)
        preprocess_dataset(df_lco, "LCO", os.path.join(args.out_dir, "oxford_lco_patches.npz"))

    if os.path.exists(nmc_file):
        df_nmc = pd.read_parquet(nmc_file)
        preprocess_dataset(df_nmc, "NMC", os.path.join(args.out_dir, "calce_nmc_patches.npz"))

    logger.info("======================================================================")
    logger.info("PATCHTST PREPROCESSING COMPLETED SUCCESSFULLY (ZERO LEAKAGE)")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
