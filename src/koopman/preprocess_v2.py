#!/usr/bin/env python3
"""
Universal Domain Normalization Preprocessing Module (preprocess_v2.py).
Solves absolute voltage domain mismatch across LFP, LCO, and NMC chemistries by normalizing
the x-axis from Absolute Voltage to State of Charge (SOC) in [0.0, 1.0] and calculating
differential capacity embeddings as dQ/d(SOC).
This guarantees that invariant electrochemical phase-transition peaks align across chemistries,
providing a unified domain-invariant feature space for Koopman Neural Operators.

Outputs compressed tensor archives to `data/koopman_processed/`:
  - stanford_lfp_soc.npz
  - oxford_lco_soc.npz
  - calce_nmc_soc.npz
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
    format="%(asctime)s [%(levelname)s] [KoopmanPreprocessV2] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("KoopmanPreprocessV2")

SEED = 42
np.random.seed(SEED)

# Universal SOC Grid: 200 uniform points from SOC=0.0 (fully discharged) to SOC=1.0 (fully charged)
L_GRID = 200
SOC_GRID = np.linspace(0.0, 1.0, L_GRID)
CYCLES_EVAL = np.arange(10, 101, 2)  # 46 early cycles (Cycles 10 to 100)

CHEMISTRY_BOUNDS = {
    "LFP": {"v_min": 2.05, "v_max": 3.50},
    "LCO": {"v_min": 2.70, "v_max": 4.20},
    "NMC": {"v_min": 2.70, "v_max": 4.20}
}


def compute_universal_dq_dsoc(v_raw: np.ndarray, q_raw: np.ndarray, chem: str) -> np.ndarray:
    """
    Computes Savitzky-Golay smoothed dQ/d(SOC) on the universal SOC grid s in [0.0, 1.0].
    Phase-transition peaks are aligned by mapping absolute voltage to fractional State of Charge.
    """
    if len(v_raw) < 10 or len(q_raw) < 10:
        return np.zeros(L_GRID, dtype=np.float32)

    b = CHEMISTRY_BOUNDS.get(chem, {"v_min": 2.50, "v_max": 4.20})
    v_min, v_max = b["v_min"], b["v_max"]

    # Normalize voltage to fractional State of Charge (SOC) in [0.0, 1.0]
    soc_raw = (v_raw - v_min) / (v_max - v_min)
    soc_raw = np.clip(soc_raw, 0.0, 1.0)

    # Sort strictly ascending by SOC
    sort_idx = np.argsort(soc_raw)
    soc_sorted = soc_raw[sort_idx]
    q_sorted = q_raw[sort_idx]

    # Ensure uniqueness of SOC points
    soc_uniq, uniq_idx = np.unique(soc_sorted, return_index=True)
    q_uniq = q_sorted[uniq_idx]

    if len(soc_uniq) < 5:
        return np.zeros(L_GRID, dtype=np.float32)

    try:
        f_q = interp1d(soc_uniq, q_uniq, kind="linear", bounds_error=False, fill_value="extrapolate")
        q_interp = f_q(SOC_GRID)
    except Exception:
        return np.zeros(L_GRID, dtype=np.float32)

    # Numerical derivative dQ/d(SOC)
    dq_dsoc = np.gradient(q_interp, SOC_GRID)

    # Savitzky-Golay smoothing (window=15, polyorder=3)
    window = min(15, len(dq_dsoc) - 2 if len(dq_dsoc) % 2 == 1 else len(dq_dsoc) - 3)
    if window >= 5:
        dq_dsoc_smooth = savgol_filter(dq_dsoc, window_length=window, polyorder=3)
    else:
        dq_dsoc_smooth = dq_dsoc

    return dq_dsoc_smooth.astype(np.float32)


def preprocess_dataset(df: pd.DataFrame, chem: str, out_path: str):
    """
    Constructs universal SOC normalized 2D matrices (num_cells, 46, 200) for Koopman Neural Operators.
    """
    logger.info(f"Universal SOC Normalization for {chem} dataset (Rows: {len(df)})...")
    cells = sorted(df["cell_id"].unique())
    N = len(cells)

    matrices_soc = np.zeros((N, len(CYCLES_EVAL), L_GRID), dtype=np.float32)
    y_eol = np.zeros(N, dtype=np.float32)

    for idx, cell in enumerate(cells):
        cell_df = df[df["cell_id"] == cell]
        eol_val = cell_df["cycle_life"].iloc[0]
        y_eol[idx] = eol_val

        for c_idx, cyc in enumerate(CYCLES_EVAL):
            cyc_df = cell_df[cell_df["cycle_number"] == cyc]
            if len(cyc_df) < 5:
                dq_dsoc = np.zeros(L_GRID, dtype=np.float32)
            else:
                dq_dsoc = compute_universal_dq_dsoc(
                    cyc_df["voltage_V"].values,
                    cyc_df["capacity_Ah"].values,
                    chem=chem
                )
            matrices_soc[idx, c_idx, :] = dq_dsoc

    # Global standardization across the dataset
    mean_val = np.mean(matrices_soc)
    std_val = np.std(matrices_soc) + 1e-8
    matrices_soc_norm = (matrices_soc - mean_val) / std_val

    np.savez_compressed(
        out_path,
        matrices_soc=matrices_soc_norm,
        y_eol=y_eol,
        cells=cells,
        chemistry=chem,
        soc_grid=SOC_GRID,
        mean_val=mean_val,
        std_val=std_val
    )
    logger.info(f"Saved universal {chem} SOC preprocessed tensor -> {out_path} (Shape: {matrices_soc_norm.shape})")
    return matrices_soc_norm, y_eol


def main():
    parser = argparse.ArgumentParser(description="Universal Domain Normalization for Koopman & DANN")
    parser.add_argument("--in-dir", type=str, default="data/patchtst_raw", help="Input raw parquet directory")
    parser.add_argument("--out-dir", type=str, default="data/koopman_processed", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("======================================================================")
    logger.info("UNIVERSAL DOMAIN NORMALIZATION: ABSOLUTE VOLTAGE -> SOC [0.0, 1.0]")
    logger.info("======================================================================")

    lfp_file = os.path.join(args.in_dir, "stanford_lfp.parquet")
    lco_file = os.path.join(args.in_dir, "oxford_lco.parquet")
    nmc_file = os.path.join(args.in_dir, "calce_nmc.parquet")

    if os.path.exists(lfp_file):
        df_lfp = pd.read_parquet(lfp_file)
        preprocess_dataset(df_lfp, "LFP", os.path.join(args.out_dir, "stanford_lfp_soc.npz"))
    else:
        logger.warning(f"Missing {lfp_file}. Run download_datasets.py first.")

    if os.path.exists(lco_file):
        df_lco = pd.read_parquet(lco_file)
        preprocess_dataset(df_lco, "LCO", os.path.join(args.out_dir, "oxford_lco_soc.npz"))

    if os.path.exists(nmc_file):
        df_nmc = pd.read_parquet(nmc_file)
        preprocess_dataset(df_nmc, "NMC", os.path.join(args.out_dir, "calce_nmc_soc.npz"))

    logger.info("======================================================================")
    logger.info("UNIVERSAL DOMAIN NORMALIZATION COMPLETED SUCCESSFULLY")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
