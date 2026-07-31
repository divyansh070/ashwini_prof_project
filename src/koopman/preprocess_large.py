#!/usr/bin/env python3
"""
Large-Scale Universal SOC Preprocessing Module (src/koopman/preprocess_large.py) - LEAKAGE-FREE.
Maps dQ/dV curves from the TRI 224-cell dataset and HUST 77-cell dataset to the
Universal State of Charge (SOC) domain [0.0, 1.0] (L=200 uniform bins) without applying
any global statistical standardization.

CRITICAL LEAKAGE PREVENTIONS:
  - Preserves RAW unscaled universal dQ/d(SOC) matrices.
  - Zero statistical standardization (mean/std/StandardScaler) is applied across the dataset before splitting.
  - All normalization parameters must be fit strictly within training folds during 5-Fold GroupKFold CV.

Outputs compressed tensor archives to `data/large_scale_processed/`:
  - tri_stanford_224_soc.npz
  - hust_77_soc.npz
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
    format="%(asctime)s [%(levelname)s] [PreprocessLarge] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PreprocessLarge")

SEED = 42
np.random.seed(SEED)

L_GRID = 200
SOC_GRID = np.linspace(0.0, 1.0, L_GRID)
CYCLES_EVAL = np.arange(10, 101, 2)  # 46 early cycles (Cycles 10 to 100)

CHEMISTRY_BOUNDS = {
    "LFP": {"v_min": 2.00, "v_max": 3.65}
}


def compute_universal_dq_dsoc(v_raw: np.ndarray, q_raw: np.ndarray, chem: str = "LFP") -> np.ndarray:
    """
    Computes Savitzky-Golay smoothed dQ/d(SOC) on the universal SOC grid s in [0.0, 1.0].
    """
    if len(v_raw) < 10 or len(q_raw) < 10:
        return np.zeros(L_GRID, dtype=np.float32)

    b = CHEMISTRY_BOUNDS.get(chem, {"v_min": 2.00, "v_max": 3.65})
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

    # Savitzky-Golay smoothing
    window = min(15, len(dq_dsoc) - 2 if len(dq_dsoc) % 2 == 1 else len(dq_dsoc) - 3)
    if window >= 5:
        dq_dsoc_smooth = savgol_filter(dq_dsoc, window_length=window, polyorder=3)
    else:
        dq_dsoc_smooth = dq_dsoc

    return dq_dsoc_smooth.astype(np.float32)


def preprocess_large_dataset(df: pd.DataFrame, dataset_name: str, out_path: str):
    """
    Constructs universal SOC raw 2D matrices (num_cells, 46, 200) for large-scale evaluation.
    No global dataset standardizations are applied to guarantee zero scaling leakage.
    """
    logger.info(f"Universal SOC Normalization for {dataset_name} (Rows: {len(df)})...")
    cells = sorted(df["cell_id"].unique())
    N = len(cells)

    matrices_soc_raw = np.zeros((N, len(CYCLES_EVAL), L_GRID), dtype=np.float32)
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
                    chem="LFP"
                )
            matrices_soc_raw[idx, c_idx, :] = dq_dsoc

    # LEAKAGE-FREE ARCHIVING: Save raw unscaled features.
    # Standard deviation/mean scaling MUST occur dynamically within each Train fold.
    np.savez_compressed(
        out_path,
        matrices_soc=matrices_soc_raw,
        y_eol=y_eol,
        cells=cells,
        dataset_name=dataset_name,
        soc_grid=SOC_GRID
    )
    logger.info(f"Saved raw universal SOC preprocessed tensor -> {out_path} (Shape: {matrices_soc_raw.shape})")
    return matrices_soc_raw, y_eol


def main():
    parser = argparse.ArgumentParser(description="Large-Scale Universal SOC Preprocessing (TRI 224 & HUST 77)")
    parser.add_argument("--in-dir", type=str, default="data/large_scale_raw", help="Input raw parquet directory")
    parser.add_argument("--out-dir", type=str, default="data/large_scale_processed", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("======================================================================")
    logger.info("LARGE-SCALE UNIVERSAL DOMAIN NORMALIZATION: VOLTAGE -> SOC [0.0, 1.0]")
    logger.info("======================================================================")

    tri_file = os.path.join(args.in_dir, "tri_stanford_224.parquet")
    hust_file = os.path.join(args.in_dir, "hust_77.parquet")

    if os.path.exists(tri_file):
        df_tri = pd.read_parquet(tri_file)
        preprocess_large_dataset(df_tri, "TRI_Stanford_224", os.path.join(args.out_dir, "tri_stanford_224_soc.npz"))
    else:
        logger.warning(f"Missing {tri_file}. Run download_large_datasets.py first.")

    if os.path.exists(hust_file):
        df_hust = pd.read_parquet(hust_file)
        preprocess_large_dataset(df_hust, "HUST_77", os.path.join(args.out_dir, "hust_77_soc.npz"))

    logger.info("======================================================================")
    logger.info("LARGE-SCALE UNIVERSAL SOC PREPROCESSING COMPLETED (ZERO LEAKAGE)")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
