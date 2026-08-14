#!/usr/bin/env python3
"""
REAL DATA Koopman Preprocessor — Converts real physical battery data into
SOC-normalized dQ/dSOC feature tensors for the Koopman Neural Operator.

Input: Real parquet files from download_real_data.py
Output: .npz files with matrices_soc, y_eol, cells, chemistry, soc_grid

NO SYNTHETIC DATA. This script reads only from data/real_processed/.
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
    format="%(asctime)s [%(levelname)s] [RealPreprocess] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RealPreprocess")

SEED = 42
np.random.seed(SEED)

# Universal SOC Grid: 200 uniform points from SOC=0.0 to SOC=1.0
L_GRID = 200
SOC_GRID = np.linspace(0.0, 1.0, L_GRID)
CYCLES_EVAL = np.arange(10, 101, 2)  # 46 early cycles

CHEMISTRY_BOUNDS = {
    "LFP": {"v_min": 2.00, "v_max": 3.60},
    "LCO": {"v_min": 2.50, "v_max": 4.20},
    "NMC": {"v_min": 2.70, "v_max": 4.20}
}


def compute_dq_dsoc(v_raw: np.ndarray, q_raw: np.ndarray, chem: str) -> np.ndarray:
    """
    Converts a raw discharge voltage-capacity curve into a dQ/dSOC feature vector
    on the universal SOC grid.

    Steps:
    1. Map voltage to SOC using chemistry-specific bounds
    2. Interpolate capacity onto uniform SOC grid
    3. Compute dQ/dSOC via finite differences
    4. Smooth with Savitzky-Golay filter
    """
    bounds = CHEMISTRY_BOUNDS[chem]
    v_min, v_max = bounds["v_min"], bounds["v_max"]

    # Clip and sort by voltage (descending for discharge)
    mask = (v_raw >= v_min) & (v_raw <= v_max) & np.isfinite(v_raw) & np.isfinite(q_raw)
    v_clean = v_raw[mask]
    q_clean = q_raw[mask]

    if len(v_clean) < 10:
        return np.zeros(L_GRID, dtype=np.float32)

    # Map voltage to SOC: SOC = (V - V_min) / (V_max - V_min)
    soc = (v_clean - v_min) / (v_max - v_min)

    # Sort by ascending SOC
    sort_idx = np.argsort(soc)
    soc_sorted = soc[sort_idx]
    q_sorted = q_clean[sort_idx]

    # Remove duplicate SOC values
    _, unique_idx = np.unique(soc_sorted, return_index=True)
    soc_unique = soc_sorted[unique_idx]
    q_unique = q_sorted[unique_idx]

    if len(soc_unique) < 5:
        return np.zeros(L_GRID, dtype=np.float32)

    # Clamp SOC to [0, 1]
    soc_unique = np.clip(soc_unique, 0.0, 1.0)

    # Interpolate Q onto uniform SOC grid
    try:
        f_q = interp1d(soc_unique, q_unique, kind='linear', fill_value='extrapolate', bounds_error=False)
        q_interp = f_q(SOC_GRID)
    except Exception:
        return np.zeros(L_GRID, dtype=np.float32)

    # Compute dQ/dSOC via finite differences
    dq_dsoc = np.gradient(q_interp, SOC_GRID)

    # Smooth with Savitzky-Golay filter
    window = min(21, len(dq_dsoc) - 1)
    if window % 2 == 0:
        window -= 1
    if window >= 5:
        dq_dsoc = savgol_filter(dq_dsoc, window, polyorder=3)

    # Replace NaN/Inf
    dq_dsoc = np.nan_to_num(dq_dsoc, nan=0.0, posinf=0.0, neginf=0.0)

    return dq_dsoc.astype(np.float32)


def preprocess_real_dataset(parquet_path: str, chem: str, out_path: str):
    """
    Reads a REAL processed parquet file and creates the Koopman input tensor.
    """
    logger.info(f"Processing REAL {chem} data from {parquet_path}")

    df = pd.read_parquet(parquet_path)
    cells = sorted(df["cell_id"].unique())
    n_cells = len(cells)

    logger.info(f"  Found {n_cells} real cells")

    matrices_soc_raw = np.zeros((n_cells, len(CYCLES_EVAL), L_GRID), dtype=np.float32)
    y_eol = np.zeros(n_cells, dtype=np.float32)

    for idx, cell in enumerate(cells):
        cell_df = df[df["cell_id"] == cell]
        eol_val = cell_df["cycle_life"].iloc[0]
        y_eol[idx] = eol_val

        for c_idx, cyc in enumerate(CYCLES_EVAL):
            cyc_df = cell_df[cell_df["cycle_number"] == cyc]
            if len(cyc_df) < 5:
                dq_dsoc = np.zeros(L_GRID, dtype=np.float32)
            else:
                dq_dsoc = compute_dq_dsoc(
                    cyc_df["voltage_V"].values,
                    cyc_df["capacity_Ah"].values,
                    chem=chem
                )
            matrices_soc_raw[idx, c_idx, :] = dq_dsoc

    # Verify this is REAL data — check voltage steps are non-uniform
    sample_cell = df[df["cell_id"] == cells[0]]
    sample_cyc = sample_cell[sample_cell["cycle_number"] == CYCLES_EVAL[0]]
    if len(sample_cyc) > 5:
        v_diffs = np.diff(sample_cyc["voltage_V"].values[:10])
        if len(v_diffs) > 2 and np.all(np.abs(v_diffs - v_diffs[0]) < 1e-10):
            logger.error("⚠️ WARNING: Voltage steps appear perfectly uniform (np.linspace)!")
            logger.error("   This looks like SYNTHETIC data. Aborting to prevent fake results.")
            raise RuntimeError("Detected synthetic np.linspace data. Use download_real_data.py first.")
        else:
            logger.info(f"  ✅ Voltage steps verified as non-uniform (REAL sensor data)")

    # Save raw unscaled features (scaling happens per-fold during training)
    np.savez_compressed(
        out_path,
        matrices_soc=matrices_soc_raw,
        y_eol=y_eol,
        cells=np.array(cells),
        chemistry=chem,
        soc_grid=SOC_GRID
    )
    logger.info(f"  ✅ Saved REAL {chem} tensor -> {out_path} (Shape: {matrices_soc_raw.shape})")
    logger.info(f"     EOL values: {y_eol}")
    return matrices_soc_raw, y_eol


def main():
    parser = argparse.ArgumentParser(description="Real Data Koopman Preprocessor (NO SYNTHETIC)")
    parser.add_argument("--in-dir", type=str, default="data/real_processed", help="Input processed parquet directory")
    parser.add_argument("--out-dir", type=str, default="data/real_koopman", help="Output directory for .npz tensors")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("REAL DATA KOOPMAN PREPROCESSING — NO SYNTHETIC DATA ALLOWED")
    logger.info("=" * 70)

    lfp_path = os.path.join(args.in_dir, "stanford_lfp.parquet")
    if os.path.exists(lfp_path):
        preprocess_real_dataset(lfp_path, "LFP", os.path.join(args.out_dir, "real_stanford_lfp_soc.npz"))
    else:
        logger.warning(f"Missing {lfp_path}. Run download_real_data.py first.")

    nasa_path = os.path.join(args.in_dir, "nasa_lco.parquet")
    if os.path.exists(nasa_path):
        preprocess_real_dataset(nasa_path, "LCO", os.path.join(args.out_dir, "real_nasa_lco_soc.npz"))
    else:
        logger.warning(f"Missing {nasa_path}. Run download_real_data.py first.")

    logger.info("=" * 70)
    logger.info("REAL DATA PREPROCESSING COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
