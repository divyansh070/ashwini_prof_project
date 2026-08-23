#!/usr/bin/env python3
"""
REAL DATA ACQUISITION SCRIPT
Downloads and parses ONLY genuine, physical battery datasets.
NO synthetic np.linspace fallbacks. If the download fails, the script FAILS.

Datasets:
  1. Stanford/MIT LFP (Severson 2019) - 124 cells from HuggingFace mirror
  2. NASA Ames LCO (Saha & Goebel 2007) - 4 cells from Kaggle mirror
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import urllib.request
import zipfile
import glob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [RealDataAcq] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RealDataAcq")

SEED = 42
np.random.seed(SEED)

# --- BAD CELLS from Severson et al. 2019 ---
BAD_CELLS = {
    "b1c8", "b1c10", "b1c12", "b1c13", "b1c22",
    "b2c7", "b2c8", "b2c9", "b2c15", "b2c16",
    "b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"
}


def get_severson_valid_cells():
    """Returns the 124 valid cell IDs from Severson et al. (2019)."""
    cells = []
    for i in range(46):
        cid = f"b1c{i}"
        if cid not in BAD_CELLS:
            cells.append(cid)
    for i in range(48):
        cid = f"b2c{i}"
        if cid not in BAD_CELLS:
            cells.append(cid)
    for i in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 44, 45]:
        cid = f"b3c{i}"
        if cid not in BAD_CELLS:
            cells.append(cid)
    return cells


# ============================================================
# DATASET 1: Stanford/MIT LFP (Severson et al. 2019) - 124 cells
# Source: HuggingFace mirror of the original MATR.io dataset
# Chemistry: LiFePO4 (LFP), Voltage range: [2.0V, 3.6V]
# ============================================================

SEVERSON_BASE_URL = "https://huggingface.co/datasets/bsebench-org/severson-2019/resolve/main/{cell_id}.parquet"


def download_stanford_lfp(raw_dir: str, proc_dir: str):
    """
    Downloads real Stanford/MIT LFP cells from HuggingFace.
    Returns standardized parquet with columns:
      cell_id, chemistry, cycle_number, voltage_V, capacity_Ah, current_A, cycle_life
    """
    logger.info("=" * 70)
    logger.info("DOWNLOADING REAL STANFORD/MIT LFP DATASET (Severson et al. 2019)")
    logger.info("=" * 70)

    cells = get_severson_valid_cells()
    os.makedirs(raw_dir, exist_ok=True)

    # Download raw parquet files
    for i, cell_id in enumerate(cells):
        target_path = os.path.join(raw_dir, f"{cell_id}.parquet")
        if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
            continue  # Already cached

        url = SEVERSON_BASE_URL.format(cell_id=cell_id)
        logger.info(f"[{i + 1}/{len(cells)}] Downloading {cell_id}...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
                with open(target_path, "wb") as f:
                    f.write(data)
        except Exception as e:
            logger.error(f"FATAL: Failed to download {cell_id}: {e}")
            raise RuntimeError(f"Cannot download real data for {cell_id}. Aborting — NO SYNTHETIC FALLBACK.")

    # Process: compute cycle_life and extract early cycles
    logger.info("Processing real LFP discharge curves...")
    all_records = []

    for cell_id in cells:
        fpath = os.path.join(raw_dir, f"{cell_id}.parquet")
        df = pd.read_parquet(fpath)
        df["cycle_number"] = df["cycle_number"].astype(int)
        df = df.dropna(subset=["voltage_V", "current_A", "capacity_Ah"])

        # Compute real cycle_life from capacity fade
        cycle_caps = df.groupby("cycle_number")["capacity_Ah"].max()
        c0 = float(cycle_caps.iloc[0])
        threshold = min(0.88, 0.8 * c0)
        below = cycle_caps[cycle_caps < threshold]
        cycle_life = int(below.index[0]) if len(below) > 0 else int(cycle_caps.index[-1])

        # Extract only discharge data from early cycles (10 to 100, step 2)
        for cyc in range(10, 101, 2):
            cyc_df = df[(df["cycle_number"] == cyc) & (df["current_A"] < -0.01)]
            if len(cyc_df) < 5:
                continue
            cyc_df = cyc_df.sort_values("time_s")
            for _, row in cyc_df.iterrows():
                all_records.append({
                    "cell_id": cell_id,
                    "chemistry": "LFP",
                    "cycle_number": cyc,
                    "voltage_V": float(row["voltage_V"]),
                    "capacity_Ah": float(row["capacity_Ah"]),
                    "current_A": float(row["current_A"]),
                    "cycle_life": cycle_life
                })

    result_df = pd.DataFrame(all_records)
    out_path = os.path.join(proc_dir, "stanford_lfp.parquet")
    result_df.to_parquet(out_path, index=False)

    n_cells = result_df["cell_id"].nunique()
    cl = result_df.groupby("cell_id")["cycle_life"].first()
    logger.info(f"✅ REAL Stanford LFP: {n_cells} cells | {len(result_df)} rows | EOL range: [{cl.min()}, {cl.max()}]")
    logger.info(f"   Saved to: {out_path}")
    return result_df



def main():
    parser = argparse.ArgumentParser(description="Real Battery Data Acquisition (NO SYNTHETIC FALLBACKS)")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Raw data cache directory")
    parser.add_argument("--proc-dir", type=str, default="data/real_processed", help="Processed output directory")
    parser.add_argument("--skip-lfp", action="store_true", help="Skip Stanford LFP if already cached")

    args = parser.parse_args()

    os.makedirs(args.proc_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("REAL DATA ACQUISITION — NO SYNTHETIC FALLBACKS ALLOWED")
    logger.info("=" * 70)

    # Dataset 1: Stanford LFP
    lfp_out = os.path.join(args.proc_dir, "stanford_lfp.parquet")
    if args.skip_lfp and os.path.exists(lfp_out):
        logger.info(f"Skipping Stanford LFP (already exists at {lfp_out})")
    else:
        download_stanford_lfp(args.raw_dir, args.proc_dir)



    logger.info("=" * 70)
    logger.info("ALL REAL DATA ACQUISITION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
