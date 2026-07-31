#!/usr/bin/env python3
"""
Data Acquisition Module for the Stanford/MIT Fast-Charging Dataset (Severson et al., 2019).
Downloads cell cycling time-series across all 124 valid cells in Batches 1, 2, and 3 from reliable public mirrors,
structures and cleans the data, computes total battery cycle life (cycles until 80% capacity retention),
and exports to Parquet.
"""

import os
import sys
import argparse
import logging
import urllib.request
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("DataAcquisition")

# Known bad or anomalous cells dropped by Severson et al. (2019)
# (hardware failures, noisy channels, or carry-over duplicates)
BAD_CELLS = {
    "b1c8", "b1c10", "b1c12", "b1c13", "b1c22",
    "b2c7", "b2c8", "b2c9", "b2c15", "b2c16",
    "b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"
}

# Generate complete 124 valid cell list across Batches 1, 2, and 3
def get_all_valid_cells():
    cells = []
    # Batch 1: indices 0 to 45
    for i in range(46):
        cid = f"b1c{i}"
        if cid not in BAD_CELLS:
            cells.append(cid)
    # Batch 2: indices 0 to 47
    for i in range(48):
        cid = f"b2c{i}"
        if cid not in BAD_CELLS:
            cells.append(cid)
    # Batch 3: indices 0 to 45 (note: some indices may not exist in raw, filter below)
    for i in [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39, 40, 41, 44, 45]:
        cid = f"b3c{i}"
        if cid not in BAD_CELLS:
            cells.append(cid)
    return cells

DEFAULT_CELLS = get_all_valid_cells()
BASE_URL = "https://huggingface.co/datasets/bsebench-org/severson-2019/resolve/main/{cell_id}.parquet"


def download_cell_file(cell_id: str, raw_dir: str, max_retries: int = 3) -> str:
    """
    Downloads a single cell Parquet file from HuggingFace mirror if not already cached locally.
    """
    os.makedirs(raw_dir, exist_ok=True)
    target_path = os.path.join(raw_dir, f"{cell_id}.parquet")

    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        logger.debug(f"Cell {cell_id} already cached at {target_path}.")
        return target_path

    url = BASE_URL.format(cell_id=cell_id)
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading cell {cell_id} from {url} (Attempt {attempt+1}/{max_retries})...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()
                with open(target_path, "wb") as f:
                    f.write(data)
            logger.info(f"Successfully cached {cell_id} ({len(data)/1e6:.2f} MB).")
            return target_path
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed for {cell_id}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to download {cell_id} after {max_retries} attempts.")
                raise e
    return target_path


def process_cell(cell_id: str, file_path: str, max_early_cycle: int = 100):
    """
    Reads a raw cell Parquet file, calculates EOL cycle life (until 80% capacity retention),
    computes cycle-level summary statistics for early cycles (1 to max_early_cycle),
    and filters early cycle time-series data.
    """
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Error reading Parquet file {file_path} for cell {cell_id}: {e}")
        return None, None

    # Ensure required columns exist
    required_cols = ["cell_id", "time_s", "voltage_V", "current_A", "temperature_C", "cycle_number", "capacity_Ah"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column {col} in cell {cell_id}")
            return None, None

    # Clean data types and sort chronologically
    df["cycle_number"] = df["cycle_number"].astype(int)
    df = df.dropna(subset=["voltage_V", "current_A", "capacity_Ah", "cycle_number"])
    df = df.sort_values(["cycle_number", "time_s"]).reset_index(drop=True)

    # Compute per-cycle maximum discharge capacity across all cycles in dataset
    cycle_caps = df.groupby("cycle_number")["capacity_Ah"].max()
    if len(cycle_caps) == 0:
        return None, None

    # Define C0 as cycle 1 capacity (or earliest available cycle)
    c0 = float(cycle_caps.iloc[0])
    # For LFP/graphite in Severson et al. (2019), nominal capacity is 1.1 Ah, EOL threshold is 80% of nominal (0.88 Ah)
    # We use 0.88 Ah or 0.8 * c0 (whichever is lower) as robust EOL threshold
    threshold = min(0.88, 0.8 * c0)

    # Determine cycle_life (first cycle where capacity drops below threshold)
    below_thresh = cycle_caps[cycle_caps < threshold]
    if len(below_thresh) > 0:
        cycle_life = int(below_thresh.index[0])
    else:
        # If cell was stopped before reaching threshold, use max cycle number
        cycle_life = int(cycle_caps.index[-1])

    batch_name = cell_id[:2]  # e.g., 'b1', 'b2', 'b3'

    # Filter early cycles (1 to max_early_cycle) for feature engineering & modeling
    df_early = df[df["cycle_number"] <= max_early_cycle].copy()

    # Calculate cycle-by-cycle summary metrics for early cycles
    summary_records = []
    for cyc, group in df_early.groupby("cycle_number"):
        summary_records.append({
            "cell_id": cell_id,
            "batch": batch_name,
            "cycle_number": int(cyc),
            "discharge_capacity_Ah": float(group["capacity_Ah"].max()),
            "max_temperature_C": float(group["temperature_C"].max()),
            "min_temperature_C": float(group["temperature_C"].min()),
            "avg_temperature_C": float(group["temperature_C"].mean()),
            "avg_voltage_V": float(group["voltage_V"].mean()),
            "avg_current_A": float(group["current_A"].mean()),
            "initial_capacity_Ah": float(c0),
            "cycle_life": int(cycle_life)
        })

    summary_df = pd.DataFrame(summary_records)
    return df_early, summary_df


def main():
    parser = argparse.ArgumentParser(description="Data Acquisition for Stanford/MIT Fast-Charging Dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory for caching raw Parquet files")
    parser.add_argument("--proc-dir", type=str, default="data/processed", help="Directory for saving processed Parquet files")
    parser.add_argument("--cells", type=str, nargs="+", default=DEFAULT_CELLS, help="List of cell IDs to process")
    parser.add_argument("--threads", type=int, default=4, help="Number of parallel download/processing threads")
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.proc_dir, exist_ok=True)

    logger.info(f"Starting Data Acquisition for {len(args.cells)} Stanford/MIT cells...")
    logger.info(f"Raw cache directory: {args.raw_dir}")
    logger.info(f"Processed output directory: {args.proc_dir}")

    # Step 1: Parallel download and caching
    cached_files = {}
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        future_to_cell = {
            executor.submit(download_cell_file, cell_id, args.raw_dir): cell_id
            for cell_id in args.cells
        }
        for future in as_completed(future_to_cell):
            cell_id = future_to_cell[future]
            try:
                path = future.result()
                cached_files[cell_id] = path
            except Exception as e:
                logger.error(f"Failed to cache cell {cell_id}: {e}")

    if len(cached_files) == 0:
        logger.error("No cells successfully cached. Aborting data acquisition.")
        sys.exit(1)

    logger.info(f"Successfully cached {len(cached_files)}/{len(args.cells)} cells.")

    # Step 2: Process and clean cells
    all_time_series = []
    all_summary = []
    logger.info("Processing time-series and computing target cycle life across early cycles...")

    for cell_id, file_path in sorted(cached_files.items()):
        ts_df, sum_df = process_cell(cell_id, file_path, max_early_cycle=100)
        if ts_df is not None and sum_df is not None:
            all_time_series.append(ts_df)
            all_summary.append(sum_df)

    if len(all_summary) == 0:
        logger.error("No valid summary datasets generated. Aborting.")
        sys.exit(1)

    # Combine into unified DataFrames
    combined_ts = pd.concat(all_time_series, ignore_index=True)
    combined_summary = pd.concat(all_summary, ignore_index=True)

    # Save to Parquet
    ts_out_path = os.path.join(args.proc_dir, "battery_time_series.parquet")
    sum_out_path = os.path.join(args.proc_dir, "battery_summary.parquet")

    combined_ts.to_parquet(ts_out_path, index=False)
    combined_summary.to_parquet(sum_out_path, index=False)

    logger.info("="*60)
    logger.info("DATA ACQUISITION SUMMARY:")
    logger.info(f"Total cells processed: {combined_summary['cell_id'].nunique()}")
    
    cell_lives = combined_summary.groupby("cell_id")["cycle_life"].first()
    logger.info(f"Cycle Life distribution across cells:")
    logger.info(f"  Mean: {cell_lives.mean():.1f} cycles")
    logger.info(f"  Min : {cell_lives.min()} cycles")
    logger.info(f"  Max : {cell_lives.max()} cycles")
    logger.info(f"Time-series dataset saved: {ts_out_path} ({os.path.getsize(ts_out_path)/1e6:.2f} MB)")
    logger.info(f"Summary dataset saved    : {sum_out_path} ({os.path.getsize(sum_out_path)/1e6:.2f} MB)")
    logger.info("="*60)
    logger.info("Data acquisition complete!")


if __name__ == "__main__":
    main()
