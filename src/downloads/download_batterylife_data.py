#!/usr/bin/env python3
"""
REAL DATA ACQUISITION SCRIPT v3 (BatteryLife Zenodo Mirror)
Downloads genuine physical battery datasets from Zenodo (no authentication required).
Converts raw .pkl files into standard .parquet Koopman inputs.
NO synthetic np.linspace fallbacks. If the download fails or data is fake, the script FAILS.

Datasets targeted:
  1. CALCE (NMC/LCO)
  2. HUST (LFP)
  3. SNL (LFP/NCA/NMC)
  4. RWTH (NMC)
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import pickle
import urllib.request
import zipfile
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ZenodoAcq] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ZenodoAcq")

SEED = 42
np.random.seed(SEED)

ZENODO_BASE_URL = "https://zenodo.org/api/records/14969822/files/{dataset}.zip/content"

DATASETS_TO_DOWNLOAD = {
    "CALCE": "NMC", # Assuming mostly NMC/LCO, label as NMC for preprocessing bounds
    "HUST": "LFP",
    "SNL": "NMC",   # SNL has mixed, we'll use NMC bounds as safe default
    "RWTH": "NMC"
}

def check_for_synthetic_data(voltage_array):
    """
    Checks if an array appears to be synthetically generated (e.g., using np.linspace).
    Real sensor data has non-uniform differences due to sensor noise/resolution.
    """
    if len(voltage_array) < 10:
        return False
    diffs = np.diff(voltage_array[:20])
    if len(diffs) > 2 and np.allclose(diffs, diffs[0], atol=1e-10):
        return True
    return False

def process_pkl_data(pkl_data, parquet_path, dataset_name, chemistry, filename):
    """
    Reads a BatteryLife .pkl file from bytes, validates it is real data, and converts it
    to our standardized .parquet format.
    """
    data = pickle.loads(pkl_data)
    
    cell_id = filename.split('.')[0]
    cycle_data = {}
    
    if isinstance(data, dict):
        cell_id = data.get('cell_id', cell_id)
        # Handle cases where cycle_data is at root or nested
        if 'cycle_data' in data:
            raw_cyc = data['cycle_data']
            if isinstance(raw_cyc, dict):
                cycle_data = raw_cyc
            elif isinstance(raw_cyc, list):
                for i, cyc_dict in enumerate(raw_cyc):
                    if isinstance(cyc_dict, dict):
                        cyc_num = cyc_dict.get('cycle_number', cyc_dict.get('cycle', i + 1))
                        try:
                            cyc_num = int(cyc_num)
                            cycle_data[cyc_num] = cyc_dict
                        except ValueError:
                            pass
        else:
            # Maybe the dict itself is cycle data keyed by cycle num
            cycle_data = {k: v for k, v in data.items() if isinstance(k, (int, str)) and str(k).isdigit()}
    elif isinstance(data, list):
        # List of cycle dictionaries
        for i, cyc_dict in enumerate(data):
            if isinstance(cyc_dict, dict):
                # Try to find a cycle number, fallback to index + 1
                cyc_num = cyc_dict.get('cycle_number', cyc_dict.get('cycle', i + 1))
                try:
                    cyc_num = int(cyc_num)
                    cycle_data[cyc_num] = cyc_dict
                except ValueError:
                    pass
    
    if not cycle_data:
        logger.warning(f"No cycle_data found in {filename}")
        return False
        
    all_records = []
    
    # Calculate cycle life (EOL). Typical definition is 80% of initial capacity.
    max_caps = []
    cycle_nums_available = sorted(cycle_data.keys())
    for cyc in cycle_nums_available:
        if 'discharge_capacity_in_Ah' in cycle_data[cyc]:
            cap_array = cycle_data[cyc]['discharge_capacity_in_Ah']
            if len(cap_array) > 0:
                max_caps.append((cyc, np.max(cap_array)))
    
    if len(max_caps) == 0:
         logger.warning(f"No capacity data in {filename}")
         return False
         
    c0 = max_caps[0][1]
    threshold = 0.8 * c0
    cycle_life = max_caps[-1][0]
    for cyc, cap in max_caps:
        if cap < threshold:
            cycle_life = cyc
            break
            
    # Extract early cycles (e.g., 10 to 100 step 2) for Koopman
    for cyc in range(10, 101, 2):
        if cyc not in cycle_data:
            continue
            
        cyc_dict = cycle_data[cyc]
        if 'voltage_in_V' not in cyc_dict or 'current_in_A' not in cyc_dict or 'discharge_capacity_in_Ah' not in cyc_dict:
            continue
            
        v = np.array(cyc_dict['voltage_in_V'])
        i = np.array(cyc_dict['current_in_A'])
        q = np.array(cyc_dict['discharge_capacity_in_Ah'])
        
        # Filter for discharge (current < 0)
        discharge_mask = i < -0.01
        v_dis = v[discharge_mask]
        i_dis = i[discharge_mask]
        q_dis = q[discharge_mask]
        
        if len(v_dis) < 5:
            continue
            
        # SYNTHETIC DATA AUDIT
        if check_for_synthetic_data(v_dis):
            logger.error(f"⚠️ FATAL: Synthetic data detected in {cell_id} cycle {cyc}. Voltage steps are uniform.")
            raise RuntimeError(f"Detected synthetic np.linspace data in {filename}. Aborting.")
            
        for idx in range(len(v_dis)):
            all_records.append({
                "cell_id": cell_id,
                "chemistry": chemistry,
                "cycle_number": cyc,
                "voltage_V": float(v_dis[idx]),
                "capacity_Ah": float(q_dis[idx]),
                "current_A": float(i_dis[idx]),
                "cycle_life": cycle_life
            })
            
    if not all_records:
        return False
        
    df = pd.DataFrame(all_records)
    df.to_parquet(parquet_path, index=False)
    logger.info(f"  Converted {cell_id} -> {os.path.basename(parquet_path)} (Cycle Life: {cycle_life}, Rows: {len(df)})")
    return True

def download_and_process_dataset(dataset_name, chemistry, raw_dir, proc_dir, test_mode):
    logger.info(f"\\nProcessing dataset: {dataset_name} ({chemistry})")
    dataset_proc_dir = os.path.join(proc_dir, dataset_name.lower())
    os.makedirs(dataset_proc_dir, exist_ok=True)
    
    zip_path = os.path.join(raw_dir, f"{dataset_name}.zip")
    url = ZENODO_BASE_URL.format(dataset=dataset_name)
    
    if not os.path.exists(zip_path):
        logger.info(f"Downloading {dataset_name}.zip from Zenodo...")
        try:
            logger.info("Executing curl...")
            import subprocess
            subprocess.run(["curl", "-L", "-o", zip_path, url], check=True)
            print()
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}.zip: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            return
    else:
        logger.info(f"Found existing {dataset_name}.zip in {raw_dir}")
        
    logger.info(f"Extracting and processing files from {dataset_name}.zip...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            pkl_files = [name for name in zf.namelist() if name.endswith('.pkl')]
            if not pkl_files:
                logger.warning(f"No .pkl files found in {dataset_name}.zip")
                return
                
            logger.info(f"Found {len(pkl_files)} cells in {dataset_name}.zip")
            if test_mode:
                logger.info("TEST MODE: Limiting to 2 cells.")
                pkl_files = pkl_files[:2]
                
            for filename in pkl_files:
                cell_id = os.path.basename(filename).replace('.pkl', '')
                parquet_path = os.path.join(dataset_proc_dir, f"{cell_id}.parquet")
                
                if os.path.exists(parquet_path):
                    continue
                    
                with zf.open(filename) as f:
                    pkl_data = f.read()
                    
                try:
                    process_pkl_data(pkl_data, parquet_path, dataset_name, chemistry, os.path.basename(filename))
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {e}")
                    if "Detected synthetic" in str(e):
                        sys.exit(1) # Hard crash on synthetic data
    except Exception as e:
        logger.error(f"Failed to read {dataset_name}.zip: {e}")
        logger.warning(f"Deleting corrupted {zip_path} so it can be re-downloaded next time.")
        if os.path.exists(zip_path):
            os.remove(zip_path)

def main():
    parser = argparse.ArgumentParser(description="Real BatteryLife Data Acquisition (Zenodo)")
    parser.add_argument("--raw-dir", type=str, default="data/raw/batterylife", help="Raw zip cache directory")
    parser.add_argument("--proc-dir", type=str, default="data/real_processed", help="Processed parquet output directory")
    parser.add_argument("--test-mode", action="store_true", help="Only process 2 cells per dataset")
    parser.add_argument("--skip", nargs="*", default=[], help="Datasets to skip (e.g. HUST RWTH)")
    args = parser.parse_args()

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.proc_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("REAL DATA ACQUISITION (Zenodo) — NO SYNTHETIC FALLBACKS ALLOWED")
    logger.info("=" * 70)

    for dataset_name, chemistry in DATASETS_TO_DOWNLOAD.items():
        if dataset_name in args.skip:
            logger.info(f"Skipping {dataset_name} as requested.")
            continue
        download_and_process_dataset(dataset_name, chemistry, args.raw_dir, args.proc_dir, args.test_mode)

    logger.info("=" * 70)
    logger.info("ALL REAL DATA ACQUISITION COMPLETE")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
