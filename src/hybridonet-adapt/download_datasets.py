#!/usr/bin/env python3
"""
HybridoNet-Adapt Data Acquisition Script (Tran et al., 2025).

Downloads and links the primary benchmark datasets evaluated in the paper:
1. TRI Dataset (Toyota Research Institute / Stanford fast-charging, Severson et al. 2019)
2. LHP Dataset (LiFePO4 varied discharge dataset / BatteryLife repository)
"""

import os
import sys
import argparse
import logging
import urllib.request
import zipfile
import io
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [HybridoDownload] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HybridoDownload")

# Zenodo mirrors and direct repository endpoints
DATASET_URLS = {
    "TRI": "https://zenodo.org/api/records/14969822/files/Stanford.zip/content", # Severson / TRI Fast Charging
    "LHP": "https://zenodo.org/api/records/14969822/files/HUST.zip/content",     # Varied discharge LFP/LHP benchmark
    "CALCE": "https://zenodo.org/api/records/14969822/files/CALCE.zip/content",
    "SNL": "https://zenodo.org/api/records/14969822/files/SNL.zip/content",
}


def download_and_extract_zenodo_dataset(dataset_key: str, dest_dir: str) -> bool:
    """Downloads and extracts a zip dataset from Zenodo mirror."""
    if dataset_key not in DATASET_URLS:
        logger.error(f"Unknown dataset key: {dataset_key}")
        return False

    url = DATASET_URLS[dataset_key]
    out_dir = os.path.join(dest_dir, dataset_key)
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"Downloading {dataset_key} from {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as response:
            zip_bytes = response.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                zf.extractall(out_dir)
        logger.info(f"Successfully extracted {dataset_key} to {out_dir}")
        return True
    except Exception as e:
        logger.warning(f"Direct download for {dataset_key} encountered: {e}")
        return False


def link_existing_project_data(project_root: str, dest_dir: str):
    """
    Checks if datasets are already downloaded in the main project data directories
    and symlinks/copies them to data/hybridonet/raw/.
    """
    source_processed = os.path.join(project_root, "data", "real_processed")
    if os.path.exists(source_processed):
        logger.info(f"Found existing processed data in {source_processed}. Linking to HybridoNet pipeline...")
        os.makedirs(dest_dir, exist_ok=True)
        for f in os.listdir(source_processed):
            if f.endswith(".parquet"):
                src_file = os.path.join(source_processed, f)
                dst_file = os.path.join(dest_dir, f)
                if not os.path.exists(dst_file):
                    try:
                        os.symlink(src_file, dst_file)
                        logger.info(f"Symlinked {f} -> {dst_file}")
                    except Exception:
                        import shutil
                        shutil.copy2(src_file, dst_file)
                        logger.info(f"Copied {f} -> {dst_file}")


def main():
    parser = argparse.ArgumentParser(description="Download TRI and LHP datasets for HybridoNet-Adapt")
    parser.add_argument("--output-dir", type=str, default="data/hybridonet/raw", help="Target download directory")
    parser.add_argument("--datasets", nargs="+", default=["TRI", "LHP"], choices=["TRI", "LHP", "CALCE", "SNL"], help="Datasets to download")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Initializing dataset acquisition for HybridoNet-Adapt baseline...")

    # First attempt to link existing project data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    link_existing_project_data(project_root, args.output_dir)

    # Download any missing target datasets
    for ds in args.datasets:
        ds_path = os.path.join(args.output_dir, ds)
        if not os.path.exists(ds_path) or len(os.listdir(ds_path)) == 0:
            download_and_extract_zenodo_dataset(ds, args.output_dir)

    logger.info("Dataset acquisition routine completed.")


if __name__ == "__main__":
    main()
