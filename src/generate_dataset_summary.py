#!/usr/bin/env python3
"""
Generates an Excel (.xlsx) report containing the initial conditions
and metadata for all downloaded real battery cells.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DatasetSummary")

def extract_metadata(dataset, cell_id):
    # Default assumptions based on dataset literature
    temp = "Unknown"
    rate = "Unknown"
    condition = "Standard Cycling"
    
    if dataset == "Stanford":
        temp = "30°C"
        rate = "Fast Charge (3C-8C) / 1C Discharge"
        condition = "Fast Charging Study"
    elif dataset == "CALCE":
        temp = "Room Temp (~25°C)"
        rate = "0.5C Charge / 1C Discharge"
        condition = "Standard Degradation"
    elif dataset == "HUST":
        temp = "25°C"
        rate = "Varied (0.5C-2C)"
        condition = "Cycle Life Testing"
    elif dataset == "RWTH":
        temp = "25°C"
        rate = "Standard"
        condition = "Grid Storage / EV Simulation"
    elif dataset == "SNL":
        # Parse from filename e.g., SNL_18650_NMC_15C_0-100_0.5-1C_a
        parts = cell_id.split('_')
        for p in parts:
            if p.endswith('C') and p[:-1].isdigit():
                temp = f"{p[:-1]}°C"
            elif '-' in p and 'C' in p and not p.startswith('0-100'):
                rate = p.replace('C', 'C') + " (Charge-Discharge)"
        condition = "Temperature & Rate Study"
        
    return temp, rate, condition

def generate_excel_report(in_dir, out_file):
    if not os.path.exists(in_dir):
        logger.error(f"Input directory {in_dir} does not exist. Run the download scripts first.")
        return

    records = []
    
    # 1. Process Stanford LFP (which might be a single parquet file in the root of in_dir)
    lfp_path = os.path.join(in_dir, "stanford_lfp.parquet")
    if os.path.exists(lfp_path):
        logger.info(f"Processing Stanford LFP dataset...")
        df_lfp = pd.read_parquet(lfp_path)
        for cell_id in df_lfp["cell_id"].unique():
            cell_df = df_lfp[df_lfp["cell_id"] == cell_id]
            cyc_life = cell_df["cycle_life"].iloc[0]
            first_cyc = cell_df["cycle_number"].min()
            initial_cap = cell_df[cell_df["cycle_number"] == first_cyc]["capacity_Ah"].max()
            
            temp, rate, cond = extract_metadata("Stanford", cell_id)
            records.append({
                "Dataset": "Stanford",
                "Cell_ID": cell_id,
                "Chemistry": "LFP",
                "Temperature": temp,
                "Charge_Discharge_Rate": rate,
                "Operating_Condition": cond,
                "Initial_Capacity_Ah": initial_cap,
                "Cycle_Life": cyc_life
            })

    # 2. Process BatteryLife datasets (which are subdirectories in in_dir)
    batterylife_datasets = ["calce", "hust", "snl", "rwth"]
    
    for dname in batterylife_datasets:
        dpath = os.path.join(in_dir, dname)
        if not os.path.exists(dpath):
            continue
            
        logger.info(f"Processing {dname.upper()} dataset...")
        for file in os.listdir(dpath):
            if not file.endswith('.parquet'):
                continue
            
            cell_id = file.replace('.parquet', '')
            df_cell = pd.read_parquet(os.path.join(dpath, file))
            
            if df_cell.empty:
                continue
                
            chem = df_cell["chemistry"].iloc[0] if "chemistry" in df_cell.columns else "Unknown"
            cyc_life = df_cell["cycle_life"].iloc[0]
            
            first_cyc = df_cell["cycle_number"].min()
            initial_cap = df_cell[df_cell["cycle_number"] == first_cyc]["capacity_Ah"].max()
            
            temp, rate, cond = extract_metadata(dname.upper(), cell_id)
            records.append({
                "Dataset": dname.upper(),
                "Cell_ID": cell_id,
                "Chemistry": chem,
                "Temperature": temp,
                "Charge_Discharge_Rate": rate,
                "Operating_Condition": cond,
                "Initial_Capacity_Ah": initial_cap,
                "Cycle_Life": cyc_life
            })

    if not records:
        logger.warning("No cell data found. Did you download the datasets?")
        return

    # Create DataFrame and save to Excel
    final_df = pd.DataFrame(records)
    
    # Sort for better readability
    final_df = final_df.sort_values(by=["Dataset", "Cycle_Life"])
    
    try:
        # Requires openpyxl installed
        final_df.to_excel(out_file, index=False)
        logger.info(f"Successfully generated {out_file} with {len(final_df)} cells.")
        
        # Print a quick summary to console
        summary_stats = final_df.groupby("Dataset").agg({
            "Cell_ID": "count",
            "Initial_Capacity_Ah": ["mean", "min", "max"],
            "Cycle_Life": ["mean", "min", "max"]
        }).round(2)
        
        print("\n" + "="*80)
        print("DATASET AGGREGATE SUMMARY")
        print("="*80)
        print(summary_stats)
        print("="*80 + "\n")
        
    except ImportError:
        logger.error("Missing openpyxl. Install it via 'pip install openpyxl' to save .xlsx files.")
        # Fallback to CSV
        csv_out = out_file.replace('.xlsx', '.csv')
        final_df.to_csv(csv_out, index=False)
        logger.info(f"Saved as CSV instead: {csv_out}")

def main():
    parser = argparse.ArgumentParser(description="Generate dataset summary Excel file")
    parser.add_argument("--in-dir", type=str, default="data/real_processed", help="Input processed data directory")
    parser.add_argument("--out-file", type=str, default="dataset_initial_conditions.xlsx", help="Output Excel filename")
    args = parser.parse_args()
    generate_excel_report(args.in_dir, args.out_file)

if __name__ == "__main__":
    main()
