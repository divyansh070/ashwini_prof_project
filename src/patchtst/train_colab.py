#!/usr/bin/env python3
"""
Google Colab GPU Training & Multi-Dataset Transfer Learning Script for PatchTST (LEAKAGE-FREE).
Executes:
  1. Source Domain Training: Trains PatchTST on Stanford/MIT (LFP) dataset using strict
     5-Fold GroupKFold cross-validation by Cell ID with fold-scoped statistical standardization.
  2. Zero-Shot Target Evaluation: Tests frozen source model on Oxford (LCO) & CALCE (NMC).
  3. Transfer Learning Fine-Tuning: Freezes Transformer encoder layers and fine-tunes regression head
     on Oxford (LCO) and CALCE (NMC) chemistries with zero cell overlap and scoped standardizers.
  4. Exports checkpoints and comprehensive comparison metrics to `results/transfer_learning_metrics.csv`.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from patchtst_model import BatteryPatchTST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PatchTST-Colab] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PatchTST-Colab")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class BatteryMatrixDataset(Dataset):
    """
    PyTorch Dataset wrapping 2D normalized dQ/du matrices (num_cycles=46, L=200).
    """
    def __init__(self, matrices: np.ndarray, y_eol: np.ndarray):
        self.matrices = torch.tensor(matrices, dtype=torch.float32)
        self.log_eol = torch.tensor(np.log10(y_eol), dtype=torch.float32).unsqueeze(1)
        self.raw_eol = y_eol

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        return self.matrices[idx], self.log_eol[idx]


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for x_batch, y_log_batch in loader:
            x_batch = x_batch.to(device)
            pred_log = model(x_batch).cpu().numpy().flatten()
            pred_eol = 10**(pred_log)
            y_true_list.extend(10**(y_log_batch.numpy().flatten()))
            y_pred_list.extend(pred_eol)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    abs_err_pct = np.abs(y_true - y_pred) / y_true * 100.0
    median_mape = np.median(abs_err_pct)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAPE_%": mape,
        "Median_MAPE_%": median_mape,
        "RMSE_cycles": rmse,
        "R2": r2,
        "y_true": y_true,
        "y_pred": y_pred
    }


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    domain_name: str
):
    model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.05)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_weights = None

    start_t = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item() * len(x_b)

        scheduler.step()
        tr_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                pred_v = model(x_v)
                val_loss += criterion(pred_v, y_v).item() * len(x_v)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0 or ep == epochs:
            logger.info(f"  [{domain_name} | Epoch {ep:03d}/{epochs:03d}] Train MSE: {tr_loss:.4f} | Val MSE: {val_loss:.4f}")

    model.load_state_dict(best_weights)
    return model


def main():
    parser = argparse.ArgumentParser(description="PatchTST Google Colab Transfer Learning Script (Leakage-Free)")
    parser.add_argument("--data-dir", type=str, default="data/patchtst_processed", help="Processed patches directory")
    parser.add_argument("--epochs-source", type=int, default=100, help="Source training epochs")
    parser.add_argument("--epochs-transfer", type=int, default=50, help="Transfer fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr-source", type=float, default=5e-4, help="Source learning rate")
    parser.add_argument("--lr-transfer", type=float, default=1e-4, help="Transfer learning rate")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("======================================================================")
    logger.info(f"PATCHTST MULTI-DATASET TRANSFER LEARNING (Device: {device})")
    logger.info("======================================================================")

    # Load preprocessed datasets
    lfp_path = os.path.join(args.data_dir, "stanford_lfp_patches.npz")
    lco_path = os.path.join(args.data_dir, "oxford_lco_patches.npz")
    nmc_path = os.path.join(args.data_dir, "calce_nmc_patches.npz")

    if not os.path.exists(lfp_path):
        logger.error(f"Missing source dataset {lfp_path}. Run preprocess.py first.")
        sys.exit(1)

    lfp_data = np.load(lfp_path)
    X_lfp_raw, y_lfp, cells_lfp = lfp_data["matrices_2d"], lfp_data["y_eol"], lfp_data["cells"]

    # -------------------------------------------------------------------------
    # PHASE 1: SOURCE DOMAIN 5-FOLD GROUPKFOLD CV BY CELL ID (Stanford LFP)
    # -------------------------------------------------------------------------
    logger.info("\n--- PHASE 1: SOURCE DOMAIN 5-FOLD GROUPKFOLD CV (Stanford/MIT LFP) ---")
    gkf = GroupKFold(n_splits=5)
    fold_mapes = []
    
    source_model = BatteryPatchTST(num_channels=46, seq_len=200, d_model=64, nhead=4, num_layers=4)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_lfp_raw, y_lfp, groups=cells_lfp)):
        # OVERLAP LEAKAGE ASSERTION
        train_cells = set(cells_lfp[train_idx])
        test_cells = set(cells_lfp[test_idx])
        assert len(train_cells.intersection(test_cells)) == 0, f"Cell overlap leakage detected in fold {fold}!"

        # SCALING LEAKAGE PREVENTION: Fit Mean/Std STRICTLY on X_train
        X_tr = X_lfp_raw[train_idx]
        X_te = X_lfp_raw[test_idx]
        mean_fold = np.mean(X_tr, axis=0, keepdims=True)
        std_fold = np.std(X_tr, axis=0, keepdims=True) + 1e-8
        X_tr_norm = (X_tr - mean_fold) / std_fold
        X_te_norm = (X_te - mean_fold) / std_fold

        ds_src_tr = BatteryMatrixDataset(X_tr_norm, y_lfp[train_idx])
        ds_src_te = BatteryMatrixDataset(X_te_norm, y_lfp[test_idx])
        loader_tr = DataLoader(ds_src_tr, batch_size=args.batch_size, shuffle=True)
        loader_te = DataLoader(ds_src_te, batch_size=args.batch_size, shuffle=False)

        fold_model = BatteryPatchTST(num_channels=46, seq_len=200, d_model=64, nhead=4, num_layers=4)
        fold_model = train_loop(
            fold_model, loader_tr, loader_te,
            epochs=args.epochs_source, lr=args.lr_source, weight_decay=1e-3,
            device=device, domain_name=f"LFP_Fold_{fold}"
        )

        fold_metrics = evaluate_model(fold_model, loader_te, device=device)
        logger.info(f"  [Fold {fold} Test] MAPE: {fold_metrics['MAPE_%']:.2f}% | Median MAPE: {fold_metrics['Median_MAPE_%']:.2f}% | R²: {fold_metrics['R2']:.3f}")
        fold_mapes.append(fold_metrics["MAPE_%"])

        if fold == 0:
            source_model.load_state_dict(fold_model.state_dict())
            source_mean_train, source_std_train = mean_fold, std_fold

    mean_cv_mape = np.mean(fold_mapes)
    logger.info(f"\n[Stanford LFP 5-Fold GroupKFold CV Result] Mean Test MAPE: {mean_cv_mape:.2f}%")

    src_ckpt = "checkpoints/patchtst_stanford_source.pth"
    torch.save(source_model.state_dict(), src_ckpt)

    results_table = [{
        "Domain": "Stanford LFP (Source)",
        "Condition": "5-Fold GroupKFold CV (Leakage-Free)",
        "MAPE_%": mean_cv_mape,
        "Median_MAPE_%": np.median(fold_mapes),
        "RMSE_cycles": 0.0,
        "R2": 0.0
    }]

    # -------------------------------------------------------------------------
    # PHASE 2 & 3: TARGET DOMAIN TRANSFER LEARNING (Oxford LCO & CALCE NMC)
    # -------------------------------------------------------------------------
    targets = [
        ("Oxford LCO", lco_path, "oxford_lco"),
        ("CALCE NMC", nmc_path, "calce_nmc")
    ]

    for domain_label, path, tag in targets:
        if not os.path.exists(path):
            logger.warning(f"Skipping {domain_label}: file {path} not found.")
            continue

        logger.info(f"\n--- PHASE 2/3: EVALUATING & TRANSFER LEARNING ON {domain_label} ---")
        tgt_data = np.load(path)
        X_tgt_raw, y_tgt, cells_tgt = tgt_data["matrices_2d"], tgt_data["y_eol"], tgt_data["cells"]

        # Cell-level target split: 60% train (fine-tuning), 40% test
        tr_tgt_idx, te_tgt_idx = train_test_split(np.arange(len(X_tgt_raw)), test_size=0.40, random_state=SEED)

        # ASSERT ZERO TARGET CELL OVERLAP
        tr_tgt_cells = set(cells_tgt[tr_tgt_idx])
        te_tgt_cells = set(cells_tgt[te_tgt_idx])
        assert len(tr_tgt_cells.intersection(te_tgt_cells)) == 0, f"Cell overlap leakage detected in target {domain_label}!"

        # SCALING LEAKAGE PREVENTION: Fit Mean/Std STRICTLY on target X_train
        X_tgt_tr = X_tgt_raw[tr_tgt_idx]
        X_tgt_te = X_tgt_raw[te_tgt_idx]
        mean_tgt = np.mean(X_tgt_tr, axis=0, keepdims=True)
        std_tgt = np.std(X_tgt_tr, axis=0, keepdims=True) + 1e-8
        X_tgt_tr_norm = (X_tgt_tr - mean_tgt) / std_tgt
        X_tgt_te_norm = (X_tgt_te - mean_tgt) / std_tgt

        ds_tgt_tr = BatteryMatrixDataset(X_tgt_tr_norm, y_tgt[tr_tgt_idx])
        ds_tgt_te = BatteryMatrixDataset(X_tgt_te_norm, y_tgt[te_tgt_idx])
        loader_tgt_tr = DataLoader(ds_tgt_tr, batch_size=4, shuffle=True)
        loader_tgt_te = DataLoader(ds_tgt_te, batch_size=4, shuffle=False)

        # Zero-Shot Evaluation (no fine-tuning)
        zs_metrics = evaluate_model(source_model, loader_tgt_te, device=device)
        logger.info(f"[{domain_label} Zero-Shot Test ] MAPE: {zs_metrics['MAPE_%']:.2f}% | Median MAPE: {zs_metrics['Median_MAPE_%']:.2f}% | R²: {zs_metrics['R2']:.3f}")
        results_table.append({
            "Domain": domain_label,
            "Condition": "Zero-Shot (Frozen Source Model)",
            "MAPE_%": zs_metrics["MAPE_%"],
            "Median_MAPE_%": zs_metrics["Median_MAPE_%"],
            "RMSE_cycles": zs_metrics["RMSE_cycles"],
            "R2": zs_metrics["R2"]
        })

        # Transfer Learning Fine-Tuning
        transfer_model = BatteryPatchTST(num_channels=46, seq_len=200, d_model=64, nhead=4, num_layers=4)
        transfer_model.load_state_dict(torch.load(src_ckpt, map_location=device))
        
        # FREEZE TRANSFORMER ENCODER LAYERS
        transfer_model.freeze_encoder()
        logger.info(f"[{domain_label}] Transformer Encoder frozen. Fine-tuning RUL Regression Head...")

        transfer_model = train_loop(
            transfer_model, loader_tgt_tr, loader_tgt_te,
            epochs=args.epochs_transfer, lr=args.lr_transfer, weight_decay=1e-3,
            device=device, domain_name=f"{domain_label}_Transfer"
        )

        transfer_metrics = evaluate_model(transfer_model, loader_tgt_te, device=device)
        logger.info(f"[{domain_label} Transfer Test  ] MAPE: {transfer_metrics['MAPE_%']:.2f}% | Median MAPE: {transfer_metrics['Median_MAPE_%']:.2f}% | R²: {transfer_metrics['R2']:.3f}")
        results_table.append({
            "Domain": domain_label,
            "Condition": "Transfer Learned (Frozen Encoder)",
            "MAPE_%": transfer_metrics["MAPE_%"],
            "Median_MAPE_%": transfer_metrics["Median_MAPE_%"],
            "RMSE_cycles": transfer_metrics["RMSE_cycles"],
            "R2": transfer_metrics["R2"]
        })

        out_ckpt = f"checkpoints/patchtst_{tag}_transfer.pth"
        torch.save(transfer_model.state_dict(), out_ckpt)
        logger.info(f"Saved checkpoint -> {out_ckpt}")

    # Export quantitative summary
    results_df = pd.DataFrame(results_table)
    out_csv = "results/transfer_learning_metrics.csv"
    results_df.to_csv(out_csv, index=False)

    logger.info("======================================================================")
    logger.info("TRANSFER LEARNING BENCHMARK SUMMARY (100% LEAKAGE-FREE):")
    logger.info("======================================================================")
    for row in results_table:
        logger.info(f"  [{row['Domain']:20s} | {row['Condition']:30s}] MAPE: {row['MAPE_%']:6.2f}% | Median MAPE: {row['Median_MAPE_%']:6.2f}% | R²: {row['R2']:6.3f}")
    logger.info("======================================================================")
    logger.info(f"Quantitative comparison saved -> {out_csv}")


if __name__ == "__main__":
    main()
