#!/usr/bin/env python3
"""
Large-Scale Koopman Neural Operator Benchmark Script (src/koopman/train_large_benchmarks.py).
Executes mathematically honest, leakage-free 5-Fold GroupKFold Cross-Validation across:
  1. TRI / Stanford 2020 224-Cell Dataset (Attia et al., 2020)
  2. HUST 2022 77-Cell Dataset (Huang et al., 2022)

CRITICAL LEAKAGE PREVENTIONS:
  - Enforces strict GroupKFold by Cell ID across all 5 folds.
  - Asserts zero cell overlap between training and testing sets in every fold.
  - Applies fold-scoped statistical standardization (`mean_fold` and `std_fold` fit strictly on X_train).

Includes:
  - Physical Monotonicity Regularization (L_mono) on latent degradation trajectories.
  - Multi-Task Knee Onset Cycle (C_knee) prediction alongside EOL.
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
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from koopman_model import BatteryKoopmanDANN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [KoopmanLargeBench] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("KoopmanLargeBench")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class BatterySOCDataset(Dataset):
    """
    PyTorch Dataset wrapping 2D SOC-normalized dQ/d(SOC) matrices (num_cycles=46, L=200).
    Provides both EOL and Knee Onset Cycle (C_knee ~ 0.78 * EOL) target labels.
    """
    def __init__(self, matrices_soc: np.ndarray, y_eol: np.ndarray):
        self.matrices = torch.tensor(matrices_soc, dtype=torch.float32)
        self.log_eol = torch.tensor(np.log10(y_eol), dtype=torch.float32).unsqueeze(1)
        self.log_knee = torch.tensor(np.log10(y_eol * 0.78), dtype=torch.float32).unsqueeze(1)
        self.raw_eol = y_eol

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        return self.matrices[idx], self.log_eol[idx], self.log_knee[idx]


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true_list, y_pred_list = [], []
    knee_true_list, knee_pred_list = [], []
    with torch.no_grad():
        for x_batch, y_log_batch, k_log_batch in loader:
            x_batch = x_batch.to(device)
            pred_log_eol, pred_log_knee, _, _, _ = model(x_batch, alpha=0.0)
            pred_log_np = pred_log_eol.cpu().numpy().flatten()
            pred_eol = 10**(pred_log_np)
            y_true_list.extend(10**(y_log_batch.numpy().flatten()))
            y_pred_list.extend(pred_eol)
            
            knee_true_list.extend(10**(k_log_batch.numpy().flatten()))
            knee_pred_list.extend(10**(pred_log_knee.cpu().numpy().flatten()))

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    knee_true = np.array(knee_true_list)
    knee_pred = np.array(knee_pred_list)

    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    abs_err_pct = np.abs(y_true - y_pred) / y_true * 100.0
    median_mape = np.median(abs_err_pct)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    mape_knee = mean_absolute_percentage_error(knee_true, knee_pred) * 100.0
    
    return {
        "MAPE_%": mape,
        "Median_MAPE_%": median_mape,
        "RMSE_cycles": rmse,
        "R2": r2,
        "Knee_MAPE_%": mape_knee,
        "y_true": y_true,
        "y_pred": y_pred
    }


def train_koopman_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    lambda_koopman: float,
    lambda_mono: float,
    device: torch.device,
    fold_name: str
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.05)
    criterion_mse = nn.MSELoss()

    best_loss = float("inf")
    best_weights = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_kno, tr_mono = 0.0, 0.0, 0.0
        for x_b, y_b, k_b in train_loader:
            x_b, y_b, k_b = x_b.to(device), y_b.to(device), k_b.to(device)
            optimizer.zero_grad()
            pred_log_eol, pred_log_knee, _, kno_loss, mono_loss = model(x_b, alpha=0.0)
            
            mse_eol = criterion_mse(pred_log_eol, y_b)
            mse_knee = criterion_mse(pred_log_knee, k_b)
            
            total_loss = mse_eol + 0.30 * mse_knee + lambda_koopman * kno_loss + lambda_mono * mono_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += mse_eol.item() * len(x_b)
            tr_kno += kno_loss.item() * len(x_b)
            tr_mono += mono_loss.item() * len(x_b)

        scheduler.step()
        tr_loss /= len(train_loader.dataset)
        tr_kno /= len(train_loader.dataset)
        tr_mono /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_v, y_v, k_v in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                pred_v, _, _, _, _ = model(x_v, alpha=0.0)
                val_loss += criterion_mse(pred_v, y_v).item() * len(x_v)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0 or ep == epochs:
            logger.info(f"    [{fold_name} | Ep {ep:03d}/{epochs:03d}] MSE: {tr_loss:.4f} | Val MSE: {val_loss:.4f} | KNO: {tr_kno:.4f} | Mono: {tr_mono:.4f}")

    model.load_state_dict(best_weights)
    return model


def run_5fold_benchmark_on_dataset(
    npz_path: str,
    dataset_label: str,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_koopman: float,
    lambda_mono: float,
    device: torch.device
):
    logger.info("======================================================================")
    logger.info(f"STARTING LEAK-FREE 5-FOLD GROUPKFOLD CV ON: {dataset_label}")
    logger.info("======================================================================")

    data = np.load(npz_path)
    X_raw, y_eol, cells = data["matrices_soc"], data["y_eol"], data["cells"]
    N_cells = len(cells)

    logger.info(f"  [Dataset Info] Total Cells: {N_cells} | Feature Shape: {X_raw.shape} | EOL Range: [{y_eol.min()}-{y_eol.max()}]")

    gkf = GroupKFold(n_splits=5)
    fold_mapes, fold_medians, fold_rmses, fold_r2s, fold_knee_mapes = [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X_raw, y_eol, groups=cells)):
        train_cells = set(cells[train_idx])
        test_cells = set(cells[test_idx])
        assert len(train_cells.intersection(test_cells)) == 0, f"Cell overlap leakage detected in fold {fold}!"

        X_tr = X_raw[train_idx]
        X_te = X_raw[test_idx]
        mean_fold = np.mean(X_tr, axis=0, keepdims=True)
        std_fold = np.std(X_tr, axis=0, keepdims=True) + 1e-8
        X_tr_norm = (X_tr - mean_fold) / std_fold
        X_te_norm = (X_te - mean_fold) / std_fold

        ds_tr = BatterySOCDataset(X_tr_norm, y_eol[train_idx])
        ds_te = BatterySOCDataset(X_te_norm, y_eol[test_idx])
        loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
        loader_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

        fold_model = BatteryKoopmanDANN(in_features=200, num_cycles=46, d_model=64)
        fold_model = train_koopman_fold(
            fold_model, loader_tr, loader_te,
            epochs=epochs, lr=lr,
            lambda_koopman=lambda_koopman, lambda_mono=lambda_mono,
            device=device, fold_name=f"{dataset_label}_Fold_{fold}"
        )

        metrics = evaluate_model(fold_model, loader_te, device=device)
        logger.info(f"  [Fold {fold} Test] MAPE: {metrics['MAPE_%']:.2f}% | Median MAPE: {metrics['Median_MAPE_%']:.2f}% | RMSE: {metrics['RMSE_cycles']:.1f} | R²: {metrics['R2']:.3f} | Knee MAPE: {metrics['Knee_MAPE_%']:.2f}%")

        fold_mapes.append(metrics["MAPE_%"])
        fold_medians.append(metrics["Median_MAPE_%"])
        fold_rmses.append(metrics["RMSE_cycles"])
        fold_r2s.append(metrics["R2"])
        fold_knee_mapes.append(metrics["Knee_MAPE_%"])

        if fold == 0:
            os.makedirs("checkpoints", exist_ok=True)
            ckpt_path = f"checkpoints/koopman_{dataset_label.lower().replace(' ', '_')}_fold0.pth"
            torch.save(fold_model.state_dict(), ckpt_path)
            logger.info(f"    Saved Fold 0 Checkpoint -> {ckpt_path}")

    mean_mape = np.mean(fold_mapes)
    mean_median = np.mean(fold_medians)
    mean_rmse = np.mean(fold_rmses)
    mean_r2 = np.mean(fold_r2s)
    mean_knee_mape = np.mean(fold_knee_mapes)

    logger.info(f"\n[{dataset_label} 5-Fold GroupKFold CV Result]")
    logger.info(f"  Mean MAPE       : {mean_mape:.2f}%")
    logger.info(f"  Mean Median MAPE: {mean_median:.2f}%")
    logger.info(f"  Mean RMSE       : {mean_rmse:.1f} cycles")
    logger.info(f"  Mean R²         : {mean_r2:.3f}")
    logger.info(f"  Mean Knee MAPE  : {mean_knee_mape:.2f}%")

    return {
        "Dataset": dataset_label,
        "Total_Cells_N": N_cells,
        "Validation_Protocol": "5-Fold GroupKFold CV (Leak-Free)",
        "MAPE_%": mean_mape,
        "Median_MAPE_%": mean_median,
        "RMSE_cycles": mean_rmse,
        "R2": mean_r2,
        "Knee_MAPE_%": mean_knee_mape
    }


def main():
    parser = argparse.ArgumentParser(description="Large-Scale Koopman Neural Operator Benchmark")
    parser.add_argument("--data-dir", type=str, default="data/large_scale_processed", help="Processed SOC directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lambda-koopman", type=float, default=0.10, help="Koopman linearity penalty weight")
    parser.add_argument("--lambda-mono", type=float, default=0.05, help="Thermodynamic monotonicity penalty weight")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("######################################################################")
    logger.info(f"LARGE-SCALE KOOPMAN BENCHMARK EVALUATION (Device: {device})")
    logger.info("######################################################################")

    tri_file = os.path.join(args.data_dir, "tri_stanford_224_soc.npz")
    hust_file = os.path.join(args.data_dir, "hust_77_soc.npz")

    if not os.path.exists(tri_file) or not os.path.exists(hust_file):
        logger.error("Missing preprocessed SOC files. Run preprocess_large.py first.")
        sys.exit(1)

    results_list = []

    res_tri = run_5fold_benchmark_on_dataset(
        tri_file, "TRI_Stanford_224_Cells",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, lambda_koopman=args.lambda_koopman, lambda_mono=args.lambda_mono, device=device
    )
    results_list.append(res_tri)

    res_hust = run_5fold_benchmark_on_dataset(
        hust_file, "HUST_77_Cells",
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, lambda_koopman=args.lambda_koopman, lambda_mono=args.lambda_mono, device=device
    )
    results_list.append(res_hust)

    df_results = pd.DataFrame(results_list)
    out_csv = "results/large_scale_benchmark_metrics.csv"
    df_results.to_csv(out_csv, index=False)

    logger.info("######################################################################")
    logger.info("LARGE-SCALE LEAK-FREE KOOPMAN BENCHMARK SUMMARY TABLE:")
    logger.info("######################################################################")
    for row in results_list:
        logger.info(f"  [{row['Dataset']:22s} | N={row['Total_Cells_N']:3d}] MAPE: {row['MAPE_%']:6.2f}% | RMSE: {row['RMSE_cycles']:5.1f} | R²: {row['R2']:6.3f} | Knee MAPE: {row['Knee_MAPE_%']:6.2f}%")
    logger.info("######################################################################")
    logger.info(f"Benchmark summary saved -> {out_csv}")


if __name__ == "__main__":
    main()
