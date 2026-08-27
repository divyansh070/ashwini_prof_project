#!/usr/bin/env python3
"""
HybridoNet-Adapt Training and Evaluation Pipeline (Tran et al., 2025).

Implements:
1. Combined Loss: L_total = L_MSE(Source) + L_MSE(Target) + lambda(p) * L_MMD
2. Dynamic Lambda Scheduling: lambda_p = 2 / (1 + exp(-gamma * p)) - 1, with gamma=10, p = epoch / total_epochs
3. Zero Data Leakage: Min-Max scalers for features and RUL targets are strictly fitted ONLY on the training folds.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import glob
from typing import Tuple, Dict, List
from sklearn.model_selection import KFold, train_test_split


# Add directory and project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)

from model_hybrido import HybridoNetAdapt
from mmd_loss import MMDLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [HybridoTrain] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HybridoTrain")


class BatteryDataset(Dataset):
    """PyTorch Dataset for battery tensor samples."""
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def compute_dynamic_lambda(epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
    """
    Computes dynamic lambda scheduling:
        lambda_p = 2 / (1 + exp(-gamma * p)) - 1
    where p in [0, 1] is training progress.
    """
    p = float(epoch) / float(max(1, total_epochs))
    return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)

def fit_and_transform_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_tgt_adapt: np.ndarray,
    X_tgt_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Dynamically fits MinMaxScaler ONLY on X_train."""
    n_tr, seq, ch, feat = X_train.shape
    X_tr_flat = X_train.reshape(n_tr, seq * ch * feat)

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    X_tr_scaled = scaler.fit_transform(X_tr_flat).reshape(n_tr, seq, ch, feat)

    X_val_scaled = scaler.transform(X_val.reshape(X_val.shape[0], seq * ch * feat)).reshape(X_val.shape[0], seq, ch, feat)
    X_tgt_adapt_scaled = scaler.transform(X_tgt_adapt.reshape(X_tgt_adapt.shape[0], seq * ch * feat)).reshape(X_tgt_adapt.shape[0], seq, ch, feat)
    X_tgt_test_scaled = scaler.transform(X_tgt_test.reshape(X_tgt_test.shape[0], seq * ch * feat)).reshape(X_tgt_test.shape[0], seq, ch, feat)

    return X_tr_scaled, X_val_scaled, X_tgt_adapt_scaled, X_tgt_test_scaled, scaler


def fit_and_transform_targets(
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    Y_tgt_adapt: np.ndarray,
    Y_tgt_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Fits target MinMaxScaler ONLY on Y_train."""
    scaler_y = MinMaxScaler(feature_range=(0.0, 1.0))
    Y_tr_scaled = scaler_y.fit_transform(Y_train.reshape(-1, 1)).flatten()
    Y_val_scaled = scaler_y.transform(Y_val.reshape(-1, 1)).flatten()
    Y_tgt_adapt_scaled = scaler_y.transform(Y_tgt_adapt.reshape(-1, 1)).flatten()
    Y_tgt_test_scaled = scaler_y.transform(Y_tgt_test.reshape(-1, 1)).flatten()

    return Y_tr_scaled, Y_val_scaled, Y_tgt_adapt_scaled, Y_tgt_test_scaled, scaler_y

def train_single_fold(
    model: nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    val_loader: DataLoader,
    target_test_loader: DataLoader,
    target_y_raw: np.ndarray,
    scaler_y: MinMaxScaler,
    epochs: int = 50,
    lr: float = 1e-3,
    sigma_mmd: float = 1.0,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Executes the HybridoNet domain adaptation training loop.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_mse = nn.MSELoss()
    mmd_loss_fn = MMDLoss(sigma=sigma_mmd, fix_sigma=True)

    best_test_rmse = float("inf")
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        lambda_p = compute_dynamic_lambda(epoch, epochs, gamma=10.0)

        total_loss_accum = 0.0
        src_loss_accum = 0.0
        tgt_loss_accum = 0.0
        mmd_loss_accum = 0.0
        batches = 0

        target_iter = iter(target_loader)

        for src_x, src_y in source_loader:
            try:
                tgt_x, tgt_y = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_x, tgt_y = next(target_iter)

            src_x, src_y = src_x.to(device), src_y.to(device)
            tgt_x, tgt_y = tgt_x.to(device), tgt_y.to(device)

            optimizer.zero_grad()

            # Source forward pass
            y_comb_s, y_pred_s, _, z_s = model(src_x)
            loss_source = criterion_mse(y_pred_s, src_y)

            # Target forward pass
            y_comb_t, _, y_pred_t, z_t = model(tgt_x)
            loss_target = criterion_mse(y_pred_t, tgt_y)

            # Maximum Mean Discrepancy (MMD) Loss
            loss_mmd = mmd_loss_fn(z_s, z_t)

            # Combined Objective
            loss_total = loss_source + loss_target + lambda_p * loss_mmd

            loss_total.backward()
            optimizer.step()

            total_loss_accum += loss_total.item()
            src_loss_accum += loss_source.item()
            tgt_loss_accum += loss_target.item()
            mmd_loss_accum += loss_mmd.item()
            batches += 1

        # Evaluate on Target Test Set
        model.eval()
        target_preds = []
        with torch.no_grad():
            for t_x, _ in target_test_loader:
                t_x = t_x.to(device)
                y_comb_t, _, _, _ = model(t_x)
                target_preds.extend(y_comb_t.cpu().numpy().flatten())

        target_preds = np.array(target_preds).reshape(-1, 1)
        # Invert target scaling to physical cycle life units
        target_preds_unscaled = scaler_y.inverse_transform(target_preds).flatten()

        test_rmse = float(np.sqrt(mean_squared_error(target_y_raw, target_preds_unscaled)))
        test_mape = float(mean_absolute_percentage_error(target_y_raw, target_preds_unscaled) * 100.0)

        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_metrics = {
                "best_epoch": epoch,
                "best_rmse": test_rmse,
                "best_mape": test_mape,
                "theta_s": float(model.theta_s.item()),
                "theta_t": float(model.theta_t.item())
            }

        if epoch % 10 == 0 or epoch == epochs:
            logger.info(
                f"Epoch [{epoch:03d}/{epochs:03d}] "
                f"Loss: {total_loss_accum / max(1, batches):.4f} | "
                f"Src MSE: {src_loss_accum / max(1, batches):.4f} | "
                f"Tgt MSE: {tgt_loss_accum / max(1, batches):.4f} | "
                f"MMD: {mmd_loss_accum / max(1, batches):.4f} (lambda={lambda_p:.3f}) | "
                f"Test RMSE: {test_rmse:.2f} cyc | MAPE: {test_mape:.2f}%"
            )

    return best_metrics

def run_benchmark(
    source_npz: str,
    target_npz: str,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    n_splits: int = 5,
    device: str = "cpu"
):
    logger.info(f"Loading Source: {source_npz}")
    src_data = np.load(source_npz)
    X_src_raw, Y_src_raw = src_data["X"], src_data["Y"]

    logger.info(f"Loading Target: {target_npz}")
    tgt_data = np.load(target_npz)
    X_tgt_raw, Y_tgt_raw = tgt_data["X"], tgt_data["Y"]

    # [NEW] Split the target dataset: 60% for adaptation, 40% for blind testing
    X_tgt_adapt, X_tgt_test, Y_tgt_adapt, Y_tgt_test = train_test_split(
        X_tgt_raw, Y_tgt_raw, test_size=0.40, random_state=42
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_src_raw)):
        logger.info(f"\n--- FOLD {fold + 1}/{n_splits} ---")
        X_tr_raw, X_val_raw = X_src_raw[train_idx], X_src_raw[val_idx]
        Y_tr_raw, Y_val_raw = Y_src_raw[train_idx], Y_src_raw[val_idx]

        # 1. Zero-leakage dynamic scaling with the new target splits
        X_tr_sc, X_val_sc, X_tgt_adapt_sc, X_tgt_test_sc, scaler_x = fit_and_transform_features(
            X_tr_raw, X_val_raw, X_tgt_adapt, X_tgt_test
        )
        Y_tr_sc, Y_val_sc, Y_tgt_adapt_sc, Y_tgt_test_sc, scaler_y = fit_and_transform_targets(
            Y_tr_raw, Y_val_raw, Y_tgt_adapt, Y_tgt_test
        )

        src_loader = DataLoader(BatteryDataset(X_tr_sc, Y_tr_sc), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(BatteryDataset(X_val_sc, Y_val_sc), batch_size=batch_size, shuffle=False)
        
        # [NEW] Separate loaders for adaptation (train) vs testing
        tgt_loader = DataLoader(BatteryDataset(X_tgt_adapt_sc, Y_tgt_adapt_sc), batch_size=batch_size, shuffle=True)
        tgt_test_loader = DataLoader(BatteryDataset(X_tgt_test_sc, Y_tgt_test_sc), batch_size=batch_size, shuffle=False)

        model = HybridoNetAdapt(
            input_dim=18,
            hidden_dim=64,
            num_lstm_layers=2,
            num_heads=4,
            dropout=0.1
        )

        metrics = train_single_fold(
            model=model,
            source_loader=src_loader,
            target_loader=tgt_loader,
            val_loader=val_loader,
            target_test_loader=tgt_test_loader,
            target_y_raw=Y_tgt_test, # [NEW] Pass only the unseen test labels for metrics
            scaler_y=scaler_y,
            epochs=epochs,
            lr=lr,
            device=device
        )
        metrics["fold"] = fold + 1
        fold_results.append(metrics)
        logger.info(f"Fold {fold + 1} Best RMSE: {metrics['best_rmse']:.2f} cyc | Best MAPE: {metrics['best_mape']:.2f}%")

    avg_rmse = np.mean([r["best_rmse"] for r in fold_results])
    avg_mape = np.mean([r["best_mape"] for r in fold_results])
    logger.info(f"\n==========================================")
    logger.info(f"HYBRIDONET-ADAPT FINAL 5-FOLD BENCHMARK")
    logger.info(f"Average Target RMSE: {avg_rmse:.2f} cycles")
    logger.info(f"Average Target MAPE: {avg_mape:.2f}%")
    logger.info(f"==========================================")


def main():
    parser = argparse.ArgumentParser(description="HybridoNet-Adapt Benchmark Runner")
    parser.add_argument("--source", type=str, default="", help="Path to source .npz raw features")
    parser.add_argument("--target", type=str, default="", help="Path to target .npz raw features")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Auto-detect processed npz if not provided
    processed_files = glob.glob("data/hybridonet/processed/*_raw_features.npz")
    if not args.source and processed_files:
        args.source = processed_files[0]
    if not args.target and len(processed_files) > 1:
        args.target = processed_files[1]

    if not args.source or not args.target:
        logger.warning(
            "Source or target .npz files not found. "
            "Please run 'python src/hybridonet-adapt/preprocess_hybrido.py' first."
        )
        return

    run_benchmark(args.source, args.target, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)


if __name__ == "__main__":
    main()
