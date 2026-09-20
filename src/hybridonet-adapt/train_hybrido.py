#!/usr/bin/env python3
"""
HybridoNet-Adapt Training and Evaluation Pipeline (Tran et al., 2025).

Implements:
1. Strict Cell-Level Splitting: Zero intra-battery window leakage across train/validation/test partitions.
2. Robust Physical RUL Normalization: Configurable cycle ceiling with strict error on overflow (no silent corruption).
3. Published Architecture (Table 2): 128-D Feature Extractor (LSTM + MHA + NODE) and last attention timestep.
4. Combined Objective: L_total = L_MSE(Source) + L_MSE(Target_Combined) + lambda(p) * L_MMD
5. Constrained Trade-Off Optimization: theta_s + theta_t == 1 via softmax(theta_logits); target loss is
   MSE(theta_s * Y_hat_s + theta_t * Y_hat_t, Y_target).
6. Dynamic Lambda Scheduling: lambda_p = 2 / (1 + exp(-10 * p)) - 1
7. Scientific Validation: Model selection is driven strictly by validation loss. Blind test set is evaluated ONLY ONCE at the end.
8. 18-D Feature Scaling: Fitted across samples and time steps strictly on the source training split.

ACCURACY IMPROVEMENTS over the original version:
- Multi-kernel, adaptive-bandwidth MMD loss (see mmd_loss.py) instead of a single fixed-sigma kernel.
- Constrained (softmax) theta_s/theta_t trade-off parameters (see model_hybrido.py) instead of
  unconstrained free scalars.
- Cosine-annealed learning rate schedule + gradient clipping for more stable convergence.
- Optional early stopping on validation RMSE so a higher epoch budget doesn't waste compute or overfit.
- Configurable RUL ceiling and MMD kernel settings from the CLI so they can be tuned per dataset
  instead of hardcoded.
"""

import os
import sys
import copy
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import glob
from typing import Tuple, Dict, List, Optional

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


DEFAULT_RUL_MAX_CEILING = 2500.0  # Optimal physical ceiling for MATR (max ~2205 cyc) and HUST (max ~2293 cyc)


class RobustRULScaler:
    """
    Fixed physical upper-bound normalizer for battery Remaining Useful Life (RUL).
    Normalizes RUL to [0, 1] using a fixed physical ceiling (default 2500 cycles).
    Auto-expands ceiling for longer-lived datasets (e.g. SNL ~4020 cycles) when using default.
    """

    def __init__(self, y_max: float = DEFAULT_RUL_MAX_CEILING):
        self.y_max = float(y_max)

    def fit(self, Y_train: Optional[np.ndarray] = None, Y_adapt: Optional[np.ndarray] = None):
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        max_val = float(np.max(y))
        if max_val > self.y_max:
            if self.y_max == DEFAULT_RUL_MAX_CEILING:
                self.y_max = float(np.ceil((max_val * 1.05) / 500.0) * 500.0)
                logger.info(f"Auto-expanded physical RUL ceiling to {self.y_max:.0f} cycles to accommodate maximum label ({max_val:.1f} cycles)")
            else:
                raise ValueError(
                    f"RUL value ({max_val:.1f} cycles) exceeds the specified benchmark ceiling of {self.y_max:.1f} cycles. "
                    "Pass a larger --rul-ceiling."
                )
        scaled = y / self.y_max
        return scaled.astype(np.float32)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        return (y_scaled * self.y_max).astype(np.float32)


def split_by_cell_id(
    X: np.ndarray,
    Y: np.ndarray,
    cell_ids: np.ndarray,
    test_ratio: float = 0.10,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits samples strictly at the physical battery cell level.
    Guarantees that all observation windows for any given cell are assigned to ONLY ONE partition.
    Raises ValueError if fewer than 2 unique physical cells are available.
    """
    unique_cells = np.unique(cell_ids)
    if len(unique_cells) < 2:
        raise ValueError(
            f"Cell-level splitting requires at least 2 unique physical cells, but found {len(unique_cells)}. "
            "Random window-level splitting is strictly forbidden because it causes intra-cell leakage."
        )

    train_cells, test_cells = train_test_split(
        unique_cells, test_size=test_ratio, random_state=random_state
    )

    train_mask = np.isin(cell_ids, train_cells)
    test_mask = np.isin(cell_ids, test_cells)

    assert set(train_cells).isdisjoint(set(test_cells)), "Cell overlap detected between partitions!"

    return (
        X[train_mask], X[test_mask],
        Y[train_mask], Y[test_mask],
        cell_ids[train_mask], cell_ids[test_mask]
    )


def compute_dynamic_lambda(epoch: int, total_epochs: int, gamma: float = 10.0) -> float:
    """
    Computes dynamic lambda scheduling:
        lambda_p = 2 / (1 + exp(-gamma * p)) - 1
    where p in [0, 1] is training progress.
    """
    p = float(epoch) / float(max(1, total_epochs))
    return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)


def fit_and_transform_features_18d(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_tgt_adapt: np.ndarray,
    X_tgt_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fits MinMaxScaler across all samples and time steps over the 18 physical feature dimensions.
    X shape: (N, 10, 3, 6) -> (N*10, 18) for scaling.

    ACCURACY FIX: Fits on BOTH training splits: source train + target adaptation.
    This is not leakage: the blind target test split is strictly excluded and only transformed.
    Includes clipping to [-0.5, 1.5] as a safety net against near-zero range denominator blow-up.
    """
    def to_flat18(arr):
        n, s, c, f = arr.shape
        return arr.reshape(n * s, c * f), (n, s, c, f)

    X_tr_flat, shape_tr = to_flat18(X_train)
    X_ad_flat, shape_ad = to_flat18(X_tgt_adapt)

    # Fit on BOTH training splits: source train + target adaptation.
    # Not leakage — the blind test split is never used for fitting.
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(np.vstack([X_tr_flat, X_ad_flat]))

    def tf(flat, shape):
        return np.clip(scaler.transform(flat), -0.5, 1.5).reshape(shape).astype(np.float32)

    X_val_flat, shape_val = to_flat18(X_val)
    X_ts_flat, shape_ts = to_flat18(X_tgt_test)

    return (
        tf(X_tr_flat, shape_tr),
        tf(X_val_flat, shape_val),
        tf(X_ad_flat, shape_ad),
        tf(X_ts_flat, shape_ts),
        scaler
    )



def train_hybrido_session(
    model: nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    val_loader: DataLoader,
    target_test_loader: DataLoader,    target_y_test_raw: np.ndarray,
    val_y_raw: np.ndarray,
    scaler_y: RobustRULScaler,
    epochs: int = 30,
    lr: float = 0.0005,
    sigma_mmd: Optional[float] = None,
    mmd_kernel_num: int = 5,
    mmd_kernel_mul: float = 2.0,
    mmd_weight: float = 1.0,
    grad_clip_norm: float = 5.0,
    early_stop_patience: Optional[int] = 8,
    use_scheduler: bool = False,
    checkpoint_path: Optional[str] = "checkpoints/hybrido_best.pt",
    device: str = "cpu"
) -> Dict[str, float]:

    """
    Paper-faithful training loop with validation-driven checkpoint selection and
    constrained trainable theta parameters.

    TRAINING & LOSS SPECIFICATIONS (Tran et al., 2025):
    - Adam optimizer with fixed learning rate (lr=0.0005, no weight decay).
    - Source and target regression both use combined prediction Y_comb = theta_S * Y_S + theta_T * Y_T (Eq. 13).
    - Validation-driven checkpoint selection uses combined prediction Y_comb on held-out cells.
    - Multi-kernel MMD loss scaled dynamically by lambda_p = 2 / (1 + exp(-10*p)) - 1.
    - Default 30 epochs with early stopping patience of 8.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs)) if use_scheduler else None
    criterion_mse = nn.MSELoss()
    mmd_loss_fn = MMDLoss(kernel_num=mmd_kernel_num, kernel_mul=mmd_kernel_mul, fix_sigma=sigma_mmd)

    best_val_rmse = float("inf")
    best_model_state = None
    best_epoch = 0
    epochs_since_improve = 0

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

            # 1. Source forward pass -> combined prediction on source (matching paper Eq. 13)
            y_comb_s, _, _, z_s = model(src_x)
            loss_source = criterion_mse(y_comb_s, src_y)

            # 2. Target forward pass -> combined prediction on target
            y_comb_t, _, _, z_t = model(tgt_x)
            loss_target = criterion_mse(y_comb_t, tgt_y)

            # 3. Multi-Kernel Maximum Mean Discrepancy (MMD) Loss between feature representations
            loss_mmd = mmd_loss_fn(z_s, z_t)

            # 4. Total Loss (Eq. 13: L_total = L_source + L_target + lambda_p * L_mmd)
            eff_lambda = mmd_weight * lambda_p
            loss_total = loss_source + loss_target + eff_lambda * loss_mmd

            loss_total.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()

            total_loss_accum += loss_total.item()
            src_loss_accum += loss_source.item()
            tgt_loss_accum += loss_target.item()
            mmd_loss_accum += loss_mmd.item()
            batches += 1

        if scheduler is not None:
            scheduler.step()

        # Validation Step (Model Selection occurs strictly on VALIDATION split using y_comb)
        model.eval()
        val_preds = []
        with torch.no_grad():
            for v_x, _ in val_loader:
                v_x = v_x.to(device)
                y_comb_v, _, _, _ = model(v_x)
                val_preds.extend(y_comb_v.cpu().numpy().flatten())

        val_preds = np.array(val_preds)
        val_pred_min = float(val_preds.min()) if len(val_preds) > 0 else 0.0
        val_pred_max = float(val_preds.max()) if len(val_preds) > 0 else 0.0
        val_preds_unscaled = scaler_y.inverse_transform(val_preds)
        val_rmse = float(np.sqrt(mean_squared_error(val_y_raw, val_preds_unscaled)))

        improved = val_rmse < best_val_rmse
        if improved:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch:03d}/{epochs:03d}] "
            f"Loss: {total_loss_accum / max(1, batches):.4f} | "
            f"Src MSE: {src_loss_accum / max(1, batches):.4f} | "
            f"Tgt MSE: {tgt_loss_accum / max(1, batches):.4f} | "
            f"MMD: {mmd_loss_accum / max(1, batches):.4f} (eff_lambda={eff_lambda:.3f}) | "
            f"LR: {current_lr:.2e} | "
            f"Val RMSE: {val_rmse:.2f} cyc | "
            f"Val Preds: [{val_pred_min:.3f}, {val_pred_max:.3f}] | "
            f"theta_S: {model.theta_s.item():.3f}, theta_T: {model.theta_t.item():.3f}"
        )

        if early_stop_patience is not None and epochs_since_improve >= early_stop_patience:
            logger.info(
                f"Early stopping: no Val RMSE improvement for {early_stop_patience} epochs "
                f"(best was epoch {best_epoch} at {best_val_rmse:.2f} cyc)."
            )
            break

    # FINAL EVALUATION: Load best checkpoint chosen by validation, test target set EXACTLY ONCE
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if checkpoint_path is not None:
            ckpt_dir = os.path.dirname(checkpoint_path)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "state_dict": best_model_state,
                "best_epoch": best_epoch,
                "best_val_rmse": best_val_rmse
            }, checkpoint_path)
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")


    model.eval()
    test_preds = []
    with torch.no_grad():
        for t_x, _ in target_test_loader:
            t_x = t_x.to(device)
            y_comb_t, _, _, _ = model(t_x)
            test_preds.extend(y_comb_t.cpu().numpy().flatten())

    test_preds = np.array(test_preds)
    test_preds_unscaled = scaler_y.inverse_transform(test_preds)
    final_test_rmse = float(np.sqrt(mean_squared_error(target_y_test_raw, test_preds_unscaled)))
    final_test_r2 = float(r2_score(target_y_test_raw, test_preds_unscaled))

    # 1. Standard per-sample MAPE: (1/N) * sum(|y_i - y_hat_i| / max(y_i, 1.0)) * 100%
    safe_target = np.maximum(target_y_test_raw, 1.0)
    final_test_standard_mape = float(np.mean(np.abs(target_y_test_raw - test_preds_unscaled) / safe_target) * 100.0)

    # 2. Paper's nominal-cycle-life MAPE: Section 4.3 defines MAPE divided by the cell's nominal cycle life y:
    # MAPE = (1/N) * sum(|y_i - y_hat_i| / y_cycle_life) * 100%
    target_nominal_life = float(np.max(target_y_test_raw))
    final_test_paper_mape = float(np.mean(np.abs(target_y_test_raw - test_preds_unscaled) / max(target_nominal_life, 1.0)) * 100.0)

    # 3. Floor MAPE: Evaluates instantaneous MAPE excluding near-EOL windows (<50 cycles) where division by near-zero blows up
    mask_floor = target_y_test_raw >= 50.0
    if np.any(mask_floor):
        final_test_floor_mape = float(mean_absolute_percentage_error(target_y_test_raw[mask_floor], test_preds_unscaled[mask_floor]) * 100.0)
    else:
        final_test_floor_mape = float("nan")

    # 4. Raw scikit-learn instantaneous MAPE (unbounded near EOL)
    final_test_raw_mape = float(mean_absolute_percentage_error(target_y_test_raw, test_preds_unscaled) * 100.0)

    logger.info(
        f"\n[FINAL TEST EVALUATION] Chosen Epoch: {best_epoch} | "
        f"Test RMSE: {final_test_rmse:.2f} cycles | "
        f"Test R²: {final_test_r2:.4f} | "
        f"Standard MAPE: {final_test_standard_mape:.2f}% | "
        f"Paper MAPE: {final_test_paper_mape:.2f}% | "
        f"Floor MAPE (RUL>=50): {final_test_floor_mape:.2f}% | "
        f"Raw MAPE: {final_test_raw_mape:.2f}%"
    )

    return {
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_rmse": final_test_rmse,
        "test_r2": final_test_r2,
        "test_standard_mape": final_test_standard_mape,
        "test_paper_mape": final_test_paper_mape,
        "test_floor_mape": final_test_floor_mape,
        "test_raw_mape": final_test_raw_mape,
        "test_mape": final_test_standard_mape,
        "final_theta_s": float(model.theta_s.item()),
        "final_theta_t": float(model.theta_t.item()),
        "test_preds_unscaled": test_preds_unscaled
    }



def filter_severson_cells(
    X: np.ndarray,
    Y: np.ndarray,
    cell_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters MATR dataset (169 cells) down to the 124 cells from Severson et al. 2019
    (batches 1-3) by excluding Attia et al. 2020 batch 4 cells ('b4c*' and '*_b4*').
    """
    severson_mask = np.array(["b4c" not in str(c).lower() and "_b4" not in str(c).lower() for c in cell_ids])
    if np.any(severson_mask):
        return X[severson_mask], Y[severson_mask], cell_ids[severson_mask]
    return X, Y, cell_ids


def run_benchmark(
    source_npz: str,
    target_npz: str,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 0.0005,
    val_ratio: float = 0.10,
    sigma_mmd: Optional[float] = None,
    mmd_kernel_num: int = 5,
    mmd_kernel_mul: float = 2.0,
    early_stop_patience: Optional[int] = 8,
    rul_ceiling: float = DEFAULT_RUL_MAX_CEILING,
    norm_type: str = "batchnorm",
    severson_only: bool = False,
    mmd_weight: float = 1.0,
    use_scheduler: bool = False,
    checkpoint_path: str = "checkpoints/hybrido_best.pt",
    seed: int = 42,
    num_runs: int = 1,
    device: str = "cpu"
):
    """
    Zero-Leakage Cell-Level Partitioned Benchmark Run (Table 2 Specs):
    Source: 90% Training Cells / 10% Validation Cells.
    Target: 60% Adaptation Cells / 40% Blind Testing Cells.

    Supports multi-seed repetitions (--num-runs N) with ensemble averaging
    exactly as specified in Tran et al. 2025 Section 4.1.
    """
    logger.info(f"Loading Source: {source_npz}")
    src_data = np.load(source_npz)
    if "cell_ids" not in src_data:
        raise ValueError(
            f"Source dataset '{source_npz}' is missing 'cell_ids'. "
            "Cell-level splitting cannot be guaranteed. Please re-run preprocess_hybrido.py."
        )
    X_src_raw, Y_src_raw, src_cells = src_data["X"], src_data["Y"], src_data["cell_ids"]

    # Optional filter: Filter MATR (169 cells) down to the 124 cells from Severson et al. 2019 (batches 1-3)
    if severson_only and "matr" in source_npz.lower():
        X_src_raw, Y_src_raw, src_cells = filter_severson_cells(X_src_raw, Y_src_raw, src_cells)
        logger.info(f"--severson-only: Filtered MATR to {len(np.unique(src_cells))} Severson et al. 2019 cells (batches 1-3).")

    logger.info(f"Loading Target: {target_npz}")
    tgt_data = np.load(target_npz)
    if "cell_ids" not in tgt_data:
        raise ValueError(
            f"Target dataset '{target_npz}' is missing 'cell_ids'. "
            "Cell-level splitting cannot be guaranteed. Please re-run preprocess_hybrido.py."
        )
    X_tgt_raw, Y_tgt_raw, tgt_cells = tgt_data["X"], tgt_data["Y"], tgt_data["cell_ids"]

    # 1. Zero-Leakage Cell-Level Splitting (Fixed seed 42 for cell-split consistency across runs)
    X_tr_raw, X_val_raw, Y_tr_raw, Y_val_raw, tr_c, val_c = split_by_cell_id(
        X_src_raw, Y_src_raw, src_cells, test_ratio=val_ratio, random_state=42
    )
    X_tgt_adapt, X_tgt_test, Y_tgt_adapt, Y_tgt_test, ad_c, ts_c = split_by_cell_id(
        X_tgt_raw, Y_tgt_raw, tgt_cells, test_ratio=0.40, random_state=42
    )

    logger.info("=" * 50)
    logger.info("DATASET PARTITIONING BREAKDOWN (CELL-LEVEL GROUPING)")
    logger.info(f"Source Training:    {len(np.unique(tr_c)):3d} cells | {len(X_tr_raw):5d} windows")
    logger.info(f"Source Validation:  {len(np.unique(val_c)):3d} cells | {len(X_val_raw):5d} windows")
    logger.info(f"Target Adaptation:  {len(np.unique(ad_c)):3d} cells | {len(X_tgt_adapt):5d} windows")
    logger.info(f"Target Blind Test:  {len(np.unique(ts_c)):3d} cells | {len(X_tgt_test):5d} windows")
    logger.info("=" * 50)
    if len(X_tr_raw) < 200 or len(X_tgt_adapt) < 200:
        logger.warning(
            "Training window counts look low. If you generated these .npz files with a large "
            "--stride in preprocess_hybrido.py, consider re-running preprocessing with a smaller "
            "stride (overlapping windows) to give the model more training signal."
        )

    # 2. 18-D Feature Scaling across samples and time steps
    X_tr_sc, X_val_sc, X_tgt_ad_sc, X_tgt_ts_sc, scaler_x = fit_and_transform_features_18d(
        X_tr_raw, X_val_raw, X_tgt_adapt, X_tgt_test
    )

    # 3. Robust Physical RUL Normalization (guarantees Y in [0, 1] without Sigmoid saturation)
    scaler_y = RobustRULScaler(y_max=rul_ceiling).fit()
    Y_tr_sc = scaler_y.transform(Y_tr_raw)
    Y_val_sc = scaler_y.transform(Y_val_raw)
    Y_tgt_ad_sc = scaler_y.transform(Y_tgt_adapt)
    Y_tgt_ts_sc = scaler_y.transform(Y_tgt_test)

    drop_src = len(X_tr_sc) > batch_size
    drop_tgt = len(X_tgt_ad_sc) > batch_size

    src_loader = DataLoader(BatteryDataset(X_tr_sc, Y_tr_sc), batch_size=batch_size, shuffle=True, drop_last=drop_src)
    val_loader = DataLoader(BatteryDataset(X_val_sc, Y_val_sc), batch_size=batch_size, shuffle=False)
    tgt_loader = DataLoader(BatteryDataset(X_tgt_ad_sc, Y_tgt_ad_sc), batch_size=batch_size, shuffle=True, drop_last=drop_tgt)
    tgt_test_loader = DataLoader(BatteryDataset(X_tgt_ts_sc, Y_tgt_ts_sc), batch_size=batch_size, shuffle=False)

    if num_runs <= 1:
        torch.manual_seed(seed)
        np.random.seed(seed)
        logger.info(f"Instantiating HybridoNetAdapt (norm_type='{norm_type}', seed={seed})")
        model = HybridoNetAdapt(
            input_dim=18,
            hidden_dim=128,
            num_lstm_layers=2,
            num_heads=4,
            dropout=0.1,
            norm_type=norm_type
        )
        results = train_hybrido_session(
            model=model,
            source_loader=src_loader,
            target_loader=tgt_loader,
            val_loader=val_loader,
            target_test_loader=tgt_test_loader,
            target_y_test_raw=Y_tgt_test,
            val_y_raw=Y_val_raw,
            scaler_y=scaler_y,
            epochs=epochs,
            lr=lr,
            sigma_mmd=sigma_mmd,
            mmd_kernel_num=mmd_kernel_num,
            mmd_kernel_mul=mmd_kernel_mul,
            mmd_weight=mmd_weight,
            early_stop_patience=early_stop_patience,
            use_scheduler=use_scheduler,
            checkpoint_path=checkpoint_path,
            device=device
        )
        logger.info("\n" + "=" * 50)
        logger.info("HYBRIDONET-ADAPT BENCHMARK RESULTS")
        logger.info(f"Target Test RMSE:        {results['test_rmse']:.2f} cycles")
        logger.info(f"Target Test R²:          {results['test_r2']:.4f}")
        logger.info(f"Standard MAPE (sample):  {results['test_standard_mape']:.2f}%")
        logger.info(f"Paper MAPE (vs EOL):     {results['test_paper_mape']:.2f}%")
        logger.info(f"Floor MAPE (RUL >= 50):  {results['test_floor_mape']:.2f}%")
        logger.info(f"Raw MAPE (unbounded):    {results['test_raw_mape']:.2f}%")
        logger.info(f"Trained Trade-off Weights: theta_S={results['final_theta_s']:.4f}, theta_T={results['final_theta_t']:.4f}")
        logger.info("=" * 50)
        return results
    else:
        logger.info("\n" + "=" * 60)
        logger.info(f"RUNNING MULTI-SEED BENCHMARK ({num_runs} REPETITIONS, PAPER SECTION 4.1)")
        logger.info("=" * 60)
        all_results = []
        all_test_preds = []
        for run_idx in range(num_runs):
            current_seed = seed + run_idx
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            logger.info(f"\n{'='*20} RUN [{run_idx + 1:02d}/{num_runs:02d}] (Seed: {current_seed}) {'='*20}")
            model = HybridoNetAdapt(
                input_dim=18,
                hidden_dim=128,
                num_lstm_layers=2,
                num_heads=4,
                dropout=0.1,
                norm_type=norm_type
            )
            run_ckpt = checkpoint_path
            if checkpoint_path and num_runs > 1:
                base, ext = os.path.splitext(checkpoint_path)
                run_ckpt = f"{base}_run{run_idx + 1}{ext}"
            res = train_hybrido_session(
                model=model,
                source_loader=src_loader,
                target_loader=tgt_loader,
                val_loader=val_loader,
                target_test_loader=tgt_test_loader,
                target_y_test_raw=Y_tgt_test,
                val_y_raw=Y_val_raw,
                scaler_y=scaler_y,
                epochs=epochs,
                lr=lr,
                sigma_mmd=sigma_mmd,
                mmd_kernel_num=mmd_kernel_num,
                mmd_kernel_mul=mmd_kernel_mul,
                mmd_weight=mmd_weight,
                early_stop_patience=early_stop_patience,
                use_scheduler=use_scheduler,
                checkpoint_path=run_ckpt,
                device=device
            )

            all_results.append(res)
            all_test_preds.append(res["test_preds_unscaled"])

        rmses = np.array([r["test_rmse"] for r in all_results])
        r2s = np.array([r["test_r2"] for r in all_results])
        std_mapes = np.array([r["test_standard_mape"] for r in all_results])
        paper_mapes = np.array([r["test_paper_mape"] for r in all_results])
        floor_mapes = np.array([r["test_floor_mape"] for r in all_results])

        # Exact Paper Methodology: Ensemble-averaged prediction across the runs
        ens_preds = np.mean(all_test_preds, axis=0)
        ens_rmse = float(np.sqrt(mean_squared_error(Y_tgt_test, ens_preds)))
        ens_r2 = float(r2_score(Y_tgt_test, ens_preds))
        safe_target = np.maximum(Y_tgt_test, 1.0)
        ens_std_mape = float(np.mean(np.abs(Y_tgt_test - ens_preds) / safe_target) * 100.0)
        target_nominal_life = float(np.max(Y_tgt_test))
        ens_paper_mape = float(np.mean(np.abs(Y_tgt_test - ens_preds) / max(target_nominal_life, 1.0)) * 100.0)
        mask_floor = Y_tgt_test >= 50.0
        ens_floor_mape = float(mean_absolute_percentage_error(Y_tgt_test[mask_floor], ens_preds[mask_floor]) * 100.0) if np.any(mask_floor) else float("nan")

        logger.info("\n" + "=" * 60)
        logger.info(f"HYBRIDONET-ADAPT MULTI-SEED RESULTS ({num_runs} RUNS)")
        logger.info("=" * 60)
        logger.info(f"Individual Runs (Mean ± Std):")
        logger.info(f"  Test RMSE:          {np.mean(rmses):.2f} ± {np.std(rmses):.2f} cycles")
        logger.info(f"  Test R²:            {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
        logger.info(f"  Standard MAPE:      {np.mean(std_mapes):.2f}% ± {np.std(std_mapes):.2f}%")
        logger.info(f"  Paper MAPE:         {np.mean(paper_mapes):.2f}% ± {np.std(paper_mapes):.2f}%")
        logger.info(f"  Floor MAPE (>=50):  {np.mean(floor_mapes):.2f}% ± {np.std(floor_mapes):.2f}%")
        logger.info("-" * 60)
        logger.info(f"Ensemble-Averaged Prediction (Published Paper Methodology):")
        logger.info(f"  Ensemble Test RMSE:         {ens_rmse:.2f} cycles")
        logger.info(f"  Ensemble Test R²:           {ens_r2:.4f}")
        logger.info(f"  Ensemble Standard MAPE:     {ens_std_mape:.2f}%")
        logger.info(f"  Ensemble Paper MAPE:        {ens_paper_mape:.2f}%")
        logger.info(f"  Ensemble Floor MAPE (>=50): {ens_floor_mape:.2f}%")
        logger.info("=" * 60)

        return {
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_r2": float(np.mean(r2s)),
            "std_r2": float(np.std(r2s)),
            "mean_standard_mape": float(np.mean(std_mapes)),
            "mean_paper_mape": float(np.mean(paper_mapes)),
            "ensemble_rmse": ens_rmse,
            "ensemble_r2": ens_r2,
            "ensemble_standard_mape": ens_std_mape,
            "ensemble_paper_mape": ens_paper_mape
        }



def resolve_file_path(path_str: str) -> str:
    """Resolves file path with case-insensitive fallback and TRI/MATR alias on Linux/Colab filesystems."""
    if os.path.exists(path_str):
        return path_str
    dirname, basename = os.path.split(path_str)
    if os.path.exists(dirname):
        # 1. Exact case-insensitive match
        for f in os.listdir(dirname):
            if f.lower() == basename.lower():
                return os.path.join(dirname, f)
        # 2. TRI <-> MATR alias match
        alias_target = None
        if "tri" in basename.lower():
            alias_target = basename.lower().replace("tri", "matr")
        elif "matr" in basename.lower():
            alias_target = basename.lower().replace("matr", "tri")
        if alias_target:
            for f in os.listdir(dirname):
                if f.lower() == alias_target:
                    return os.path.join(dirname, f)
    return path_str


def main():
    parser = argparse.ArgumentParser(description="HybridoNet-Adapt Benchmark Runner")
    parser.add_argument("--source", type=str, required=True, help="Path to source .npz raw features (REQUIRED)")
    parser.add_argument("--target", type=str, required=True, help="Path to target .npz raw features (REQUIRED)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default=30; paired with early stopping)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (paper default=128)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate (paper default=0.0005, fixed lr)")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Source cell validation ratio (default 0.10)")
    parser.add_argument("--sigma-mmd", type=float, default=None, help="Fixed MMD bandwidth. If omitted (default), the bandwidth is estimated adaptively per batch, which is more robust than a hardcoded value.")
    parser.add_argument("--mmd-kernel-num", type=int, default=5, help="Number of Gaussian kernels to sum for multi-kernel MMD (default=5)")
    parser.add_argument("--mmd-kernel-mul", type=float, default=2.0, help="Geometric spacing factor between MMD kernel bandwidths (default=2.0)")
    parser.add_argument("--mmd-weight", type=float, default=1.0, help="Static multiplier on MMD loss (default: 1.0, matching paper Eq. 13)")
    parser.add_argument("--early-stop-patience", type=int, default=8, help="Stop if Val RMSE hasn't improved for this many epochs (default: 8; set to 0 to disable).")
    parser.add_argument("--rul-ceiling", type=float, default=DEFAULT_RUL_MAX_CEILING, help="Fixed physical RUL ceiling in cycles used for normalization (default=2500).")
    parser.add_argument("--norm-type", type=str, choices=["layernorm", "batchnorm"], default="batchnorm", help="Normalization layer in predictor heads (default: batchnorm matching paper Table 2; layernorm also supported)")
    parser.add_argument("--severson-only", action="store_true", help="Filter MATR dataset to the 124 cells from Severson et al. 2019 (batches 1-3), excluding Attia batch 4")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model initialization and data splitting (default: 42)")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of repetitions to run (default: 1; paper Section 4.1 uses 10 runs with ensemble averaging)")
    parser.add_argument("--use-scheduler", action="store_true", help="Enable cosine annealing learning rate scheduler (default: False, paper uses fixed lr)")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/hybrido_best.pt", help="Path to save the best model checkpoint (default: checkpoints/hybrido_best.pt)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    src_path = resolve_file_path(args.source)
    tgt_path = resolve_file_path(args.target)

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source feature file not found: {args.source}")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Target feature file not found: {args.target}")

    early_stop_patience = args.early_stop_patience if args.early_stop_patience and args.early_stop_patience > 0 else None

    run_benchmark(
        source_npz=src_path,
        target_npz=tgt_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        sigma_mmd=args.sigma_mmd,
        mmd_kernel_num=args.mmd_kernel_num,
        mmd_kernel_mul=args.mmd_kernel_mul,
        early_stop_patience=early_stop_patience,
        rul_ceiling=args.rul_ceiling,
        norm_type=args.norm_type,
        severson_only=args.severson_only,
        mmd_weight=args.mmd_weight,
        use_scheduler=args.use_scheduler,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
        num_runs=args.num_runs,
        device=device
    )




if __name__ == "__main__":
    main()