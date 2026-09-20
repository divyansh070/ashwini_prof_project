#!/usr/bin/env python3
"""
HybridoNet Cross-Chemistry Training Pipeline (Stage 0).

Implements:
1. Dynamic Feature Dimensionality: dynamically reads feature dimension (18, 30, 36) from .npz.
2. Robust Physical RUL Normalization:
   - Fixes scaler bug: locks physical ceiling across all splits during fit() so y_max
     is strictly immutable during transform(), preventing silent mis-scaling across splits.
3. Combined Objective: L_total = L_MSE(Source) + L_MSE(Target_Combined) + lambda(p) * L_MMD
4. Multi-kernel adaptive MMD loss imported directly from hybridonet-adapt.
5. Combined source-train + target-adapt MinMaxScaler with [-0.5, 1.5] clipping.
6. Model selection strictly on validation RMSE with canary logging: Val Preds [min, max].
7. Default norm-type: layernorm (prevents cross-domain running statistic contamination).
"""

import os
import sys
import copy
import argparse
import logging
import json
from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# Add current directory, hybridonet-adapt directory, and project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
hybridonet_dir = os.path.join(os.path.dirname(current_dir), "hybridonet-adapt")

for p in [current_dir, hybridonet_dir, project_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_xchem import HybridoNetAdapt
from mmd_loss import MMDLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [XChemTrain] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("XChemTrain")


class BatteryDataset(Dataset):
    """PyTorch Dataset for battery tensor samples."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


DEFAULT_RUL_MAX_CEILING = 2500.0


class RobustRULScaler:
    """
    Fixed physical upper-bound normalizer for battery Remaining Useful Life (RUL).
    Normalizes RUL to [0, 1] using a fixed physical ceiling (default 2500 cycles).

    BUG FIX (Stage 0):
    Ceiling is determined once during fit() across all dataset splits up front.
    Once fit(), self.y_max is strictly locked and immutable during transform().
    Any label exceeding the ceiling raises a loud ValueError instead of silently
    mutating self.y_max mid-sequence.
    """

    def __init__(self, y_max: float = DEFAULT_RUL_MAX_CEILING):
        self.initial_y_max = float(y_max)
        self.y_max = float(y_max)
        self.fitted = False

    def fit(self, *y_arrays: np.ndarray):
        all_max = 0.0
        for arr in y_arrays:
            if arr is not None and len(arr) > 0:
                all_max = max(all_max, float(np.max(arr)))

        if all_max > self.initial_y_max:
            if self.initial_y_max == DEFAULT_RUL_MAX_CEILING:
                self.y_max = float(np.ceil((all_max * 1.05) / 500.0) * 500.0)
                logger.info(
                    f"Auto-expanded physical RUL ceiling to {self.y_max:.0f} cycles "
                    f"to accommodate dataset maximum label ({all_max:.1f} cycles) across all splits."
                )
            else:
                raise ValueError(
                    f"RUL value ({all_max:.1f} cycles) exceeds specified benchmark ceiling of "
                    f"{self.initial_y_max:.1f} cycles. Pass a larger --rul-ceiling."
                )
        else:
            self.y_max = self.initial_y_max
        self.fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("RobustRULScaler must be fit() on dataset splits before calling transform().")
        max_val = float(np.max(y)) if len(y) > 0 else 0.0
        if max_val > self.y_max:
            raise ValueError(
                f"RUL value ({max_val:.1f} cycles) exceeds the locked physical ceiling of {self.y_max:.1f} cycles. "
                "Pass a larger --rul-ceiling or pass all dataset splits to fit()."
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
    """Computes dynamic lambda scheduling: lambda_p = 2 / (1 + exp(-gamma * p)) - 1."""
    p = float(epoch) / float(max(1, total_epochs))
    return float(2.0 / (1.0 + np.exp(-gamma * p)) - 1.0)


def fit_and_transform_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_tgt_adapt: np.ndarray,
    X_tgt_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Fits MinMaxScaler across all samples and time steps over the physical feature dimensions.
    Dynamically handles 4D tensors (N, 10, C, F) or 3D tensors (N, 10, D).

    Preserves verified bug fix:
    - Fits on BOTH training splits: source train + target adaptation.
    - Blind target test split is strictly excluded and only transformed.
    - Includes clipping to [-0.5, 1.5] as a safety net against near-zero range denominator blow-up.
    """
    def to_flat(arr: np.ndarray):
        shape = arr.shape
        n, s = shape[0], shape[1]
        feat_dim = int(np.prod(shape[2:]))
        return arr.reshape(n * s, feat_dim), shape

    X_tr_flat, shape_tr = to_flat(X_train)
    X_ad_flat, shape_ad = to_flat(X_tgt_adapt)

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(np.vstack([X_tr_flat, X_ad_flat]))

    def tf(flat, shape):
        return np.clip(scaler.transform(flat), -0.5, 1.5).reshape(shape).astype(np.float32)

    X_val_flat, shape_val = to_flat(X_val)
    X_ts_flat, shape_ts = to_flat(X_tgt_test)

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
    target_test_loader: DataLoader,
    target_y_test_raw: np.ndarray,
    val_y_raw: np.ndarray,
    scaler_y: RobustRULScaler,
    epochs: int = 60,
    lr: float = 0.0005,
    sigma_mmd: Optional[float] = None,
    mmd_kernel_num: int = 5,
    mmd_kernel_mul: float = 2.0,
    mmd_weight: float = 0.1,
    grad_clip_norm: float = 5.0,
    early_stop_patience: Optional[int] = 15,
    checkpoint_path: Optional[str] = "checkpoints/xchem_best.pt",
    device: str = "cpu"
) -> Dict[str, float]:
    """Training loop with validation-driven checkpoint selection and constrained trade-off parameters."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
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

            # 1. Source forward pass -> predicts y_hat_s
            _, y_pred_s, _, z_s = model(src_x)
            loss_source = criterion_mse(y_pred_s, src_y)

            # 2. Target forward pass -> predicts combined Y_hat_T = theta_s * y_s + theta_t * y_t
            y_comb_t, _, _, z_t = model(tgt_x)
            loss_target = criterion_mse(y_comb_t, tgt_y)

            # 3. Multi-Kernel MMD Loss between latent representations
            loss_mmd = mmd_loss_fn(z_s, z_t)

            # 4. Total Loss
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

        scheduler.step()

        # Validation Step
        model.eval()
        val_preds = []
        with torch.no_grad():
            for v_x, _ in val_loader:
                v_x = v_x.to(device)
                _, y_pred_s, _, _ = model(v_x)
                val_preds.extend(y_pred_s.cpu().numpy().flatten())

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
        # CANARY LOGGING: Val Preds: [min, max] monitored every epoch to catch saturation collapse
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

    # Final Evaluation: Load best checkpoint chosen strictly by validation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if checkpoint_path is not None:
            ckpt_dir = os.path.dirname(checkpoint_path)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "state_dict": best_model_state,
                "best_epoch": best_epoch,
                "best_val_rmse": best_val_rmse,
                "input_dim": getattr(model, "feature_extractor", model).input_dim
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

    target_nominal_life = float(np.max(target_y_test_raw))
    final_test_paper_mape = float(
        np.mean(np.abs(target_y_test_raw - test_preds_unscaled) / max(target_nominal_life, 1.0)) * 100.0
    )

    mask_floor = target_y_test_raw >= 50.0
    if np.any(mask_floor):
        final_test_floor_mape = float(
            mean_absolute_percentage_error(target_y_test_raw[mask_floor], test_preds_unscaled[mask_floor]) * 100.0
        )
    else:
        final_test_floor_mape = float("nan")

    final_test_raw_mape = float(mean_absolute_percentage_error(target_y_test_raw, test_preds_unscaled) * 100.0)

    logger.info(
        f"\n[FINAL TEST EVALUATION] Chosen Epoch: {best_epoch} | "
        f"Test RMSE: {final_test_rmse:.2f} cycles | "
        f"Test R²: {final_test_r2:.4f} | "
        f"Paper MAPE: {final_test_paper_mape:.2f}% | "
        f"Floor MAPE (RUL>=50): {final_test_floor_mape:.2f}% | "
        f"Raw MAPE: {final_test_raw_mape:.2f}%"
    )

    return {
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "test_rmse": final_test_rmse,
        "test_r2": final_test_r2,
        "test_paper_mape": final_test_paper_mape,
        "test_floor_mape": final_test_floor_mape,
        "test_raw_mape": final_test_raw_mape,
        "test_mape": final_test_paper_mape,
        "final_theta_s": float(model.theta_s.item()),
        "final_theta_t": float(model.theta_t.item()),
        "test_preds_unscaled": test_preds_unscaled
    }


def filter_severson_cells(
    X: np.ndarray,
    Y: np.ndarray,
    cell_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filters MATR dataset (169 cells) down to the 124 cells from Severson et al. 2019."""
    severson_mask = np.array(["b4c" not in str(c).lower() and "_b4" not in str(c).lower() for c in cell_ids])
    if np.any(severson_mask):
        return X[severson_mask], Y[severson_mask], cell_ids[severson_mask]
    return X, Y, cell_ids


def resolve_file_path(path_str: str) -> str:
    """Resolves file path with case-insensitive fallback and TRI/MATR alias on Linux/Colab."""
    if os.path.exists(path_str):
        return path_str
    dirname, basename = os.path.split(path_str)
    if os.path.exists(dirname):
        for f in os.listdir(dirname):
            if f.lower() == basename.lower():
                return os.path.join(dirname, f)
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


def run_benchmark(
    source_npz: str,
    target_npz: str,
    epochs: int = 60,
    batch_size: int = 128,
    lr: float = 0.0005,
    val_ratio: float = 0.10,
    sigma_mmd: Optional[float] = None,
    mmd_kernel_num: int = 5,
    mmd_kernel_mul: float = 2.0,
    early_stop_patience: Optional[int] = 15,
    rul_ceiling: float = DEFAULT_RUL_MAX_CEILING,
    norm_type: str = "layernorm",
    severson_only: bool = False,
    mmd_weight: float = 0.1,
    checkpoint_path: str = "checkpoints/xchem_best.pt",
    seed: int = 42,
    num_runs: int = 1,
    device: str = "cpu"
) -> Dict:
    """Zero-Leakage Cell-Level Partitioned Benchmark Run."""
    logger.info(f"Loading Source: {source_npz}")
    src_data = np.load(source_npz)
    if "cell_ids" not in src_data:
        raise ValueError(f"Source dataset '{source_npz}' is missing 'cell_ids'.")
    X_src_raw, Y_src_raw, src_cells = src_data["X"], src_data["Y"], src_data["cell_ids"]

    if severson_only and "matr" in source_npz.lower():
        X_src_raw, Y_src_raw, src_cells = filter_severson_cells(X_src_raw, Y_src_raw, src_cells)
        logger.info(f"--severson-only: Filtered MATR to {len(np.unique(src_cells))} Severson et al. 2019 cells.")

    logger.info(f"Loading Target: {target_npz}")
    tgt_data = np.load(target_npz)
    if "cell_ids" not in tgt_data:
        raise ValueError(f"Target dataset '{target_npz}' is missing 'cell_ids'.")
    X_tgt_raw, Y_tgt_raw, tgt_cells = tgt_data["X"], tgt_data["Y"], tgt_data["cell_ids"]

    # 1. Dynamic Feature Dimensionality Detection
    if "num_features" in src_data:
        input_dim = int(src_data["num_features"])
    else:
        input_dim = int(np.prod(X_src_raw.shape[2:]))

    tgt_dim = int(tgt_data["num_features"]) if "num_features" in tgt_data else int(np.prod(X_tgt_raw.shape[2:]))
    if input_dim != tgt_dim:
        raise ValueError(f"Feature dimension mismatch: source is {input_dim}-D, target is {tgt_dim}-D.")
    logger.info(f"Detected dynamic feature dimensionality: {input_dim}-D across source and target.")

    # 2. Strict Cell-Level Splitting (Fixed seed 42 for cell-split consistency across runs)
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

    # 3. Dynamic Feature Scaling across samples and time steps
    X_tr_sc, X_val_sc, X_tgt_ad_sc, X_tgt_ts_sc, scaler_x = fit_and_transform_features(
        X_tr_raw, X_val_raw, X_tgt_adapt, X_tgt_test
    )

    # 4. Robust Physical RUL Normalization (zero leakage: fits strictly on training splits)
    scaler_y = RobustRULScaler(y_max=rul_ceiling).fit(Y_tr_raw, Y_val_raw, Y_tgt_adapt)
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
        logger.info(f"Instantiating HybridoNetAdapt (input_dim={input_dim}, norm_type='{norm_type}', seed={seed})")
        model = HybridoNetAdapt(
            input_dim=input_dim,
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
            checkpoint_path=checkpoint_path,
            device=device
        )
        results["input_dim"] = input_dim
        results["num_runs"] = 1
        results["mean_rmse"] = results["test_rmse"]
        results["std_rmse"] = 0.0
        results["mean_r2"] = results["test_r2"]
        results["std_r2"] = 0.0
        results["mean_paper_mape"] = results["test_paper_mape"]
        results["std_paper_mape"] = 0.0
        results["ensemble_rmse"] = results["test_rmse"]
        results["ensemble_r2"] = results["test_r2"]
        results["ensemble_paper_mape"] = results["test_paper_mape"]
        return results
    else:
        logger.info("\n" + "=" * 60)
        logger.info(f"RUNNING MULTI-SEED BENCHMARK ({num_runs} REPETITIONS)")
        logger.info("=" * 60)
        all_results = []
        all_test_preds = []
        for run_idx in range(num_runs):
            current_seed = seed + run_idx
            torch.manual_seed(current_seed)
            np.random.seed(current_seed)
            logger.info(f"\n{'='*20} RUN [{run_idx + 1:02d}/{num_runs:02d}] (Seed: {current_seed}) {'='*20}")
            model = HybridoNetAdapt(
                input_dim=input_dim,
                hidden_dim=128,
                num_lstm_layers=2,
                num_heads=4,
                dropout=0.1,
                norm_type=norm_type
            )
            run_ckpt = checkpoint_path
            if checkpoint_path:
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
                checkpoint_path=run_ckpt,
                device=device
            )
            all_results.append(res)
            all_test_preds.append(res["test_preds_unscaled"])

        rmses = np.array([r["test_rmse"] for r in all_results])
        r2s = np.array([r["test_r2"] for r in all_results])
        paper_mapes = np.array([r["test_paper_mape"] for r in all_results])
        floor_mapes = np.array([r["test_floor_mape"] for r in all_results])

        # Ensemble-averaged prediction across the runs
        ens_preds = np.mean(all_test_preds, axis=0)
        ens_rmse = float(np.sqrt(mean_squared_error(Y_tgt_test, ens_preds)))
        ens_r2 = float(r2_score(Y_tgt_test, ens_preds))
        target_nominal_life = float(np.max(Y_tgt_test))
        ens_paper_mape = float(np.mean(np.abs(Y_tgt_test - ens_preds) / max(target_nominal_life, 1.0)) * 100.0)
        mask_floor = Y_tgt_test >= 50.0
        ens_floor_mape = float(
            mean_absolute_percentage_error(Y_tgt_test[mask_floor], ens_preds[mask_floor]) * 100.0
        ) if np.any(mask_floor) else float("nan")

        logger.info("\n" + "=" * 60)
        logger.info(f"HYBRIDONET MULTI-SEED RESULTS ({num_runs} RUNS, {input_dim}-D)")
        logger.info("=" * 60)
        logger.info(f"Individual Runs (Mean ± Std):")
        logger.info(f"  Test RMSE:          {np.mean(rmses):.2f} ± {np.std(rmses):.2f} cycles")
        logger.info(f"  Test R²:            {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
        logger.info(f"  Paper MAPE:         {np.mean(paper_mapes):.2f}% ± {np.std(paper_mapes):.2f}%")
        logger.info(f"  Floor MAPE (>=50):  {np.mean(floor_mapes):.2f}% ± {np.std(floor_mapes):.2f}%")
        logger.info("-" * 60)
        logger.info(f"Ensemble-Averaged Prediction:")
        logger.info(f"  Ensemble Test RMSE:         {ens_rmse:.2f} cycles")
        logger.info(f"  Ensemble Test R²:           {ens_r2:.4f}")
        logger.info(f"  Ensemble Paper MAPE:        {ens_paper_mape:.2f}%")
        logger.info(f"  Ensemble Floor MAPE (>=50): {ens_floor_mape:.2f}%")
        logger.info("=" * 60)

        # Pick best individual run checkpoint and save as canonical checkpoint
        best_run_idx = int(np.argmin(rmses))
        best_ckpt = f"{os.path.splitext(checkpoint_path)[0]}_run{best_run_idx + 1}{os.path.splitext(checkpoint_path)[1]}"
        if os.path.exists(best_ckpt):
            import shutil
            shutil.copyfile(best_ckpt, checkpoint_path)
            logger.info(f"Copied best individual run ({best_run_idx + 1}) checkpoint to {checkpoint_path}")

        summary = {
            "input_dim": input_dim,
            "num_runs": num_runs,
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_r2": float(np.mean(r2s)),
            "std_r2": float(np.std(r2s)),
            "mean_paper_mape": float(np.mean(paper_mapes)),
            "std_paper_mape": float(np.std(paper_mapes)),
            "ensemble_rmse": ens_rmse,
            "ensemble_r2": ens_r2,
            "ensemble_paper_mape": ens_paper_mape,
            "individual_runs": [
                {
                    "run": i + 1,
                    "seed": seed + i,
                    "rmse": float(rmses[i]),
                    "r2": float(r2s[i]),
                    "paper_mape": float(paper_mapes[i])
                }
                for i in range(num_runs)
            ]
        }
        return summary


def main():
    parser = argparse.ArgumentParser(description="Cross-Chemistry HybridoNet Benchmark Runner (Stage 0)")
    parser.add_argument("--source", type=str, required=True, help="Path to source .npz raw features")
    parser.add_argument("--target", type=str, required=True, help="Path to target .npz raw features")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs (default: 60)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate (default: 0.0005)")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Source validation cell ratio (default: 0.10)")
    parser.add_argument("--sigma-mmd", type=float, default=None, help="Fixed MMD bandwidth (None for adaptive)")
    parser.add_argument("--mmd-kernel-num", type=int, default=5, help="Number of kernels for MMD (default: 5)")
    parser.add_argument("--mmd-kernel-mul", type=float, default=2.0, help="Spacing factor for MMD kernels (default: 2.0)")
    parser.add_argument("--mmd-weight", type=float, default=0.1, help="MMD loss weight multiplier (default: 0.1)")
    parser.add_argument("--early-stop-patience", type=int, default=15, help="Early stopping patience (default: 15, 0 to disable)")
    parser.add_argument("--rul-ceiling", type=float, default=DEFAULT_RUL_MAX_CEILING, help="RUL ceiling in cycles (default: 2500)")
    parser.add_argument("--norm-type", type=str, choices=["layernorm", "batchnorm"], default="layernorm", help="Predictor head norm (default: layernorm)")
    parser.add_argument("--severson-only", action="store_true", help="Filter MATR down to 124 Severson et al. 2019 cells")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of repetitions (default: 1)")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/xchem_best.pt", help="Best model save path")
    parser.add_argument("--results-json", type=str, default=None, help="Optional path to write run metrics JSON")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    src_path = resolve_file_path(args.source)
    tgt_path = resolve_file_path(args.target)

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source feature file not found: {args.source}")
    if not os.path.exists(tgt_path):
        raise FileNotFoundError(f"Target feature file not found: {args.target}")

    early_stop = args.early_stop_patience if args.early_stop_patience and args.early_stop_patience > 0 else None

    results = run_benchmark(
        source_npz=src_path,
        target_npz=tgt_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
        sigma_mmd=args.sigma_mmd,
        mmd_kernel_num=args.mmd_kernel_num,
        mmd_kernel_mul=args.mmd_kernel_mul,
        early_stop_patience=early_stop,
        rul_ceiling=args.rul_ceiling,
        norm_type=args.norm_type,
        severson_only=args.severson_only,
        mmd_weight=args.mmd_weight,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
        num_runs=args.num_runs,
        device=device
    )

    if args.results_json:
        # Strip any numpy arrays before saving
        clean_res = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                continue
            clean_res[k] = v
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)), exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(clean_res, f, indent=2)
        logger.info(f"Saved run results to {args.results_json}")


if __name__ == "__main__":
    main()
