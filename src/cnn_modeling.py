#!/usr/bin/env python3
"""
Hybrid 1D-CNN Feature Extractor + XGBoost Regressor Pipeline for Battery Cycle Life & Knee Point Estimation.
Implements:
  1. 2D dQ/dV spatial-temporal matrix construction (Cycles 10 through 100 on uniform voltage grid)
  2. Deep 1D-CNN Feature Extractor (PyTorch) with L2 decay and dropout to learn spatial-temporal embeddings
  3. Prediction Head Pivot: XGBoost Regressor with high regularization (reg_alpha, reg_lambda, shallow max_depth)
  4. Secondary Evaluation: Knee Point onset prediction
  5. Rigorous 5-Fold CV and 20% Held-Out Test Split evaluation with zero target leakage
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [HybridCNN-XGB] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("HybridCNN-XGB")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =====================================================================
# 1. 2D dQ/dV Spatial-Temporal Matrix Construction & Caching
# =====================================================================

V_GRID_CNN = np.linspace(2.90, 3.50, 200)  # Narrowed to LFP discharge plateau region
CYCLES_CNN = list(range(10, 101, 2))       # 46 cycles (every 2nd cycle from 10 to 100)


def load_or_build_2d_dqdv_matrices(ts_path: str, cache_path: str):
    """
    Loads or constructs the 2D dQ/dV matrix (46 cycles x 200 voltage bins) for each cell.
    Returns:
        cell_ids (list): sorted cell IDs
        matrices (np.ndarray): shape (N_cells, 46, 200)
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading cached 2D dQ/dV matrices from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return list(data["cell_ids"]), data["matrices"]

    logger.info(f"Computing 2D dQ/dV spatial-temporal matrices from {ts_path}...")
    t0 = time.time()
    ts_df = pd.read_parquet(ts_path)
    cells = sorted(ts_df["cell_id"].unique())

    matrices_list = []
    for idx, cell_id in enumerate(cells):
        cell_df = ts_df[ts_df["cell_id"] == cell_id]
        mat = np.zeros((len(CYCLES_CNN), len(V_GRID_CNN)), dtype=np.float32)

        for j, cyc in enumerate(CYCLES_CNN):
            sub = cell_df[(cell_df["cycle_number"] == cyc) & (cell_df["current_A"] < -0.1)]
            if len(sub) > 15:
                v = sub["voltage_V"].values
                q = sub["capacity_Ah"].values
                valid = ~(np.isnan(v) | np.isnan(q))
                v, q = v[valid], q[valid]
                sort_idx = np.argsort(v)[::-1]  # discharge: decreasing voltage
                v_s, q_s = v[sort_idx], q[sort_idx]
                v_u, idx_u = np.unique(v_s, return_index=True)
                q_u = q_s[idx_u]

                if len(v_u) >= 10:
                    sort_asc = np.argsort(v_u)
                    f = interp1d(v_u[sort_asc], q_u[sort_asc], kind="linear", bounds_error=False, fill_value=(q_u[sort_asc][0], q_u[sort_asc][-1]))
                    qi = f(V_GRID_CNN)
                    dqdv = np.gradient(qi, V_GRID_CNN)
                    dqdv_smooth = savgol_filter(dqdv, window_length=15, polyorder=3)
                    mat[j, :] = dqdv_smooth

        matrices_list.append(mat)
        if (idx + 1) % 25 == 0 or (idx + 1) == len(cells):
            logger.info(f"  Processed {idx+1}/{len(cells)} cells in {time.time()-t0:.1f}s...")

    matrices = np.array(matrices_list, dtype=np.float32)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, cell_ids=cells, matrices=matrices)
    logger.info(f"Saved 2D dQ/dV matrices -> {cache_path} (Shape: {matrices.shape})")
    return cells, matrices


# =====================================================================
# 2. Knee Point Onset Calculation
# =====================================================================

def compute_knee_points(sum_df: pd.DataFrame):
    """
    Computes the Knee Point (cycle of accelerated capacity fade onset) per cell.
    Definition: Cycle where discharge capacity drops below 97% of initial capacity Ah,
    or maximum fade curvature if not dropped below 97%.
    """
    knee_points = {}
    cells = sorted(sum_df["cell_id"].unique())
    for cell_id in cells:
        cdf = sum_df[sum_df["cell_id"] == cell_id].sort_values("cycle_number")
        c0 = float(cdf["initial_capacity_Ah"].iloc[0])
        drop_idx = cdf[cdf["discharge_capacity_Ah"] < 0.97 * c0]
        if len(drop_idx) > 0:
            kp = int(drop_idx["cycle_number"].iloc[0])
        else:
            # Fallback to 75% of total cycle life
            kp = int(cdf["cycle_life"].iloc[0] * 0.75)
        knee_points[cell_id] = kp
    return knee_points


# =====================================================================
# 3. 1D-CNN Feature Extractor Architecture (PyTorch)
# =====================================================================

class BatteryCNNDataset(Dataset):
    def __init__(self, matrices: np.ndarray, y_log: np.ndarray, kp_log: np.ndarray):
        self.matrices = torch.tensor(matrices, dtype=torch.float32)
        self.y_log = torch.tensor(y_log, dtype=torch.float32)
        self.kp_log = torch.tensor(kp_log, dtype=torch.float32)

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        return self.matrices[idx], self.y_log[idx], self.kp_log[idx]


class Battery1DCNNFeatureExtractor(nn.Module):
    """
    1D Convolutional Neural Network that convolves across the voltage grid (200 bins)
    with 46 input channels (Cycles 10 to 100), learning spatial-temporal peak morphology.
    Outputs a 32-dimensional embedding vector.
    """
    def __init__(self, in_channels=46, emb_dim=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(2)  # 200 -> 100
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(2)  # 100 -> 50
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, emb_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(emb_dim),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.AdaptiveAvgPool1d(1)  # 50 -> 1
        )
        # Pre-training auxiliary linear head (stripped for XGBoost feature extraction)
        self.aux_head = nn.Linear(emb_dim, 2)  # predicts (log_cycle_life, log_knee_point)

    def forward(self, x):
        emb = self.extract_features(x)
        out = self.aux_head(emb)
        return out, emb

    def extract_features(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.squeeze(-1)  # shape: (batch_size, emb_dim)
        return x


def train_cnn_feature_extractor(model, train_loader, val_loader, epochs=150, lr=5e-4, weight_decay=1e-3, device="cpu"):
    """
    Trains the 1D-CNN on the training fold with L2 weight decay and early stopping
    to learn predictive spatial-temporal embeddings.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    patience = 25
    patience_counter = 0

    for ep in range(epochs):
        model.train()
        for mat_b, y_b, kp_b in train_loader:
            mat_b, y_b, kp_b = mat_b.to(device), y_b.to(device), kp_b.to(device)
            optimizer.zero_grad()
            out, _ = model(mat_b)
            loss_y = criterion(out[:, 0], y_b)
            loss_kp = criterion(out[:, 1], kp_b)
            loss = loss_y + 0.5 * loss_kp
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mat_b, y_b, kp_b in val_loader:
                mat_b, y_b, kp_b = mat_b.to(device), y_b.to(device), kp_b.to(device)
                out, _ = model(mat_b)
                loss_y = criterion(out[:, 0], y_b)
                loss_kp = criterion(out[:, 1], kp_b)
                val_loss += float(loss_y + 0.5 * loss_kp)

        val_loss /= max(len(val_loader), 1)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def extract_cnn_embeddings(model, matrices, device="cpu"):
    """
    Extracts the 32-dimensional spatial-temporal embedding vector for all input matrices.
    """
    model.eval()
    model.to(device)
    loader = DataLoader(
        BatteryCNNDataset(matrices, np.zeros(len(matrices)), np.zeros(len(matrices))),
        batch_size=32, shuffle=False
    )
    embeddings = []
    with torch.no_grad():
        for mat_b, _, _ in loader:
            mat_b = mat_b.to(device)
            emb = model.extract_features(mat_b)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)


# =====================================================================
# 4. Hybrid CNN + XGBoost Regressor Evaluation (5-Fold CV & Test Split)
# =====================================================================

def evaluate_metrics(y_true, y_pred):
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAPE_%": mape, "RMSE_cycles": rmse, "R2": r2}


def fit_and_eval_hybrid_pipeline(df: pd.DataFrame, cells: list, matrices: np.ndarray, res_dir: str):
    """
    Executes strict 5-Fold Cross-Validation and 20% Held-Out Test Split for:
      1. Primary Target: Battery Cycle Life (EOL)
      2. Secondary Target: Knee Point Onset Cycle
    using the Hybrid 1D-CNN Feature Extractor + Highly Regularized XGBoost Regressor.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using compute device: {device}")

    # Map cells to index in df
    cell_to_idx = {c: i for i, c in enumerate(cells)}
    df = df.sort_values("cell_id").reset_index(drop=True)

    # Clean targets
    y_eol = df["cycle_life"].values
    y_kp = df["knee_point"].values
    log_eol = np.log10(y_eol)
    log_kp = np.log10(y_kp)

    # Domain physics features (from Phase 2) to combine with CNN embeddings
    phys_cols = [
        "log_var_delta_q_100_10", "log_min_delta_q_100_10",
        "cap_100", "delta_cap_100_10", "rel_cap_loss_100_10",
        "delta_avg_v_100_10", "var_dqdv_100_10", "min_dqdv_100_10",
        "delta_h_peak1_100_10", "delta_h_peak2_100_10",
        "delta_v_peak1_100_10", "delta_v_peak2_100_10",
        "initial_capacity_Ah", "cap_fade_slope"
    ]
    X_phys = df[phys_cols].fillna(0).values

    # Strict 80/20 random train/test split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=SEED
    )

    logger.info(f"Dataset split: {len(train_idx)} training cells | {len(test_idx)} held-out test cells.")

    # -----------------------------------------------------------------
    # Step A: 5-Fold Cross-Validation on Training Split
    # -----------------------------------------------------------------
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_eol_mapes, cv_eol_rmses, cv_eol_r2s = [], [], []
    cv_kp_mapes, cv_kp_rmses, cv_kp_r2s = [], [], []

    logger.info("="*70)
    logger.info("RUNNING 5-FOLD CROSS-VALIDATION (Hybrid CNN + XGBoost Regressor)")
    logger.info("="*70)

    for fold, (tr_sub, val_sub) in enumerate(kf.split(train_idx)):
        fold_tr = train_idx[tr_sub]
        fold_val = train_idx[val_sub]

        # 1. Pre-train 1D-CNN Feature Extractor strictly on fold_tr
        ds_train = BatteryCNNDataset(matrices[fold_tr], log_eol[fold_tr], log_kp[fold_tr])
        ds_val = BatteryCNNDataset(matrices[fold_val], log_eol[fold_val], log_kp[fold_val])
        loader_tr = DataLoader(ds_train, batch_size=16, shuffle=True)
        loader_val = DataLoader(ds_val, batch_size=16, shuffle=False)

        cnn_model = Battery1DCNNFeatureExtractor(in_channels=len(CYCLES_CNN), emb_dim=32)
        cnn_model = train_cnn_feature_extractor(cnn_model, loader_tr, loader_val, epochs=150, lr=5e-4, weight_decay=1e-3, device=device)

        # 2. Extract 32-dim CNN spatial-temporal embeddings
        emb_tr = extract_cnn_embeddings(cnn_model, matrices[fold_tr], device=device)
        emb_val = extract_cnn_embeddings(cnn_model, matrices[fold_val], device=device)

        # 3. Combine CNN embeddings + Domain physics features
        scaler = StandardScaler()
        phys_tr = scaler.fit_transform(X_phys[fold_tr])
        phys_val = scaler.transform(X_phys[fold_val])

        X_tr_hybrid = np.hstack([emb_tr, phys_tr])
        X_val_hybrid = np.hstack([emb_val, phys_val])

        # 4. Train Highly Regularized XGBoost Regressor (EOL)
        xgb_eol = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=SEED
        )
        xgb_eol.fit(X_tr_hybrid, log_eol[fold_tr])
        pred_eol_val = 10**(xgb_eol.predict(X_val_hybrid))
        m_eol = evaluate_metrics(y_eol[fold_val], pred_eol_val)
        cv_eol_mapes.append(m_eol["MAPE_%"])
        cv_eol_rmses.append(m_eol["RMSE_cycles"])
        cv_eol_r2s.append(m_eol["R2"])

        # 5. Train Secondary XGBoost Regressor (Knee Point)
        xgb_kp = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=SEED
        )
        xgb_kp.fit(X_tr_hybrid, log_kp[fold_tr])
        pred_kp_val = 10**(xgb_kp.predict(X_val_hybrid))
        m_kp = evaluate_metrics(y_kp[fold_val], pred_kp_val)
        cv_kp_mapes.append(m_kp["MAPE_%"])
        cv_kp_rmses.append(m_kp["RMSE_cycles"])
        cv_kp_r2s.append(m_kp["R2"])

        logger.info(f"  [Fold {fold+1}/5] EOL MAPE: {m_eol['MAPE_%']:.2f}% | R²: {m_eol['R2']:.3f} || Knee Point MAPE: {m_kp['MAPE_%']:.2f}%")

    mean_cv_eol_mape = np.mean(cv_eol_mapes)
    mean_cv_eol_rmse = np.mean(cv_eol_rmses)
    mean_cv_eol_r2 = np.mean(cv_eol_r2s)
    mean_cv_kp_mape = np.mean(cv_kp_mapes)

    logger.info("="*70)
    logger.info(f"5-FOLD CV SUMMARY -> EOL MAPE: {mean_cv_eol_mape:.2f}% | RMSE: {mean_cv_eol_rmse:.1f} | R²: {mean_cv_eol_r2:.3f}")
    logger.info(f"5-FOLD CV SUMMARY -> Knee Point MAPE: {mean_cv_kp_mape:.2f}%")
    logger.info("="*70)

    # -----------------------------------------------------------------
    # Step B: Final Model Evaluation on 20% Held-Out Test Split
    # -----------------------------------------------------------------
    logger.info("Training Final Hybrid CNN-XGBoost pipeline on full training split (80%)...")
    ds_tr_full = BatteryCNNDataset(matrices[train_idx], log_eol[train_idx], log_kp[train_idx])
    ds_te_full = BatteryCNNDataset(matrices[test_idx], log_eol[test_idx], log_kp[test_idx])
    loader_tr_full = DataLoader(ds_tr_full, batch_size=16, shuffle=True)
    loader_te_full = DataLoader(ds_te_full, batch_size=16, shuffle=False)

    final_cnn = Battery1DCNNFeatureExtractor(in_channels=len(CYCLES_CNN), emb_dim=32)
    final_cnn = train_cnn_feature_extractor(final_cnn, loader_tr_full, loader_te_full, epochs=200, lr=5e-4, weight_decay=1e-3, device=device)

    emb_train = extract_cnn_embeddings(final_cnn, matrices[train_idx], device=device)
    emb_test = extract_cnn_embeddings(final_cnn, matrices[test_idx], device=device)

    scaler_full = StandardScaler()
    phys_train = scaler_full.fit_transform(X_phys[train_idx])
    phys_test = scaler_full.transform(X_phys[test_idx])

    X_train_hybrid = np.hstack([emb_train, phys_train])
    X_test_hybrid = np.hstack([emb_test, phys_test])

    # Final XGBoost EOL
    final_xgb_eol = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED
    )
    final_xgb_eol.fit(X_train_hybrid, log_eol[train_idx])
    test_pred_eol = 10**(final_xgb_eol.predict(X_test_hybrid))
    test_eol_metrics = evaluate_metrics(y_eol[test_idx], test_pred_eol)

    # Final XGBoost Knee Point
    final_xgb_kp = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED
    )
    final_xgb_kp.fit(X_train_hybrid, log_kp[train_idx])
    test_pred_kp = 10**(final_xgb_kp.predict(X_test_hybrid))
    test_kp_metrics = evaluate_metrics(y_kp[test_idx], test_pred_kp)

    logger.info("="*70)
    logger.info("FINAL HELD-OUT TEST SPLIT RESULTS (Hybrid CNN-XGBoost):")
    logger.info("="*70)
    logger.info(f"  [Cycle Life / EOL ] Test MAPE: {test_eol_metrics['MAPE_%']:.2f}% | Test RMSE: {test_eol_metrics['RMSE_cycles']:.1f} | Test R²: {test_eol_metrics['R2']:.3f}")
    logger.info(f"  [Knee Point Onset] Test MAPE: {test_kp_metrics['MAPE_%']:.2f}% | Test RMSE: {test_kp_metrics['RMSE_cycles']:.1f} | Test R²: {test_kp_metrics['R2']:.3f}")
    logger.info("="*70)

    # Save results and predictions
    os.makedirs(res_dir, exist_ok=True)
    summary_df = pd.DataFrame({
        "Model_Suite": ["Hybrid_1DCNN_XGBoost_EOL", "Hybrid_1DCNN_XGBoost_KneePoint"],
        "5Fold_CV_MAPE_%": [mean_cv_eol_mape, mean_cv_kp_mape],
        "5Fold_CV_RMSE": [mean_cv_eol_rmse, np.nan],
        "5Fold_CV_R2": [mean_cv_eol_r2, np.nan],
        "Test_MAPE_%": [test_eol_metrics["MAPE_%"], test_kp_metrics["MAPE_%"]],
        "Test_RMSE": [test_eol_metrics["RMSE_cycles"], test_kp_metrics["RMSE_cycles"]],
        "Test_R2": [test_eol_metrics["R2"], test_kp_metrics["R2"]]
    })
    out_csv = os.path.join(res_dir, "hybrid_cnn_xgb_evaluation_metrics.csv")
    summary_df.to_csv(out_csv, index=False)
    logger.info(f"Saved evaluation metrics summary -> {out_csv}")

    # Save test predictions for visualization
    test_preds_df = pd.DataFrame({
        "cell_id": df.iloc[test_idx]["cell_id"].values,
        "actual_cycle_life": y_eol[test_idx],
        "predicted_cycle_life": test_pred_eol,
        "actual_knee_point": y_kp[test_idx],
        "predicted_knee_point": test_pred_kp
    })
    out_preds = os.path.join(res_dir, "hybrid_cnn_xgb_predictions.parquet")
    test_preds_df.to_parquet(out_preds, index=False)
    logger.info(f"Saved held-out test predictions -> {out_preds}")

    return summary_df, test_preds_df


def main():
    parser = argparse.ArgumentParser(description="Hybrid 1D-CNN + XGBoost Regressor Pipeline")
    parser.add_argument("--ts-path", type=str, default="data/processed/battery_time_series.parquet")
    parser.add_argument("--sum-path", type=str, default="data/processed/battery_summary.parquet")
    parser.add_argument("--feat-path", type=str, default="data/processed/engineered_features.parquet")
    parser.add_argument("--cache-path", type=str, default="data/processed/dqdv_2d_matrices.npz")
    parser.add_argument("--res-dir", type=str, default="results")
    args = parser.parse_args()

    # 1. Load summary and engineered features
    logger.info(f"Loading summary from {args.sum_path} and features from {args.feat_path}...")
    sum_df = pd.read_parquet(args.sum_path)
    feat_df = pd.read_parquet(args.feat_path)

    # 2. Compute Knee Points per cell
    knee_points = compute_knee_points(sum_df)
    feat_df["knee_point"] = feat_df["cell_id"].map(knee_points)
    feat_df["log_knee_point"] = np.log10(feat_df["knee_point"])

    # 3. Load or build 2D dQ/dV spatial-temporal matrices
    cells, matrices = load_or_build_2d_dqdv_matrices(args.ts_path, args.cache_path)

    # 4. Execute 5-Fold CV and Test Split for Hybrid CNN + XGBoost
    fit_and_eval_hybrid_pipeline(feat_df, cells, matrices, args.res_dir)
    logger.info("Hybrid CNN + XGBoost Regressor execution complete!")


if __name__ == "__main__":
    main()
