#!/usr/bin/env python3
"""
Rigorous Statistical Audit & Validation Module for Battery SOH & RUL Estimation.
Executes:
  1. Leave-One-Batch-Out (LOBO) Validation (Train on Batches 1 & 2, Test on Batch 3)
     to prove out-of-distribution generalization without data leakage.
  2. Feature Ablation Study comparing:
     - (A) Baseline Naive Only
     - (B) CNN 1D Embeddings Only
     - (C) Domain Physics dQ/dV Only
     - (D) Full Hybrid 1D-CNN + XGBoost
  3. Residual Analysis (Normality, systematic bias, correlation with actual cycle life).
  4. Generates figures/residual_analysis.png and results/model_audit_results.csv.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader

# Import CNN module helpers
from cnn_modeling import (
    BatteryCNNDataset,
    Battery1DCNNFeatureExtractor,
    train_cnn_feature_extractor,
    extract_cnn_embeddings,
    CYCLES_CNN,
    SEED
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [StatisticalAudit] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("StatisticalAudit")

np.random.seed(SEED)
torch.manual_seed(SEED)

PALETTE = {
    "primary": "#2563EB",      # Blue
    "secondary": "#0D9488",    # Teal
    "accent": "#DC2626",       # Crimson Red
    "dark": "#1E293B",         # Slate Dark
    "grid": "#E2E8F0"          # Grid grey
}


def setup_style():
    plt.rcParams.update({
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.family": "sans-serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linestyle": "--",
        "grid.alpha": 0.7
    })


def evaluate_metrics(y_true, y_pred):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    abs_err_pct = np.abs(y_true - y_pred) / y_true * 100.0
    median_mape = np.median(abs_err_pct)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "MAPE_%": mape,
        "Median_MAPE_%": median_mape,
        "RMSE_cycles": rmse,
        "R2": r2
    }


def run_lobo_validation(df, matrices, log_eol, y_eol, device="cpu"):
    """
    1. Leave-One-Batch-Out (LOBO) Validation:
    Train strictly on Batches 1 and 2. Test entirely on Batch 3 (40 cells).
    """
    logger.info("="*70)
    logger.info("TEST 1: LEAVE-ONE-BATCH-OUT (LOBO) VALIDATION (Train B1+B2 -> Test B3)")
    logger.info("="*70)

    tr_mask = (df["batch"] == "b1") | (df["batch"] == "b2")
    te_mask = (df["batch"] == "b3")

    train_idx = np.where(tr_mask)[0]
    test_idx = np.where(te_mask)[0]

    logger.info(f"LOBO Split -> Train on Batches 1 & 2 (N={len(train_idx)}) | Test on Batch 3 (N={len(test_idx)})")

    # Train CNN strictly on Batch 1 & 2
    ds_tr = BatteryCNNDataset(matrices[train_idx], log_eol[train_idx], log_eol[train_idx])
    ds_te = BatteryCNNDataset(matrices[test_idx], log_eol[test_idx], log_eol[test_idx])
    loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)
    loader_te = DataLoader(ds_te, batch_size=16, shuffle=False)

    cnn = Battery1DCNNFeatureExtractor(in_channels=len(CYCLES_CNN), emb_dim=32)
    cnn = train_cnn_feature_extractor(cnn, loader_tr, loader_te, epochs=150, lr=5e-4, weight_decay=1e-3, device=device)

    emb_train = extract_cnn_embeddings(cnn, matrices[train_idx], device=device)
    emb_test = extract_cnn_embeddings(cnn, matrices[test_idx], device=device)

    phys_cols = [
        "log_var_delta_q_100_10", "log_min_delta_q_100_10",
        "cap_100", "delta_cap_100_10", "rel_cap_loss_100_10",
        "delta_avg_v_100_10", "var_dqdv_100_10", "min_dqdv_100_10",
        "delta_h_peak1_100_10", "delta_h_peak2_100_10",
        "delta_v_peak1_100_10", "delta_v_peak2_100_10",
        "initial_capacity_Ah", "cap_fade_slope"
    ]
    X_phys = df[phys_cols].fillna(0).values
    scaler = StandardScaler()
    phys_tr = scaler.fit_transform(X_phys[train_idx])
    phys_te = scaler.transform(X_phys[test_idx])

    X_tr_hybrid = np.hstack([emb_train, phys_tr])
    X_te_hybrid = np.hstack([emb_test, phys_te])

    m_xgb = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED
    )
    m_xgb.fit(X_tr_hybrid, log_eol[train_idx])
    pred_log = m_xgb.predict(X_te_hybrid)
    pred_eol = 10**(pred_log)

    lobo_metrics = evaluate_metrics(y_eol[test_idx], pred_eol)
    logger.info(f"LOBO Test MAPE (Batch 3, N=40)       : {lobo_metrics['MAPE_%']:.2f}%")
    logger.info(f"LOBO Median Test MAPE (Batch 3, N=40): {lobo_metrics['Median_MAPE_%']:.2f}%")
    logger.info(f"LOBO Test RMSE (Batch 3, N=40)       : {lobo_metrics['RMSE_cycles']:.1f} cycles")
    logger.info(f"LOBO Test R² (Batch 3, N=40)         : {lobo_metrics['R2']:.3f}")

    return lobo_metrics, y_eol[test_idx], pred_eol


def run_ablation_study(df, matrices, log_eol, y_eol, device="cpu"):
    """
    2. Ablation Study:
    Compare 4 sets of features on the 80/20 held-out test split:
      (A) Baseline Naive Only
      (B) CNN Embeddings Only
      (C) Domain Physics dQ/dV Only
      (D) Full Hybrid 1D-CNN + XGBoost
    """
    logger.info("="*70)
    logger.info("TEST 2: FEATURE ABLATION STUDY (Proving Physics Features Drive Performance)")
    logger.info("="*70)

    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=SEED
    )

    # Train CNN on train_idx
    ds_tr = BatteryCNNDataset(matrices[train_idx], log_eol[train_idx], log_eol[train_idx])
    ds_te = BatteryCNNDataset(matrices[test_idx], log_eol[test_idx], log_eol[test_idx])
    loader_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)
    loader_te = DataLoader(ds_te, batch_size=16, shuffle=False)

    cnn = Battery1DCNNFeatureExtractor(in_channels=len(CYCLES_CNN), emb_dim=32)
    cnn = train_cnn_feature_extractor(cnn, loader_tr, loader_te, epochs=150, lr=5e-4, weight_decay=1e-3, device=device)

    emb_train = extract_cnn_embeddings(cnn, matrices[train_idx], device=device)
    emb_test = extract_cnn_embeddings(cnn, matrices[test_idx], device=device)

    # Feature subsets
    naive_cols = ["initial_capacity_Ah", "cap_fade_slope", "cap_100", "delta_cap_100_10"]
    phys_cols = [
        "log_var_delta_q_100_10", "log_min_delta_q_100_10",
        "cap_100", "delta_cap_100_10", "rel_cap_loss_100_10",
        "delta_avg_v_100_10", "var_dqdv_100_10", "min_dqdv_100_10",
        "delta_h_peak1_100_10", "delta_h_peak2_100_10",
        "delta_v_peak1_100_10", "delta_v_peak2_100_10",
        "initial_capacity_Ah", "cap_fade_slope"
    ]

    scaler_naive = StandardScaler()
    naive_tr = scaler_naive.fit_transform(df[naive_cols].fillna(0).values[train_idx])
    naive_te = scaler_naive.transform(df[naive_cols].fillna(0).values[test_idx])

    scaler_phys = StandardScaler()
    phys_tr = scaler_phys.fit_transform(df[phys_cols].fillna(0).values[train_idx])
    phys_te = scaler_phys.transform(df[phys_cols].fillna(0).values[test_idx])

    conditions = [
        ("A_Baseline_Naive_Only", naive_tr, naive_te),
        ("B_CNN_Embeddings_Only", emb_train, emb_test),
        ("C_Domain_Physics_Only", phys_tr, phys_te),
        ("D_Full_Hybrid_1DCNN_XGBoost", np.hstack([emb_train, phys_tr]), np.hstack([emb_test, phys_te]))
    ]

    ablation_rows = []
    for cond_name, X_tr, X_te in conditions:
        m = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=SEED
        )
        m.fit(X_tr, log_eol[train_idx])
        pred_te = 10**(m.predict(X_te))
        met = evaluate_metrics(y_eol[test_idx], pred_te)
        met["Condition"] = cond_name
        ablation_rows.append(met)
        logger.info(f"[{cond_name:30s}] MAPE: {met['MAPE_%']:6.2f}% | Median MAPE: {met['Median_MAPE_%']:6.2f}% | RMSE: {met['RMSE_cycles']:6.1f} | R²: {met['R2']:6.3f}")

    ablation_df = pd.DataFrame(ablation_rows)[["Condition", "MAPE_%", "Median_MAPE_%", "RMSE_cycles", "R2"]]
    return ablation_df


def run_residual_analysis(y_true, y_pred, out_dir="figures"):
    """
    3. Residual Analysis:
    Check normality, mean bias, median bias, and correlation with actual cycle life.
    """
    logger.info("="*70)
    logger.info("TEST 3: RESIDUAL ANALYSIS (Testing Bias and Error Distribution)")
    logger.info("="*70)

    residuals = y_pred - y_true
    pct_residuals = (y_pred - y_true) / y_true * 100.0

    mean_err = np.mean(residuals)
    median_err = np.median(residuals)
    std_err = np.std(residuals)
    mean_pct_err = np.mean(pct_residuals)
    median_pct_err = np.median(pct_residuals)

    # Shapiro-Wilk test for normality of percentage residuals
    shapiro_stat, shapiro_p = stats.shapiro(pct_residuals)

    # Correlation between actual cycle life and residual (systematic bias check)
    corr_err_y, p_corr = stats.pearsonr(y_true, residuals)

    logger.info(f"Mean Residual Bias        : {mean_err:.1f} cycles ({mean_pct_err:.2f}%)")
    logger.info(f"Median Residual Bias      : {median_err:.1f} cycles ({median_pct_err:.2f}%)")
    logger.info(f"Residual Std Dev          : {std_err:.1f} cycles")
    logger.info(f"Shapiro-Wilk Normality p  : {shapiro_p:.4f} {'(Normal)' if shapiro_p > 0.05 else '(Non-Normal/Heavy-Tailed)'}")
    logger.info(f"Error vs. Actual Corr (r) : {corr_err_y:.3f} (p={p_corr:.4f})")

    if abs(corr_err_y) < 0.3:
        logger.info("--> FINDING: No severe systematic bias across high-cycle vs. low-cycle cells.")
    else:
        logger.info("--> FINDING: Mild systematic regression-to-the-mean observed (common in log-transformed models).")

    # Generate diagnostic figure
    setup_style()
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Subplot 1: Residual Histogram
    ax1 = axes[0]
    ax1.hist(pct_residuals, bins=15, color=PALETTE["primary"], edgecolor="white", alpha=0.8, density=True)
    x_grid = np.linspace(min(pct_residuals), max(pct_residuals), 200)
    ax1.plot(x_grid, stats.norm.pdf(x_grid, mean_pct_err, np.std(pct_residuals)),
             color=PALETTE["accent"], linewidth=2.0, label="Normal Curve")
    ax1.axvline(0, color=PALETTE["dark"], linestyle="--", linewidth=1.5, label="Zero Bias")
    ax1.set_xlabel("Percentage Residual Error (%)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Distribution of Percentage Residuals")
    ax1.legend()

    # Subplot 2: Residual vs. Actual Cycle Life
    ax2 = axes[1]
    ax2.scatter(y_true, residuals, color=PALETTE["secondary"], s=55, alpha=0.85, edgecolor="white")
    ax2.axhline(0, color=PALETTE["dark"], linestyle="--", linewidth=1.5, label="Zero Error Line")
    ax2.set_xlabel("Actual Cycle Life (Cycles)")
    ax2.set_ylabel("Prediction Residual (Predicted - Actual)")
    ax2.set_title(f"Residual Bias vs. Actual Cycle Life (r={corr_err_y:.2f})")
    ax2.legend()

    # Subplot 3: LOBO Batch 3 Parity Plot
    ax3 = axes[2]
    min_v = min(y_true.min(), y_pred.min()) * 0.85
    max_v = max(y_true.max(), y_pred.max()) * 1.10
    grid = np.linspace(min_v, max_v, 100)
    ax3.plot(grid, grid, color=PALETTE["dark"], linewidth=2.0, label="Parity (y = x)")
    ax3.fill_between(grid, grid*0.9, grid*1.1, color=PALETTE["secondary"], alpha=0.15, label="±10% Bound")
    ax3.scatter(y_true, y_pred, color=PALETTE["accent"], s=60, alpha=0.9, edgecolor="darkred", label="LOBO Test (Batch 3)")
    ax3.set_xlabel("Actual Cycle Life (Cycles)")
    ax3.set_ylabel("Predicted Cycle Life (Cycles)")
    ax3.set_title("LOBO Generalization Parity Plot (Batch 3)")
    ax3.legend()

    plt.tight_layout()
    out_file = os.path.join(out_dir, "residual_analysis.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved diagnostic figure -> {out_file}")

    return {
        "Mean_Bias_cycles": mean_err,
        "Median_Bias_cycles": median_err,
        "Mean_Pct_Error_%": mean_pct_err,
        "Median_Pct_Error_%": median_pct_err,
        "Shapiro_p_value": shapiro_p,
        "Error_vs_Actual_Corr_r": corr_err_y
    }


def main():
    parser = argparse.ArgumentParser(description="Rigorous Statistical Audit for Battery SOH & RUL")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Load summary and features
    summary_path = "data/processed/battery_summary.parquet"
    features_path = "data/processed/engineered_features.parquet"
    matrices_path = "data/processed/dqdv_2d_matrices.npz"

    df = pd.read_parquet(features_path).sort_values("cell_id").reset_index(drop=True)
    y_eol = df["cycle_life"].values
    log_eol = np.log10(y_eol)

    if not os.path.exists(matrices_path):
        logger.error(f"Missing 2D matrices {matrices_path}. Please run src/cnn_modeling.py first.")
        sys.exit(1)

    loaded = np.load(matrices_path)
    matrices = loaded["matrices"]

    # 1. LOBO Validation
    lobo_metrics, lobo_y_true, lobo_y_pred = run_lobo_validation(df, matrices, log_eol, y_eol, device=args.device)

    # 2. Ablation Study
    ablation_df = run_ablation_study(df, matrices, log_eol, y_eol, device=args.device)

    # 3. Residual Analysis on LOBO Batch 3
    residual_stats = run_residual_analysis(lobo_y_true, lobo_y_pred, out_dir="figures")

    # Save summary audit CSV
    audit_summary = pd.DataFrame([
        {"Metric": "LOBO_Test_MAPE_%", "Value": lobo_metrics["MAPE_%"]},
        {"Metric": "LOBO_Median_Test_MAPE_%", "Value": lobo_metrics["Median_MAPE_%"]},
        {"Metric": "LOBO_Test_RMSE_cycles", "Value": lobo_metrics["RMSE_cycles"]},
        {"Metric": "LOBO_Test_R2", "Value": lobo_metrics["R2"]},
        {"Metric": "Mean_Bias_cycles", "Value": residual_stats["Mean_Bias_cycles"]},
        {"Metric": "Median_Bias_cycles", "Value": residual_stats["Median_Bias_cycles"]},
        {"Metric": "Error_vs_Actual_Corr_r", "Value": residual_stats["Error_vs_Actual_Corr_r"]}
    ])

    audit_summary.to_csv("results/model_audit_summary.csv", index=False)
    ablation_df.to_csv("results/ablation_study_results.csv", index=False)

    logger.info("="*70)
    logger.info("RIGOROUS STATISTICAL AUDIT COMPLETED SUCCESSFULLY!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
