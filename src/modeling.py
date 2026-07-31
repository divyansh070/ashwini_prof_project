#!/usr/bin/env python3
"""
Modeling & Hypothesis Testing Module for Lithium-Ion Battery SOH & RUL Estimation.
Implements:
  1. Baseline Naive Model (Raw features only: early capacity slope, mean temp, charge rate)
  2. Benchmark Severson Model (Log10 Delta Variance of dQ/dV feature with ElasticNet)
  3. Full Physics-Informed Model (All 29 dQ/dV peak, variance, and IR features) using ElasticNet, RF, LightGBM.
Enforces N=124 overfitting safeguards (max_depth<=3, min_samples_leaf>=5, num_leaves<=8, ElasticNetCV)
and evaluates Test MAPE (%), RMSE (cycles), and R² across 5-Fold CV and a 20% held-out test split.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import lightgbm as lgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Modeling")


# Feature sets definition
BASELINE_FEATURES = [
    "cap_100", "delta_cap_100_10", "rel_cap_loss_100_10",
    "cap_fade_slope", "delta_temp_100_10", "delta_avg_v_100_10", "initial_capacity_Ah"
]

SEVERSON_FEATURE = ["log_var_delta_q_100_10"]

# All physics-informed features from Feature Engineering (dQ/dV + Delta Q(V) Severson features)
PHYSICS_FEATURES = [
    "var_dqdv_100_10", "min_dqdv_100_10", "max_dqdv_100_10", "mean_dqdv_100_10",
    "skew_dqdv_100_10", "kurt_dqdv_100_10", "l1_dqdv_100_10", "l2_dqdv_100_10",
    "var_delta_q_100_10", "min_delta_q_100_10", "log_var_delta_q_100_10", "log_min_delta_q_100_10",
    "v_peak1_10", "v_peak1_100", "h_peak1_10", "h_peak1_100",
    "delta_v_peak1_100_10", "delta_h_peak1_100_10",
    "v_peak2_10", "v_peak2_100", "h_peak2_10", "h_peak2_100",
    "delta_v_peak2_100_10", "delta_h_peak2_100_10",
    "cap_10", "cap_100", "delta_cap_100_10", "rel_cap_loss_100_10",
    "cap_fade_slope", "delta_temp_100_10", "delta_avg_v_100_10", "initial_capacity_Ah",
    "log10_var_dqdv"
]


def get_model(algo_name: str, n_features: int):
    """
    Returns an initialized model adhering strictly to N=124 overfitting safeguards.
    """
    if algo_name == "ElasticNet":
        return ElasticNetCV(
            alphas=np.logspace(-4, 2, 50),
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.99, 1.0],
            cv=5,
            max_iter=5000,
            random_state=42
        )
    elif algo_name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=5,
            max_features="sqrt" if n_features > 5 else 1.0,
            random_state=42
        )
    elif algo_name == "LightGBM":
        return lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=3,
            num_leaves=8,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8 if n_features > 5 else 1.0,
            random_state=42,
            verbose=-1
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")


def evaluate_metrics(y_true_cycles, y_pred_cycles):
    """
    Computes MAPE (%), RMSE (cycles), and R² in linear cycle space.
    """
    mape = float(mean_absolute_percentage_error(y_true_cycles, y_pred_cycles) * 100.0)
    rmse = float(np.sqrt(mean_squared_error(y_true_cycles, y_pred_cycles)))
    r2 = float(r2_score(y_true_cycles, y_pred_cycles))
    return {"MAPE_%": mape, "RMSE_cycles": rmse, "R2": r2}


def run_pipeline_for_model_suite(df: pd.DataFrame):
    """
    Executes 5-Fold CV and 20% held-out test evaluation for the Model Suite:
      - Baseline Naive (ElasticNet, LightGBM)
      - Benchmark Severson (ElasticNet)
      - Full Physics-Informed (ElasticNet, RandomForest, LightGBM)
    """
    # Create log10 variance feature for Severson & Physics models
    df["log10_var_dqdv"] = np.log10(np.abs(df["var_dqdv_100_10"]) + 1e-12)

    # Clean target
    y = df["cycle_life"].values
    log_y = np.log10(y)

    # 80/20 train/test split (stratified or random seeded)
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=42
    )

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    y_train = y[train_idx]
    y_test = y[test_idx]
    log_y_train = log_y[train_idx]
    log_y_test = log_y[test_idx]

    logger.info(f"Dataset split: {len(df_train)} training cells | {len(df_test)} held-out test cells.")

    suite_configs = [
        ("1_Baseline_Naive", "ElasticNet", BASELINE_FEATURES),
        ("1_Baseline_Naive", "LightGBM", BASELINE_FEATURES),
        ("2_Benchmark_Severson", "ElasticNet", SEVERSON_FEATURE),
        ("3_Full_Physics_Informed", "ElasticNet", PHYSICS_FEATURES),
        ("3_Full_Physics_Informed", "RandomForest", PHYSICS_FEATURES),
        ("3_Full_Physics_Informed", "LightGBM", PHYSICS_FEATURES)
    ]

    results_summary = []
    prediction_records = []
    feature_importance_records = []

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for suite_name, algo_name, feat_cols in suite_configs:
        logger.info(f"Evaluating -> Suite: {suite_name} | Algo: {algo_name} | Features: {len(feat_cols)}")

        # 1. 5-Fold Cross-Validation on Training Set (prevent target leakage by scaling per fold)
        oof_preds_log = np.zeros(len(df_train))

        for fold, (trn_idx, val_idx) in enumerate(kf.split(df_train)):
            X_trn = df_train.loc[trn_idx, feat_cols].values
            y_trn_log = log_y_train[trn_idx]
            X_val = df_train.loc[val_idx, feat_cols].values

            scaler = StandardScaler()
            X_trn_scaled = scaler.fit_transform(X_trn)
            X_val_scaled = scaler.transform(X_val)

            model = get_model(algo_name, len(feat_cols))
            model.fit(X_trn_scaled, y_trn_log)

            oof_preds_log[val_idx] = model.predict(X_val_scaled)

        oof_preds_cycles = 10 ** oof_preds_log
        cv_metrics = evaluate_metrics(y_train, oof_preds_cycles)

        # 2. Final Training on Full Train Set & Evaluation on Held-out Test Set
        scaler_full = StandardScaler()
        X_train_full = scaler_full.fit_transform(df_train[feat_cols].values)
        X_test_scaled = scaler_full.transform(df_test[feat_cols].values)

        final_model = get_model(algo_name, len(feat_cols))
        final_model.fit(X_train_full, log_y_train)

        test_preds_log = final_model.predict(X_test_scaled)
        test_preds_cycles = 10 ** test_preds_log

        test_metrics = evaluate_metrics(y_test, test_preds_cycles)

        logger.info(f"  [5-Fold CV ] MAPE: {cv_metrics['MAPE_%']:5.2f}% | RMSE: {cv_metrics['RMSE_cycles']:5.1f} | R²: {cv_metrics['R2']:.3f}")
        logger.info(f"  [Test Split] MAPE: {test_metrics['MAPE_%']:5.2f}% | RMSE: {test_metrics['RMSE_cycles']:5.1f} | R²: {test_metrics['R2']:.3f}")

        # Store evaluation results
        results_summary.append({
            "model_suite": suite_name,
            "algorithm": algo_name,
            "num_features": len(feat_cols),
            "cv_mape_%": cv_metrics["MAPE_%"],
            "cv_rmse_cycles": cv_metrics["RMSE_cycles"],
            "cv_r2": cv_metrics["R2"],
            "test_mape_%": test_metrics["MAPE_%"],
            "test_rmse_cycles": test_metrics["RMSE_cycles"],
            "test_r2": test_metrics["R2"]
        })

        # Record test predictions for parity plots
        for i, cid in enumerate(df_test["cell_id"].values):
            prediction_records.append({
                "model_suite": suite_name,
                "algorithm": algo_name,
                "cell_id": cid,
                "actual_cycle_life": float(y_test[i]),
                "predicted_cycle_life": float(test_preds_cycles[i])
            })

        # Extract feature importances or coefficients for Full Physics LightGBM / RF / ElasticNet
        if suite_name == "3_Full_Physics_Informed":
            if algo_name in ["LightGBM", "RandomForest"]:
                imp = final_model.feature_importances_
                imp_norm = imp / (np.sum(imp) + 1e-12)
            elif algo_name == "ElasticNet":
                imp = np.abs(final_model.coef_)
                imp_norm = imp / (np.sum(imp) + 1e-12)
            else:
                imp_norm = np.zeros(len(feat_cols))

            for feat_name, val in zip(feat_cols, imp_norm):
                feature_importance_records.append({
                    "model_suite": suite_name,
                    "algorithm": algo_name,
                    "feature": feat_name,
                    "importance": float(val)
                })

    return pd.DataFrame(results_summary), pd.DataFrame(prediction_records), pd.DataFrame(feature_importance_records)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Modeling & Hypothesis Testing Pipeline")
    parser.add_argument("--proc-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--feat-file", type=str, default="engineered_features.parquet", help="Input features file")
    parser.add_argument("--out-dir", type=str, default="results", help="Path to results output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    feat_path = os.path.join(args.proc_dir, args.feat_file)

    if not os.path.exists(feat_path):
        logger.error(f"Feature dataset not found at {feat_path}. Run Phase 2 first.")
        sys.exit(1)

    logger.info(f"Loading engineered feature dataset from {feat_path}...")
    df = pd.read_parquet(feat_path)

    logger.info("="*60)
    logger.info("PHASE 3: MODEL SUITE EXECUTION & HYPOTHESIS TESTING")
    logger.info("Enforcing N=124 Safeguards: max_depth<=3, min_samples_leaf>=5, num_leaves<=8")
    logger.info("="*60)

    res_df, pred_df, fi_df = run_pipeline_for_model_suite(df)

    res_path = os.path.join(args.out_dir, "model_evaluation_metrics.csv")
    res_json_path = os.path.join(args.out_dir, "model_evaluation_metrics.json")
    pred_path = os.path.join(args.out_dir, "model_predictions.parquet")
    fi_path = os.path.join(args.out_dir, "feature_importances.parquet")

    res_df.to_csv(res_path, index=False)
    res_df.to_json(res_json_path, orient="records", indent=2)
    pred_df.to_parquet(pred_path, index=False)
    fi_df.to_parquet(fi_path, index=False)

    logger.info("="*60)
    logger.info("MODEL SUITE COMPARISON SUMMARY (Held-out 20% Test Split):")
    logger.info("="*60)
    for idx, row in res_df.iterrows():
        logger.info(f"{row['model_suite']:<24} | {row['algorithm']:<12} | "
                    f"Test MAPE: {row['test_mape_%']:5.2f}% | "
                    f"Test RMSE: {row['test_rmse_cycles']:6.1f} | "
                    f"Test R²: {row['test_r2']:5.3f}")
    logger.info("="*60)
    logger.info(f"Saved evaluation metrics   : {res_path}")
    logger.info(f"Saved model predictions    : {pred_path}")
    logger.info(f"Saved feature importances  : {fi_path}")
    logger.info("Phase 3 modeling complete!")


if __name__ == "__main__":
    main()
