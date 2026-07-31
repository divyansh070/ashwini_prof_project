import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import optuna
import warnings

warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.koopman.train_da_colab import (
    BatterySOCDataset,
    BatteryKoopmanDANN,
    train_source_loop,
    train_dann_loop,
    evaluate_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("KoopmanOptuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_dataloaders(data_dir: str, batch_size: int = 16):
    """
    Load Stanford LFP (Source) and Oxford LCO (Target) datasets and prepare DataLoaders.
    """
    lfp_path = os.path.join(data_dir, "stanford_lfp_soc.npz")
    lco_path = os.path.join(data_dir, "oxford_lco_soc.npz")

    if not os.path.exists(lfp_path) or not os.path.exists(lco_path):
        logger.error(f"Missing processed SOC datasets in {data_dir}. Run preprocess_v2.py first.")
        sys.exit(1)

    # 1. Source (Stanford LFP)
    lfp_data = np.load(lfp_path)
    X_lfp, y_lfp = lfp_data["matrices_soc"], lfp_data["y_eol"]
    mean_lfp = np.mean(X_lfp, axis=0, keepdims=True)
    std_lfp = np.std(X_lfp, axis=0, keepdims=True) + 1e-8
    X_lfp_norm = (X_lfp - mean_lfp) / std_lfp

    ds_src_all = BatterySOCDataset(X_lfp_norm, y_lfp, domain_label=0)
    loader_src_all = DataLoader(ds_src_all, batch_size=batch_size, shuffle=True)

    # 2. Target (Oxford LCO - 8 cells)
    lco_data = np.load(lco_path)
    X_lco, y_lco = lco_data["matrices_soc"], lco_data["y_eol"]

    # Strict fold-scoped normalization for Oxford (first 6 train, last 2 test/val)
    tr_idx = np.array([0, 1, 2, 3, 4, 5])
    te_idx = np.array([6, 7])

    mean_lco = np.mean(X_lco[tr_idx], axis=0, keepdims=True)
    std_lco = np.std(X_lco[tr_idx], axis=0, keepdims=True) + 1e-8
    X_lco_tr_norm = (X_lco[tr_idx] - mean_lco) / std_lco
    X_lco_te_norm = (X_lco[te_idx] - mean_lco) / std_lco

    ds_lco_tr = BatterySOCDataset(X_lco_tr_norm, y_lco[tr_idx], domain_label=1)
    ds_lco_te = BatterySOCDataset(X_lco_te_norm, y_lco[te_idx], domain_label=1)
    loader_lco_tr = DataLoader(ds_lco_tr, batch_size=4, shuffle=True)
    loader_lco_te = DataLoader(ds_lco_te, batch_size=4, shuffle=False)

    return loader_src_all, loader_lco_tr, loader_lco_te


def get_source_model(
    loader_src_all: DataLoader,
    device: torch.device,
    epochs_source: int = 50,
    lr_source: float = 1e-3,
    lambda_koopman: float = 0.1,
    lambda_mono: float = 0.05,
    ckpt_path: str = "checkpoints/koopman_dann_stanford_source.pth",
):
    """
    Load pre-trained source model checkpoint if available, otherwise train source model once.
    """
    model = BatteryKoopmanDANN(in_features=200, num_cycles=46, d_model=64).to(device)

    if os.path.exists(ckpt_path):
        logger.info(f"Loading pre-trained source model from {ckpt_path}...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return model

    logger.info("No pre-trained source checkpoint found. Training source model on Stanford LFP...")
    os.makedirs("checkpoints", exist_ok=True)
    model = train_source_loop(
        model,
        loader_src_all,
        loader_src_all,
        epochs=epochs_source,
        lr=lr_source,
        lambda_koopman=lambda_koopman,
        lambda_mono=lambda_mono,
        device=device,
        domain_name="Stanford_LFP_Source",
    )
    torch.save(model.state_dict(), ckpt_path)
    logger.info(f"Saved source model checkpoint -> {ckpt_path}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization for Koopman DANN on Oxford LCO")
    parser.add_argument("--data-dir", type=str, default="data/koopman_processed", help="Directory with SOC .npz files")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--epochs-dann", type=int, default=40, help="DANN adaptation epochs per trial")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for source dataloader")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"======================================================================")
    logger.info(f"OPTUNA HYPERPARAMETER SWEEP FOR KOOPMAN DANN (Oxford LCO Target)")
    logger.info(f"Device: {device} | Total Trials: {args.n_trials} | DANN Epochs per Trial: {args.epochs_dann}")
    logger.info(f"======================================================================")

    loader_src_all, loader_lco_tr, loader_lco_te = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    source_model = get_source_model(loader_src_all, device=device)
    source_state_dict = {k: v.cpu().clone() for k, v in source_model.state_dict().items()}

    def objective(trial: optuna.Trial) -> float:
        # Bayesian Search Hyperparameters
        lambda_dann = trial.suggest_float("lambda_dann", 0.1, 1.0)
        lambda_koopman = trial.suggest_float("lambda_koopman", 0.01, 0.2)
        lambda_mono = trial.suggest_float("lambda_mono", 0.01, 0.1)
        lr_dann = trial.suggest_float("lr_dann", 1e-5, 1e-3, log=True)

        logger.info(
            f"[Trial {trial.number:02d}/{args.n_trials:02d}] "
            f"lambda_dann={lambda_dann:.4f}, lambda_koopman={lambda_koopman:.4f}, "
            f"lambda_mono={lambda_mono:.4f}, lr_dann={lr_dann:.2e}"
        )

        model = BatteryKoopmanDANN(in_features=200, num_cycles=46, d_model=64).to(device)
        model.load_state_dict(source_state_dict)

        model = train_dann_loop(
            model=model,
            source_loader=loader_src_all,
            target_loader=loader_lco_tr,
            target_val_loader=loader_lco_te,
            epochs=args.epochs_dann,
            lr=lr_dann,
            lambda_koopman=lambda_koopman,
            lambda_mono=lambda_mono,
            lambda_dann=lambda_dann,
            device=device,
            target_name=f"Trial_{trial.number:02d}",
            patience=10,
        )

        metrics = evaluate_model(model, loader_lco_te, device=device)
        linear_mape = metrics["MAPE_%"]
        logger.info(f"  --> Trial {trial.number:02d} Linear-Space Test MAPE: {linear_mape:.2f}% (R²: {metrics['R2']:.3f})")
        return linear_mape

    study = optuna.create_study(direction="minimize", study_name="koopman_dann_oxford_lco")
    study.optimize(objective, n_trials=args.n_trials)

    best_trial = study.best_trial
    logger.info("\n======================================================================")
    logger.info("OPTUNA BAYESIAN HYPERPARAMETER OPTIMIZATION SUMMARY")
    logger.info("======================================================================")
    logger.info(f"Best Trial Number         : {best_trial.number}")
    logger.info(f"Best Linear-Space MAPE (%) : {best_trial.value:.2f}%")
    logger.info("Optimized Hyperparameters:")
    for param_key, param_val in best_trial.params.items():
        logger.info(f"  - {param_key:<16}: {param_val:.5f}")
    logger.info("======================================================================")

    os.makedirs("results", exist_ok=True)
    best_params_path = "results/optuna_best_params_oxford.json"
    with open(best_params_path, "w") as f:
        json.dump(
            {
                "best_trial_number": best_trial.number,
                "best_linear_space_mape_pct": best_trial.value,
                "best_params": best_trial.params,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved best hyperparameters -> {best_params_path}")

    df_trials = study.trials_dataframe()
    trials_csv_path = "results/optuna_study_oxford_lco.csv"
    df_trials.to_csv(trials_csv_path, index=False)
    logger.info(f"Saved full Optuna study history -> {trials_csv_path}")


if __name__ == "__main__":
    main()
