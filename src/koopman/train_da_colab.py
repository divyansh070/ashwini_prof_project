#!/usr/bin/env python3
"""
Domain Adversarial Transfer Learning (DANN) & Koopman Neural Operator Training Script (train_da_colab.py).
Executes:
  1. Source Domain Training: Trains Koopman Neural Operator model on Stanford/MIT (LFP) dataset.
  2. Domain Adversarial Adaptation (DANN): Actively aligns latent feature distributions of Stanford (Source)
     and Oxford/CALCE (Target) domains using a Gradient Reversal Layer (GRL) and Physics-Informed Koopman linearity.
  3. Evaluates Zero-Shot vs. Domain-Adversarial Transfer Learning performance and exports checkpoints
     and quantitative metrics to `results/domain_adversarial_metrics.csv`.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

from koopman_model import BatteryKoopmanDANN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Koopman-DANN-Colab] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Koopman-DANN-Colab")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class BatterySOCDataset(Dataset):
    """
    PyTorch Dataset wrapping 2D SOC-normalized dQ/d(SOC) matrices (num_cycles=46, L=200).
    """
    def __init__(self, matrices_soc: np.ndarray, y_eol: np.ndarray, domain_label: int = 0):
        self.matrices = torch.tensor(matrices_soc, dtype=torch.float32)
        self.log_eol = torch.tensor(np.log10(y_eol), dtype=torch.float32).unsqueeze(1)
        self.raw_eol = y_eol
        self.domain_label = torch.tensor(domain_label, dtype=torch.long)

    def __len__(self):
        return len(self.matrices)

    def __getitem__(self, idx):
        return self.matrices[idx], self.log_eol[idx], self.domain_label


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for x_batch, y_log_batch, _ in loader:
            x_batch = x_batch.to(device)
            pred_log, _, _ = model(x_batch, alpha=0.0)
            pred_log_np = pred_log.cpu().numpy().flatten()
            pred_eol = 10**(pred_log_np)
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


def train_source_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    lambda_koopman: float,
    device: torch.device,
    domain_name: str
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.05)
    criterion_mse = nn.MSELoss()

    best_loss = float("inf")
    best_weights = None

    logger.info(f"[{domain_name}] Training Koopman model for {epochs} epochs (lr={lr:.1e})...")
    start_t = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_kno = 0.0, 0.0
        for x_b, y_b, _ in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred_log, _, kno_loss = model(x_b, alpha=0.0)
            mse_loss = criterion_mse(pred_log, y_b)
            total_loss = mse_loss + lambda_koopman * kno_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += mse_loss.item() * len(x_b)
            tr_kno += kno_loss.item() * len(x_b)

        scheduler.step()
        tr_loss /= len(train_loader.dataset)
        tr_kno /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_v, y_v, _ in val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                pred_v, _, _ = model(x_v, alpha=0.0)
                val_loss += criterion_mse(pred_v, y_v).item() * len(x_v)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0 or ep == epochs:
            logger.info(f"  [Epoch {ep:03d}/{epochs:03d}] MSE: {tr_loss:.4f} | Val MSE: {val_loss:.4f} | KNO Loss: {tr_kno:.4f}")

    elap = time.time() - start_t
    logger.info(f"[{domain_name}] Training finished in {elap:.1f}s. Best Val MSE: {best_loss:.4f}")
    model.load_state_dict(best_weights)
    return model


def train_dann_loop(
    model: nn.Module,
    source_loader: DataLoader,
    target_loader: DataLoader,
    target_val_loader: DataLoader,
    epochs: int,
    lr: float,
    lambda_koopman: float,
    lambda_dann: float,
    device: torch.device,
    target_name: str
):
    """
    Explicit Domain-Adversarial Adaptation (DANN) loop.
    Simultaneously minimizes source RUL prediction error and Koopman linearity loss
    while maximizing Domain Discriminator confusion via Gradient Reversal Layer (GRL).
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.05)
    criterion_mse = nn.MSELoss()
    criterion_domain = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_weights = None

    logger.info(f"[{target_name}] Starting Domain-Adversarial Transfer Learning (DANN) for {epochs} epochs...")
    start_t = time.time()

    total_steps = epochs * max(len(source_loader), len(target_loader))
    current_step = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_mse, tr_dom, tr_kno = 0.0, 0.0, 0.0

        src_iter = iter(source_loader)
        tgt_iter = iter(target_loader)
        num_batches = max(len(source_loader), len(target_loader))

        for _ in range(num_batches):
            current_step += 1
            # Dynamic GRL adaptation coefficient alpha in [0, 1] (Ganin et al., 2016)
            p = float(current_step) / total_steps
            alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

            try:
                x_src, y_src, d_src = next(src_iter)
            except StopIteration:
                src_iter = iter(source_loader)
                x_src, y_src, d_src = next(src_iter)

            try:
                x_tgt, y_tgt, d_tgt = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(target_loader)
                x_tgt, y_tgt, d_tgt = next(tgt_iter)

            x_src, y_src, d_src = x_src.to(device), y_src.to(device), d_src.to(device)
            x_tgt, y_tgt, d_tgt = x_tgt.to(device), y_tgt.to(device), d_tgt.to(device)

            optimizer.zero_grad()

            # Forward source
            pred_src, dom_src, kno_src = model(x_src, alpha=alpha)
            mse_loss = criterion_mse(pred_src, y_src)
            dom_loss_src = criterion_domain(dom_src, d_src)

            # Forward target
            _, dom_tgt, kno_tgt = model(x_tgt, alpha=alpha)
            dom_loss_tgt = criterion_domain(dom_tgt, d_tgt)

            total_loss = (
                mse_loss
                + lambda_koopman * (kno_src + kno_tgt) * 0.5
                + lambda_dann * (dom_loss_src + dom_loss_tgt) * 0.5
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_mse += mse_loss.item()
            tr_dom += (dom_loss_src.item() + dom_loss_tgt.item()) * 0.5
            tr_kno += (kno_src.item() + kno_tgt.item()) * 0.5

        scheduler.step()
        tr_mse /= num_batches
        tr_dom /= num_batches
        tr_kno /= num_batches

        # Evaluate on target validation set
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for x_v, y_v, _ in target_val_loader:
                x_v, y_v = x_v.to(device), y_v.to(device)
                pred_v, _, _ = model(x_v, alpha=0.0)
                val_mse += criterion_mse(pred_v, y_v).item() * len(x_v)
        val_mse /= len(target_val_loader.dataset)

        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == epochs:
            logger.info(f"  [Epoch {ep:03d}/{epochs:03d}] alpha={alpha:.2f} | Src MSE: {tr_mse:.4f} | Dom Loss: {tr_dom:.4f} | Tgt Val MSE: {val_mse:.4f}")

    elap = time.time() - start_t
    logger.info(f"[{target_name}] DANN adaptation finished in {elap:.1f}s. Best Target Val MSE: {best_val_loss:.4f}")
    model.load_state_dict(best_weights)
    return model


def main():
    parser = argparse.ArgumentParser(description="Koopman Neural Operator & DANN Domain Adversarial Training")
    parser.add_argument("--data-dir", type=str, default="data/koopman_processed", help="Universal SOC processed directory")
    parser.add_argument("--epochs-source", type=int, default=100, help="Source training epochs")
    parser.add_argument("--epochs-dann", type=int, default=60, help="DANN adversarial adaptation epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr-source", type=float, default=5e-4, help="Source learning rate")
    parser.add_argument("--lr-dann", type=float, default=2e-4, help="DANN adaptation learning rate")
    parser.add_argument("--lambda-koopman", type=float, default=0.1, help="Koopman linearity penalty weight")
    parser.add_argument("--lambda-dann", type=float, default=0.5, help="Domain adversarial loss weight")
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("======================================================================")
    logger.info(f"KOOPMAN NEURAL OPERATOR & DANN ADVERSARIAL TRANSFER LEARNING ({device})")
    logger.info("======================================================================")

    # Load preprocessed SOC datasets
    lfp_path = os.path.join(args.data_dir, "stanford_lfp_soc.npz")
    lco_path = os.path.join(args.data_dir, "oxford_lco_soc.npz")
    nmc_path = os.path.join(args.data_dir, "calce_nmc_soc.npz")

    if not os.path.exists(lfp_path):
        logger.error(f"Missing source dataset {lfp_path}. Run preprocess_v2.py first.")
        sys.exit(1)

    lfp_data = np.load(lfp_path)
    X_lfp, y_lfp = lfp_data["matrices_soc"], lfp_data["y_eol"]

    # Source 80/20 train/test split (domain_label = 0)
    train_idx, test_idx = train_test_split(np.arange(len(X_lfp)), test_size=0.20, random_state=SEED)
    ds_src_tr = BatterySOCDataset(X_lfp[train_idx], y_lfp[train_idx], domain_label=0)
    ds_src_te = BatterySOCDataset(X_lfp[test_idx], y_lfp[test_idx], domain_label=0)
    loader_src_tr = DataLoader(ds_src_tr, batch_size=args.batch_size, shuffle=True)
    loader_src_te = DataLoader(ds_src_te, batch_size=args.batch_size, shuffle=False)

    # -------------------------------------------------------------------------
    # PHASE 1: SOURCE DOMAIN TRAINING (Stanford LFP Koopman Operator)
    # -------------------------------------------------------------------------
    logger.info("\n--- PHASE 1: SOURCE DOMAIN TRAINING (Stanford/MIT LFP) ---")
    source_model = BatteryKoopmanDANN(in_features=200, num_cycles=46, d_model=64)
    source_model = train_source_loop(
        source_model, loader_src_tr, loader_src_te,
        epochs=args.epochs_source, lr=args.lr_source,
        lambda_koopman=args.lambda_koopman, device=device, domain_name="Stanford_LFP_Koopman"
    )

    source_metrics = evaluate_model(source_model, loader_src_te, device=device)
    src_ckpt = "checkpoints/koopman_dann_stanford_source.pth"
    torch.save(source_model.state_dict(), src_ckpt)
    logger.info(f"[Stanford LFP Koopman Test] MAPE: {source_metrics['MAPE_%']:.2f}% | Median MAPE: {source_metrics['Median_MAPE_%']:.2f}% | R²: {source_metrics['R2']:.3f}")

    results_table = [{
        "Domain": "Stanford LFP (Source)",
        "Architecture": "Koopman Neural Operator (KNO)",
        "Condition": "Source Trained (From Scratch)",
        "MAPE_%": source_metrics["MAPE_%"],
        "Median_MAPE_%": source_metrics["Median_MAPE_%"],
        "RMSE_cycles": source_metrics["RMSE_cycles"],
        "R2": source_metrics["R2"]
    }]

    # -------------------------------------------------------------------------
    # PHASE 2 & 3: DOMAIN ADVERSARIAL TRANSFER LEARNING (Oxford LCO & CALCE NMC)
    # -------------------------------------------------------------------------
    targets = [
        ("Oxford LCO", lco_path, "oxford_lco"),
        ("CALCE NMC", nmc_path, "calce_nmc")
    ]

    for domain_label_str, path, tag in targets:
        if not os.path.exists(path):
            logger.warning(f"Skipping {domain_label_str}: file {path} not found.")
            continue

        logger.info(f"\n--- PHASE 2/3: DANN ADVERSARIAL ADAPTATION ON {domain_label_str} ---")
        tgt_data = np.load(path)
        X_tgt, y_tgt = tgt_data["matrices_soc"], tgt_data["y_eol"]

        # Target split: 60% train/adaptation (domain_label = 1), 40% test
        tr_tgt_idx, te_tgt_idx = train_test_split(np.arange(len(X_tgt)), test_size=0.40, random_state=SEED)
        ds_tgt_tr = BatterySOCDataset(X_tgt[tr_tgt_idx], y_tgt[tr_tgt_idx], domain_label=1)
        ds_tgt_te = BatterySOCDataset(X_tgt[te_tgt_idx], y_tgt[te_tgt_idx], domain_label=1)
        loader_tgt_tr = DataLoader(ds_tgt_tr, batch_size=4, shuffle=True)
        loader_tgt_te = DataLoader(ds_tgt_te, batch_size=4, shuffle=False)

        # 1. Zero-Shot Evaluation (no DANN adaptation)
        zs_metrics = evaluate_model(source_model, loader_tgt_te, device=device)
        logger.info(f"[{domain_label_str} Zero-Shot Test ] MAPE: {zs_metrics['MAPE_%']:.2f}% | Median MAPE: {zs_metrics['Median_MAPE_%']:.2f}% | R²: {zs_metrics['R2']:.3f}")
        results_table.append({
            "Domain": domain_label_str,
            "Architecture": "Koopman Neural Operator (KNO)",
            "Condition": "Zero-Shot (No Adaptation)",
            "MAPE_%": zs_metrics["MAPE_%"],
            "Median_MAPE_%": zs_metrics["Median_MAPE_%"],
            "RMSE_cycles": zs_metrics["RMSE_cycles"],
            "R2": zs_metrics["R2"]
        })

        # 2. Domain Adversarial Adaptation (DANN)
        dann_model = BatteryKoopmanDANN(in_features=200, num_cycles=46, d_model=64)
        dann_model.load_state_dict(torch.load(src_ckpt, map_location=device))

        dann_model = train_dann_loop(
            dann_model, loader_src_tr, loader_tgt_tr, loader_tgt_te,
            epochs=args.epochs_dann, lr=args.lr_dann,
            lambda_koopman=args.lambda_koopman, lambda_dann=args.lambda_dann,
            device=device, target_name=f"{domain_label_str}_DANN"
        )

        dann_metrics = evaluate_model(dann_model, loader_tgt_te, device=device)
        logger.info(f"[{domain_label_str} DANN Test      ] MAPE: {dann_metrics['MAPE_%']:.2f}% | Median MAPE: {dann_metrics['Median_MAPE_%']:.2f}% | R²: {dann_metrics['R2']:.3f}")
        results_table.append({
            "Domain": domain_label_str,
            "Architecture": "Koopman DANN (Explicit Adaptation)",
            "Condition": "Domain-Adversarially Aligned",
            "MAPE_%": dann_metrics["MAPE_%"],
            "Median_MAPE_%": dann_metrics["Median_MAPE_%"],
            "RMSE_cycles": dann_metrics["RMSE_cycles"],
            "R2": dann_metrics["R2"]
        })

        out_ckpt = f"checkpoints/koopman_dann_{tag}_transfer.pth"
        torch.save(dann_model.state_dict(), out_ckpt)
        logger.info(f"Saved DANN checkpoint -> {out_ckpt}")

    # Export quantitative summary
    results_df = pd.DataFrame(results_table)
    out_csv = "results/domain_adversarial_metrics.csv"
    results_df.to_csv(out_csv, index=False)

    logger.info("======================================================================")
    logger.info("KOOPMAN DANN DOMAIN ADAPTATION SUMMARY:")
    logger.info("======================================================================")
    for row in results_table:
        logger.info(f"  [{row['Domain']:18s} | {row['Condition']:28s}] MAPE: {row['MAPE_%']:6.2f}% | Median MAPE: {row['Median_MAPE_%']:6.2f}% | R²: {row['R2']:6.3f}")
    logger.info("======================================================================")
    logger.info(f"Quantitative comparison saved -> {out_csv}")


if __name__ == "__main__":
    main()
