#!/usr/bin/env python3
"""
Visualization & Reporting Module for Lithium-Ion Battery SOH & RUL Estimation.
Generates publication-quality figures:
  1. figures/dqdv_curve_evolution.png - Raw vs. Smoothed dQ/dV curves at Cycles 10, 50, 100
  2. figures/feature_importance.png - Top physics features ranked by importance/coefficients
  3. figures/predicted_vs_actual_cycle_life.png - Parity plot comparing Baseline vs. Physics models with +/-10% error bounds
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Visualization")

# Professional publication color palette
PALETTE = {
    "primary": "#2563EB",      # Blue
    "secondary": "#0D9488",    # Teal
    "accent": "#DC2626",       # Crimson Red
    "purple": "#7C3Aed",       # Violet
    "amber": "#D97706",        # Amber
    "dark": "#1E293B",         # Slate Dark
    "light": "#F8FAFC",        # Slate Light
    "grid": "#E2E8F0"          # Grid grey
}


def setup_style():
    """Applies modern clean aesthetic for publication figures."""
    plt.rcParams.update({
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.family": "sans-serif",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        "xtick.color": PALETTE["dark"],
        "ytick.color": PALETTE["dark"],
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": "#CBD5E1"
    })


def plot_dqdv_evolution(dqdv_path: str, out_dir: str):
    """
    Plots Raw vs Smoothed dQ/dV curves at Cycles 10, 50, 100 for representative cells,
    demonstrating Savitzky-Golay peak preservation and electrochemical phase-transition shifts.
    """
    logger.info("Generating dqdv_curve_evolution.png...")
    df = pd.read_parquet(dqdv_path)
    
    # Pick two representative cells (one high cycle life, one moderate/low cycle life)
    cells = sorted(df["cell_id"].unique())
    cell1 = cells[0] if len(cells) > 0 else "b1c0"
    cell2 = cells[len(cells)//2] if len(cells) > 1 else cell1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    # Panel 1: Raw vs Smoothed dQ/dV at Cycle 10 and 100 for cell1
    ax = axes[0]
    c10 = df[(df["cell_id"] == cell1) & (df["cycle_number"] == 10)]
    c100 = df[(df["cell_id"] == cell1) & (df["cycle_number"] == 100)]

    if len(c10) > 0:
        ax.plot(c10["voltage_V"], c10["dqdv_raw"], color="#94A3B8", alpha=0.5, label="Cycle 10 (Raw Noise)")
        ax.plot(c10["voltage_V"], c10["dqdv_smooth"], color=PALETTE["primary"], linewidth=2.5, label="Cycle 10 (Savitzky-Golay)")
    if len(c100) > 0:
        ax.plot(c100["voltage_V"], c100["dqdv_smooth"], color=PALETTE["accent"], linewidth=2.5, linestyle="--", label="Cycle 100 (Savitzky-Golay)")

    ax.set_title(f"A. Savitzky-Golay Peak Preservation ({cell1})")
    ax.set_xlabel("Discharge Voltage (V)")
    ax.set_ylabel("Differential Capacity dQ/dV (Ah/V)")
    ax.set_xlim(2.8, 3.5)
    ax.legend(loc="lower left")

    # Annotate LFP Phase-Transition Peaks
    ax.annotate("Peak 1 (~3.30V)\nHigh-V Plateau", xy=(3.30, -1.8), xytext=(3.32, -0.8),
                arrowprops=dict(facecolor=PALETTE["dark"], shrink=0.05, width=1.0, headwidth=6),
                fontweight="bold", color=PALETTE["dark"])
    ax.annotate("Peak 2 (~3.22V)\nLow-V Plateau", xy=(3.22, -1.2), xytext=(3.02, -0.5),
                arrowprops=dict(facecolor=PALETTE["dark"], shrink=0.05, width=1.0, headwidth=6),
                fontweight="bold", color=PALETTE["dark"])

    # Panel 2: Cycle Evolution (Cycles 10, 50, 100)
    ax = axes[1]
    colors = {10: PALETTE["primary"], 50: PALETTE["amber"], 100: PALETTE["accent"]}
    for cyc in [10, 50, 100]:
        sub = df[(df["cell_id"] == cell1) & (df["cycle_number"] == cyc)]
        if len(sub) > 0:
            ax.plot(sub["voltage_V"], sub["dqdv_smooth"], color=colors[cyc], linewidth=2.2, label=f"Cycle {cyc}")

    ax.set_title(f"B. dQ/dV Curve Evolution ({cell1})")
    ax.set_xlabel("Discharge Voltage (V)")
    ax.set_ylabel("dQ/dV (Ah/V)")
    ax.set_xlim(2.9, 3.45)
    ax.legend(loc="lower left")

    # Panel 3: Delta(dQ/dV) Difference Curve (Cycle 100 - Cycle 10)
    ax = axes[2]
    if len(c10) > 0 and len(c100) > 0:
        v_grid = c10["voltage_V"].values
        delta_dqdv = c100["dqdv_smooth"].values - c10["dqdv_smooth"].values
        ax.plot(v_grid, delta_dqdv, color=PALETTE["purple"], linewidth=2.5, label="Δ(dQ/dV) = Cyc 100 - Cyc 10")
        ax.fill_between(v_grid, 0, delta_dqdv, color=PALETTE["purple"], alpha=0.15)
        ax.axhline(0, color=PALETTE["dark"], linestyle="--", linewidth=1.0)

    ax.set_title("C. Differential Capacity Shift Δ(dQ/dV)")
    ax.set_xlabel("Discharge Voltage (V)")
    ax.set_ylabel("Δ(dQ/dV) Difference (Ah/V)")
    ax.set_xlim(2.8, 3.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "dqdv_curve_evolution.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure -> {out_file}")


def plot_feature_importance(fi_path: str, out_dir: str):
    """
    Plots top physics features ranked by importance/coefficients for Full Physics-Informed models.
    """
    logger.info("Generating feature_importance.png...")
    df = pd.read_parquet(fi_path)
    sub = df[df["model_suite"] == "3_Full_Physics_Informed"].copy()

    if len(sub) == 0:
        logger.warning("No feature importance records found for 3_Full_Physics_Informed.")
        return

    # Aggregate importance across algorithms (mean normalized importance)
    agg = sub.groupby("feature")["importance"].mean().reset_index()
    agg = agg.sort_values(by="importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    y_pos = np.arange(len(agg))
    bars = ax.barh(y_pos, agg["importance"], color=PALETTE["primary"], edgecolor=PALETTE["dark"], alpha=0.85)

    # Highlight top 3 features with accent teal color
    for idx in [-1, -2, -3]:
        if abs(idx) <= len(bars):
            bars[idx].set_color(PALETTE["secondary"])

    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg["feature"], fontweight="bold")
    ax.set_xlabel("Normalized Mean Importance / Relative Weight")
    ax.set_title("Top 15 Physics-Informed dQ/dV & Early-Cycle Features (N=124 Cells)")

    # Value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2.0, f"{width:.3f}",
                va="center", ha="left", fontsize=9, fontweight="bold", color=PALETTE["dark"])

    ax.set_xlim(0, max(agg["importance"]) * 1.15)
    plt.tight_layout()
    out_file = os.path.join(out_dir, "feature_importance.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure -> {out_file}")


def plot_predicted_vs_actual(pred_path: str, metrics_path: str, out_dir: str):
    """
    Plots Parity Plot comparing Baseline vs. Physics models with +/-10% error bounds.
    """
    logger.info("Generating predicted_vs_actual_cycle_life.png...")
    df = pd.read_parquet(pred_path)
    metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None

    # We will plot 3 key suites:
    # 1. Baseline Naive (LightGBM or ElasticNet)
    # 2. Benchmark Severson (ElasticNet)
    # 3. Full Physics-Informed (ElasticNet or LightGBM)
    
    fig, ax = plt.subplots(figsize=(8.5, 8.5))

    min_val = min(df["actual_cycle_life"].min(), df["predicted_cycle_life"].min()) * 0.85
    max_val = max(df["actual_cycle_life"].max(), df["predicted_cycle_life"].max()) * 1.10
    grid_vals = np.linspace(min_val, max_val, 200)

    # Shaded +/- 10% error bound
    ax.fill_between(grid_vals, grid_vals * 0.90, grid_vals * 1.10, color=PALETTE["secondary"], alpha=0.15, label="±10% Error Bound")
    
    # Shaded +/- 20% error bound lines
    ax.plot(grid_vals, grid_vals * 0.80, color=PALETTE["dark"], linestyle=":", linewidth=1.2, alpha=0.5, label="±20% Error Bound")
    ax.plot(grid_vals, grid_vals * 1.20, color=PALETTE["dark"], linestyle=":", linewidth=1.2, alpha=0.5)

    # Identity line y = x
    ax.plot(grid_vals, grid_vals, color=PALETTE["dark"], linestyle="-", linewidth=2.0, label="Parity (y = x)")

    # Plot scatters
    models_to_plot = [
        ("1_Baseline_Naive", "LightGBM", "#94A3B8", "v", "1. Baseline Naive (LightGBM)"),
        ("2_Benchmark_Severson", "ElasticNet", PALETTE["amber"], "s", "2. Benchmark Severson (ElasticNet)"),
        ("3_Full_Physics_Informed", "LightGBM", PALETTE["primary"], "o", "3. Full Physics-Informed (LightGBM)")
    ]

    for suite_name, algo_name, color, marker, label in models_to_plot:
        sub = df[(df["model_suite"] == suite_name) & (df["algorithm"] == algo_name)]
        if len(sub) > 0:
            ax.scatter(sub["actual_cycle_life"], sub["predicted_cycle_life"],
                       color=color, marker=marker, s=65, alpha=0.85, edgecolor="white",
                       linewidth=0.8, label=label)

    hybrid_path = "results/hybrid_cnn_xgb_predictions.parquet"
    if os.path.exists(hybrid_path):
        hdf = pd.read_parquet(hybrid_path)
        ax.scatter(hdf["actual_cycle_life"], hdf["predicted_cycle_life"],
                   color=PALETTE["accent"], marker="P", s=90, alpha=0.95, edgecolor="darkred",
                   linewidth=1.0, label="4. Hybrid 1D-CNN + XGBoost Regressor")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual Cycle Life (Cycles)")
    ax.set_ylabel("Predicted Cycle Life (Cycles)")
    ax.set_title("Predicted vs. Actual Battery Cycle Life (Held-Out Test Split)")
    ax.legend(loc="upper left")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "predicted_vs_actual_cycle_life.png")
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure -> {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Visualization & Publication Figure Generation")
    parser.add_argument("--proc-dir", type=str, default="data/processed", help="Path to processed data directory")
    parser.add_argument("--res-dir", type=str, default="results", help="Path to results directory")
    parser.add_argument("--out-dir", type=str, default="figures", help="Path to output figures directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    setup_style()

    dqdv_path = os.path.join(args.proc_dir, "dqdv_curves.parquet")
    fi_path = os.path.join(args.res_dir, "feature_importances.parquet")
    pred_path = os.path.join(args.res_dir, "model_predictions.parquet")
    metrics_path = os.path.join(args.res_dir, "model_evaluation_metrics.csv")

    if os.path.exists(dqdv_path):
        plot_dqdv_evolution(dqdv_path, args.out_dir)
    else:
        logger.warning(f"File not found: {dqdv_path}")

    if os.path.exists(fi_path):
        plot_feature_importance(fi_path, args.out_dir)
    else:
        logger.warning(f"File not found: {fi_path}")

    if os.path.exists(pred_path):
        plot_predicted_vs_actual(pred_path, metrics_path, args.out_dir)
    else:
        logger.warning(f"File not found: {pred_path}")

    logger.info("All Phase 4 publication figures generated successfully!")


if __name__ == "__main__":
    main()
