#!/usr/bin/env python3
"""
Stage 0 Benchmark Summary & Comparison Reporter.

Reads per-arm JSON results from `results/stage0_arm*.json`, formats a
comparison table comparing Arm A (18-D baseline), Arm B (30-D Q,I deltas),
and Arm C (36-D all deltas), evaluates Stage 0 pass criteria, and outputs
both `results/stage0_comparison.json` and `results/stage0_comparison.md`.
"""

import os
import sys
import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Stage 0 Comparison Reporter")
    parser.add_argument("--results-dir", default="results", help="Directory containing per-arm JSON results")
    parser.add_argument("--out-dir", default="results", help="Directory to save comparison outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    arms = [
        ("armA", "Arm A (Baseline 18-D)", "None (within-cycle stats)"),
        ("armB", "Arm B (Rate 30-D)", "Q, I deltas (12 rate feats)"),
        ("armC", "Arm C (All 36-D)", "V, I, Q deltas (18 rate feats)")
    ]

    loaded_data = {}
    for arm_id, arm_label, desc in arms:
        json_path = os.path.join(args.results_dir, f"stage0_{arm_id}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                loaded_data[arm_id] = json.load(f)
        else:
            loaded_data[arm_id] = None

    table_rows = []
    comparison_json = {
        "benchmark": "Stage 0 Cross-Chemistry Transfer (MATR -> HUST)",
        "arms": {}
    }

    print("\n" + "=" * 80)
    print("STAGE 0 CROSS-CHEMISTRY BENCHMARK COMPARISON TABLE")
    print("=" * 80)

    header = (
        f"{'Arm':<24} | {'Dim':<5} | {'Test R² (Mean±Std)':<20} | "
        f"{'Test RMSE (cyc)':<18} | {'Paper MAPE':<16} | {'Ensemble R²':<12}"
    )
    print(header)
    print("-" * 80)

    md_lines = [
        "# Stage 0 Benchmark Comparison: Cross-Cycle Rate Features (MATR → HUST)\n",
        "## Quantitative Summary (5-Seed Repetitions)\n",
        "| Arm | Features | Delta Signals | Test R² (Mean ± Std) | Test RMSE (cyc) | Paper MAPE | Ensemble R² | Ensemble RMSE |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for arm_id, arm_label, desc in arms:
        data = loaded_data.get(arm_id)
        if data is None:
            print(f"{arm_label:<24} | {'N/A':<5} | {'NOT RUN / PENDING':<20} | {'-':<18} | {'-':<16} | {'-':<12}")
            md_lines.append(f"| **{arm_label}** | N/A | {desc} | *Pending* | - | - | - | - |")
            continue

        feat_dim = data.get("input_dim", "?")
        mean_r2 = data.get("mean_r2", 0.0)
        std_r2 = data.get("std_r2", 0.0)
        mean_rmse = data.get("mean_rmse", 0.0)
        std_rmse = data.get("std_rmse", 0.0)
        mean_mape = data.get("mean_paper_mape", 0.0)
        std_mape = data.get("std_paper_mape", 0.0)
        ens_r2 = data.get("ensemble_r2", 0.0)
        ens_rmse = data.get("ensemble_rmse", 0.0)

        r2_str = f"{mean_r2:.4f} ± {std_r2:.4f}"
        rmse_str = f"{mean_rmse:.2f} ± {std_rmse:.2f}"
        mape_str = f"{mean_mape:.2f}% ± {std_mape:.2f}%"
        ens_r2_str = f"{ens_r2:.4f}"
        ens_rmse_str = f"{ens_rmse:.2f} cyc"

        print(f"{arm_label:<24} | {feat_dim:<5} | {r2_str:<20} | {rmse_str:<18} | {mape_str:<16} | {ens_r2_str:<12}")
        md_lines.append(
            f"| **{arm_label}** | {feat_dim}-D | {desc} | {r2_str} | {rmse_str} | {mape_str} | {ens_r2_str} | {ens_rmse_str} |"
        )

        comparison_json["arms"][arm_id] = {
            "label": arm_label,
            "description": desc,
            "input_dim": feat_dim,
            "mean_r2": mean_r2,
            "std_r2": std_r2,
            "mean_rmse": mean_rmse,
            "std_rmse": std_rmse,
            "mean_paper_mape": mean_mape,
            "std_paper_mape": std_mape,
            "ensemble_r2": ens_r2,
            "ensemble_rmse": ens_rmse,
            "individual_runs": data.get("individual_runs", [])
        }

    print("=" * 80)

    # Pass Criteria Evaluation
    md_lines.append("\n## Stage 0 Pass Criteria Assessment\n")
    arm_a = loaded_data.get("armA")
    arm_b = loaded_data.get("armB")
    arm_c = loaded_data.get("armC")

    baseline_r2 = arm_a.get("mean_r2", 0.863) if arm_a else 0.863
    baseline_std = arm_a.get("std_r2", 0.02) if arm_a else 0.02

    md_lines.append(f"1. **Quantitative Criterion**: R² > baseline ({baseline_r2:.4f}) by more than seed std ({baseline_std:.4f}):")
    if arm_b:
        b_r2 = arm_b.get("mean_r2", 0.0)
        diff_b = b_r2 - baseline_r2
        status_b = "PASSED" if diff_b > baseline_std else ("MARGINAL" if diff_b > 0 else "FAILED")
        md_lines.append(f"   - **Arm B (30-D Q,I)**: Mean R² = {b_r2:.4f} (Δ = {diff_b:+.4f}) -> **{status_b}**")
    else:
        md_lines.append("   - **Arm B (30-D Q,I)**: Pending execution.")

    if arm_c:
        c_r2 = arm_c.get("mean_r2", 0.0)
        diff_c = c_r2 - baseline_r2
        status_c = "PASSED" if diff_c > baseline_std else ("MARGINAL" if diff_c > 0 else "FAILED")
        md_lines.append(f"   - **Arm C (36-D All)**: Mean R² = {c_r2:.4f} (Δ = {diff_c:+.4f}) -> **{status_c}**")
    else:
        md_lines.append("   - **Arm C (36-D All)**: Pending execution.")

    md_lines.append("\n2. **Qualitative Criterion 1 (Figure A)**: Predicted starting RUL in `figures/stage0_<arm>/A_trajectories.png` spreads across lifespans (950–2280 cyc) rather than clustering at 1400–2000 cyc.")
    md_lines.append("3. **Qualitative Criterion 2 (Figure F)**: Bias curve in `figures/stage0_<arm>/F_error_anatomy.png` flattens above RUL 1000 cyc.")

    md_content = "\n".join(md_lines) + "\n"

    # Write results files
    md_out_file = os.path.join(args.out_dir, "stage0_comparison.md")
    json_out_file = os.path.join(args.out_dir, "stage0_comparison.json")

    with open(md_out_file, "w") as f:
        f.write(md_content)
    with open(json_out_file, "w") as f:
        json.dump(comparison_json, f, indent=2)

    print(f"\nSaved comparison summary to:\n  - {md_out_file}\n  - {json_out_file}\n")


if __name__ == "__main__":
    main()
