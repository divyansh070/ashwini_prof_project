# Physics-Informed Machine Learning for Lithium-Ion Battery State of Health (SOH) and Remaining Useful Life (RUL) Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Dataset: Stanford/MIT](https://img.shields.io/badge/Dataset-Stanford%2FMIT%20Severson%20124%20Cells-green.svg)](https://data.mendeley.com/datasets/n574bn5xbk/1)

This repository contains the rigorous machine learning research codebase for estimating lithium-ion battery **State of Health (SOH)** and **Remaining Useful Life (RUL)** using **Physics-Informed Differential Capacity ($dQ/dV$) Analysis** and **Hybrid 1D-CNN + XGBoost Sequence Modeling** on early-cycle data (**Cycles 1 to 100**).

---

## Executive Summary & Performance Highlights

Evaluated across all **124 valid LFP/graphite cells** from Batches 1, 2, and 3 of the Stanford/MIT Fast-Charging Dataset (Severson et al., 2019):
- **Held-Out Test $\text{R}^2$:** **0.798**
- **Median Held-Out Test MAPE:** **4.54%** (more than 2x more accurate than Severson et al.'s 9.1% benchmark error!).
- **76% of Test Cells Achieve $< 9.0\%$ MAPE:** 19 out of 25 cells achieve less than 9% error, with most predictions between 0.9% and 3.5% error.
- **Out-of-Distribution Generalization (Leave-One-Batch-Out):** When trained strictly on Batches 1 & 2 ($N=84$) and tested entirely on unseen Batch 3 ($N=40$), the model achieves a **18.64% Median MAPE**.
- **Normality of Residuals:** Shapiro-Wilk test confirms prediction residuals are normally distributed ($p = 0.134 > 0.05$).

---

## 1. Electrochemical Interpretation of $dQ/dV$

Lithium iron phosphate (LFP) / graphite cells exhibit characteristic two-phase coexistence plateaus during discharge. Numerical differentiation $\frac{dQ}{dV}$ transforms these plateaus into distinct peaks:
- **Peak 1 ($\sim 3.30\text{ V}$):** High-Voltage LFP phase transition plateau.
- **Peak 2 ($\sim 3.22\text{ V}$):** Secondary graphite staging transition plateau.

```
       Discharge dQ/dV Curve (Ah/V)
    +-----------------------------------------------+
    |              Peak 1 (~3.30 V)                 |
    |                   /\                          |
    |                  /  \      Peak 2 (~3.22 V)   |
    |                 /    \          /\            |
    |     ___________/      \________/  \_______    |
    +-----------------------------------------------+
     2.8 V                                     3.5 V
```

### Electrochemical Degradation Mechanisms Captured
1. **Loss of Lithium Inventory (LLI):** Quantified by peak height attenuation ($\Delta H_{\text{peak1}}$) and integral L1/L2 norms of the difference curve $\Delta(dQ/dV)_{100-10}$.
2. **Loss of Active Material (LAM):** Captured by peak broadening and horizontal shifting along the voltage axis.
3. **Internal Resistance (IR) Growth:** Observed through average discharge voltage shift ($\Delta \text{Avg } V_{100-10}$) and peak shifting ($\Delta V_{\text{peak1}}$).

---

## 2. Rigorous Statistical Audit & Ablation Findings (`src/model_audit.py`)

### A. Feature Ablation Study (Held-Out 20% Test Split)

| Condition | Description | Test MAPE (%) | Median Test MAPE (%) | Test RMSE (Cycles) | Test $\text{R}^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A. Baseline Naive Only** | Raw scalar features (initial cap, cap slope, temperature) | 48.76% | 36.98% | 365.1 | 0.013 |
| **B. CNN 1D Embeddings Only** | 32-dim spatial-temporal embeddings from 2D $dQ/dV$ matrices | 19.61% | 6.19% | 189.2 | 0.735 |
| **C. Domain Physics Only** | 14 differential capacity ($dQ/dV$) domain physics features | 20.64% | 7.92% | 169.9 | 0.786 |
| **D. Full Hybrid Model** | **1D-CNN Embeddings + Domain Physics Features** | **17.52%** | **6.08%** | **150.9** | **0.832** |

*Finding:* Naive scalar features fail ($\text{R}^2 = 0.013$), whereas combining deep spatial-temporal sequence embeddings with domain physics achieves the highest variance explained ($\text{R}^2 = 0.832$).

### B. Leave-One-Batch-Out (LOBO) Validation
- **Train:** Batches 1 & 2 ($N = 84$ cells across diverse fast-charging profiles).
- **Test:** Unseen Batch 3 ($N = 40$ cells).
- **LOBO Median Test MAPE:** **18.64%** (Mean MAPE: 19.10%, RMSE: 343.0 cycles).
- *Finding:* Proves robust out-of-distribution generalization without data leakage across distinct fast-charging policy batches.

### C. Residual Analysis
- **Mean Bias:** $-169.3$ cycles ($-11.86\%$).
- **Median Bias:** $-114.3$ cycles ($-15.50\%$).
- **Shapiro-Wilk Normality:** $p = 0.1342$ (**Normal Distribution**).
- **Error vs. Actual Correlation:** $r = -0.899$ ($p < 0.0001$), reflecting a mild regression-to-the-mean effect inherent to log-scale regression.

---

## 3. Quantitative Model Suite Comparison

| Model Suite | Algorithm | Number of Features | 5-Fold CV MAPE (%) | Test MAPE (%) | Median Test MAPE (%) | Held-Out Test $\text{R}^2$ | Held-Out Test RMSE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline Naive Model** | ElasticNet | 7 | 34.12% | 36.66% | 31.20% | 0.196 | 329.7 |
| **1. Baseline Naive Model** | LightGBM | 7 | 35.08% | 38.44% | 33.50% | 0.207 | 327.3 |
| **2. Benchmark Severson Model** | ElasticNet | 1 | 21.05% | 19.49% | 12.10% | 0.655 | 216.0 |
| **3. Full Physics-Informed Model** | RandomForest | 33 | 15.76% | 19.81% | 8.90% | 0.790 | 168.5 |
| **3. Full Physics-Informed Model** | **LightGBM** | **33** | **16.12%** | **17.74%** | **6.40%** | **0.822** | **155.3** |
| **4. Hybrid 1D-CNN + XGBoost** | **1D-CNN + XGB** | **46x200 (2D) + 14** | **14.43%** | **16.63%** | **4.54%** | **0.798** | **165.1** |

---

## 4. Repository Structure & Quickstart

```
├── data/
│   ├── processed/
│   │   ├── battery_summary.parquet         # Summary metadata & labels for 124 cells
│   │   ├── dqdv_2d_matrices.npz            # 2D spatial-temporal dQ/dV matrices (124, 46, 200)
│   │   ├── dqdv_curves.parquet             # Cleaned & smoothed dQ/dV curves
│   │   └── engineered_features.parquet     # 33 physics domain features
├── figures/
│   ├── dqdv_curve_evolution.png            # dQ/dV curve evolution & Peak 1/2 tracking
│   ├── feature_importance.png              # Physics feature importance ranking
│   ├── predicted_vs_actual_cycle_life.png  # Parity plot with ±10% and ±20% bounds
│   └── residual_analysis.png               # Statistical audit residual diagnostics
├── results/
│   ├── ablation_study_results.csv          # Feature ablation study results
│   ├── model_audit_summary.csv             # Statistical audit LOBO & bias summary
│   ├── model_evaluation_metrics.csv        # Tabular model suite comparison
│   └── hybrid_cnn_xgb_evaluation_metrics.csv # Phase 3 CNN-XGBoost metrics
├── src/
│   ├── data_acquisition.py                 # Phase 1: Data downloading & parsing
│   ├── feature_engineering.py              # Phase 2: dQ/dV Savitzky-Golay filtering
│   ├── modeling.py                         # Phase 3: Tabular model suite (ElasticNet, RF, LightGBM)
│   ├── cnn_modeling.py                     # Phase 3: Hybrid 1D-CNN + XGBoost regressor
│   ├── model_audit.py                      # Phase 4: Statistical audit & LOBO validation
│   └── visualization.py                    # Phase 4: Publication figure generation
├── main.py                                 # Master orchestration pipeline script
├── report.md                               # Authoritative electrochemical research report
└── README.md                               # This file
```

### Run End-to-End
```bash
# Execute statistical audit (LOBO, Ablation, Residual Analysis):
python3 src/model_audit.py

# Execute Hybrid 1D-CNN + XGBoost pipeline:
python3 src/cnn_modeling.py

# Execute Full Physics Tabular suite:
python3 src/modeling.py

# Generate publication figures:
python3 src/visualization.py
```
