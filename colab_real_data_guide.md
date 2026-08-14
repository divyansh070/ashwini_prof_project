# Colab Guide: Running on REAL Battery Data

> **IMPORTANT:** This guide uses ONLY real, physical battery datasets.
> All synthetic `np.linspace` data has been eliminated.

## Step 1: Clone & Install

```python
!git clone https://github.com/divyansh070/ashwini_prof_project.git
%cd ashwini_prof_project
!pip install torch numpy pandas scipy scikit-learn matplotlib optuna
```

## Step 2: Download & Process REAL Stanford LFP Data (124 cells)

```python
# Downloads 124 real cells from HuggingFace (Severson et al. 2019)
!python3 src/download_real_data.py --raw-dir data/raw --proc-dir data/real_processed --skip-nasa

# Convert to Koopman SOC tensors
!python3 src/preprocess_real_data.py --in-dir data/real_processed --out-dir data/real_koopman
```

**Expected output:**
```
✅ REAL Stanford LFP: 124 cells | ~2.3M rows | EOL range: [110, 1934]
✅ Voltage steps verified as non-uniform (REAL sensor data)
✅ Saved REAL LFP tensor -> data/real_koopman/real_stanford_lfp_soc.npz (Shape: (124, 46, 200))
```

## Step 3: Train & Evaluate Koopman on Real Data

This runs 5-Fold GroupKFold Cross-Validation on 124 real LFP cells:

```python
!python3 src/koopman/train_da_colab.py \
    --source-path data/real_koopman/real_stanford_lfp_soc.npz \
    --epochs-source 100 \
    --batch-size 16 \
    --lr-source 5e-4
```

## Step 4: Verify Results

```python
import pandas as pd

print("=" * 85)
print("REAL DATA RESULTS — LINEAR-SPACE CYCLES")
print("=" * 85)
df = pd.read_csv("results/domain_adversarial_metrics.csv")
display(df)
```

## Step 5: Run Leakage Audit

```python
!python3 src/leakage_audit.py
```

---

## What You Can Claim

| Claim | Status |
|-------|--------|
| Koopman Neural Operator on 124 real LFP cells | ✅ Legitimate |
| 5-Fold GroupKFold CV with leakage-free evaluation | ✅ Legitimate |
| Linear-space MAPE/RMSE/R² metrics | ✅ Legitimate |
| Cross-chemistry transfer (LFP → LCO) | ❌ Needs real LCO data |
| 445 cells across 5 datasets | ❌ Only 124 real cells currently |

## How to Add Real Cross-Chemistry Data (Future Work)

For real cross-chemistry transfer, you need to manually download:
1. **CALCE CS2 (NMC)**: Go to https://calce.umd.edu/battery-data, download the CS2 Excel files
2. **Oxford (LCO)**: Contact the Birkl et al. authors for access to the raw .mat files

Place raw files in `data/raw/calce_nmc/` or `data/raw/oxford_lco/`, then the existing
parsers in `src/patchtst/download_datasets.py` will detect and use them instead of
falling back to synthetic generation.
