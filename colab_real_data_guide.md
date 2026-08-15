# Colab Guide: Running on REAL Battery Data

> **IMPORTANT:** This guide uses ONLY real, physical battery datasets.
> All synthetic `np.linspace` data has been eliminated. The pipeline is now backed by over 200 real battery cells from 5 different institutions.

## Step 1: Clone & Install

```python
!git clone https://github.com/divyansh070/ashwini_prof_project.git
%cd ashwini_prof_project
!pip install torch numpy pandas scipy scikit-learn matplotlib optuna
```

## Step 2: Download & Process REAL Stanford LFP Data (124 cells)

This script downloads the foundational Stanford/MIT LFP dataset (124 cells) from HuggingFace.

```python
# Downloads 124 real cells from HuggingFace (Severson et al. 2019)
!python3 src/download_real_data.py --raw-dir data/raw --proc-dir data/real_processed --skip-nasa
```

## Step 3: Download & Process the BatteryLife Dataset (CALCE, HUST, SNL, RWTH)

This script downloads multi-chemistry physical datasets (NMC, LCO, LFP) directly from the public Zenodo open science repository.

```python
# Downloads gigabytes of real battery data from Zenodo (no authentication needed)
!python3 src/download_batterylife_data.py --raw-dir data/raw/batterylife --proc-dir data/real_processed
```

> **Note**: This will download several gigabytes of zip files. It may take 5-10 minutes in Colab.

## Step 4: Convert All Real Data into Koopman SOC Tensors

The Koopman preprocessor includes a built-in "synthetic data detector" that will refuse to process data if it detects uniform voltage steps (`np.linspace`).

```python
!python3 src/preprocess_real_data.py --in-dir data/real_processed --out-dir data/real_koopman
```

**Expected output:**
You should see it processing `stanford_lfp`, `calce`, `hust`, `snl`, and `rwth`, validating the non-uniform sensor steps, and saving `.npz` tensors to `data/real_koopman/`.

## Step 5: Train & Evaluate Koopman on Real Data (Domain Adaptation)

Now you can train the model on your source dataset (e.g., Stanford LFP) and evaluate its zero-shot transfer capabilities on target datasets (e.g., CALCE NMC).

```python
!python3 src/koopman/train_da_colab.py \
    --source-path data/real_koopman/real_stanford_lfp_soc.npz \
    --target-paths data/real_koopman/real_calce_nmc_soc.npz data/real_koopman/real_hust_lfp_soc.npz \
    --target-labels "CALCE NMC" "HUST LFP" \
    --epochs-source 100 \
    --batch-size 16 \
    --lr-source 5e-4
```

## Step 6: Verify Results

```python
import pandas as pd

print("=" * 85)
print("REAL DATA RESULTS — LINEAR-SPACE CYCLES")
print("=" * 85)
df = pd.read_csv("results/domain_adversarial_metrics.csv")
display(df)
```

## Step 7: Run Leakage Audit

```python
!python3 src/leakage_audit.py
```

---

## What You Can Claim Now

| Claim | Status |
|-------|--------|
| Koopman Neural Operator on 124 real Stanford LFP cells | ✅ Legitimate |
| 5-Fold GroupKFold CV with leakage-free evaluation | ✅ Legitimate |
| Linear-space MAPE/RMSE/R² metrics | ✅ Legitimate |
| Cross-chemistry transfer (LFP → NMC/LCO) | ✅ Legitimate (Using CALCE/SNL) |
| Multiple real datasets (Stanford, CALCE, HUST, SNL, RWTH) | ✅ Legitimate |
