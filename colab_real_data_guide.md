# Colab Guide: Running on REAL Battery Data (No Synthetic Fallbacks)

> **IMPORTANT:** This guide uses ONLY real, physical battery datasets.
> The old synthetic `np.linspace` data has been completely replaced.

## Step 1: Clone the Repository

```python
!git clone https://github.com/divyansh070/ashwini_prof_project.git
%cd ashwini_prof_project
```

## Step 2: Install Dependencies

```python
!pip install torch numpy pandas scipy scikit-learn matplotlib optuna
```

## Step 3: Download REAL Datasets

This downloads:
- **Stanford/MIT LFP** (124 cells) from HuggingFace Severson-2019 mirror
- **NASA Ames LCO** (4 cells) from official NASA S3 bucket

```python
!python3 src/download_real_data.py --raw-dir data/raw --proc-dir data/real_processed
```

**Expected output:**
```
✅ REAL Stanford LFP: 124 cells | ~12M rows | EOL range: [110, 1934]
✅ REAL NASA LCO: 4 cells | ~50K rows | Cell lives: {B0005: 127, B0006: 168, ...}
```

> If you see "FATAL" errors, the download failed. DO NOT proceed with synthetic data.

## Step 4: Preprocess into Koopman SOC Tensors

```python
!python3 src/preprocess_real_data.py --in-dir data/real_processed --out-dir data/real_koopman
```

**Expected output:**
```
✅ Voltage steps verified as non-uniform (REAL sensor data)
✅ Saved REAL LFP tensor -> data/real_koopman/real_stanford_lfp_soc.npz (Shape: (124, 46, 200))
✅ Saved REAL LCO tensor -> data/real_koopman/real_nasa_lco_soc.npz (Shape: (4, 46, 200))
```

## Step 5: Train Koopman on Real Stanford LFP (Source Domain)

```python
!python3 src/koopman/train_da_colab.py \
    --source-path data/real_koopman/real_stanford_lfp_soc.npz \
    --target-paths data/real_koopman/real_nasa_lco_soc.npz \
    --target-labels "NASA LCO" \
    --epochs 100 \
    --lr 1e-3 \
    --batch-size 16
```

## Step 6: Verify Results

```python
import pandas as pd

print("=" * 85)
print("REAL DATA RESULTS (Linear-Space Cycles)")
print("=" * 85)
df = pd.read_csv("results/domain_adversarial_metrics.csv")
display(df)
```

## Step 7: Run Leakage Audit

```python
!python3 src/leakage_audit.py
```

---

## Dataset Summary

| Dataset | Cells | Chemistry | Source | Status |
|---------|-------|-----------|--------|--------|
| Stanford/MIT LFP (2019) | 124 | LiFePO₄ | HuggingFace mirror | ✅ REAL |
| NASA Ames LCO (2007) | 4 | LiCoO₂ | NASA S3 bucket | ✅ REAL |

## Notes

- The NASA dataset has only 4 cells (B0005, B0006, B0007, B0018). This is small but
  represents genuine physical LCO degradation data with real sensor noise.
- Cross-chemistry transfer: Stanford LFP (source) → NASA LCO (target) is a legitimate
  cross-chemistry experiment since LFP and LCO have fundamentally different voltage
  plateaus and degradation mechanisms.
- For larger-scale LCO/NMC experiments, you would need to manually download the Oxford
  or CALCE datasets from their university portals (requires registration).
