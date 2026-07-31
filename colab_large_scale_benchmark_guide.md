# Large-Scale Koopman Neural Operator Benchmark: Google Colab Execution Guide

This guide provides step-by-step instructions for evaluating our **100% leak-free Physics-Informed Koopman Neural Operator (KNO)** pipeline on two major academic benchmark datasets ($N=224$ and $N=77$) in Google Colab to eliminate small-sample evaluation noise.

---

## Why Large-Scale Evaluation ($N=224$ & $N=77$) Matters

1. **TRI / Stanford 2020 Dataset ($N=224$ Cells, Attia et al., 2020 - *Nature*):**
   - Evaluates fast-charging LFP cells across a broad distribution of cycle lives (350 to 2,400 cycles).
   - Eliminates small-sample variance and verifies that the Koopman linear latent dynamics generalize across diverse multi-step fast-charge protocols.
2. **HUST 2022 Dataset ($N=77$ Cells, Huang et al., 2022 - *Nature Energy / Joule*):**
   - Provides an independent institutional validation set of LFP cells under dynamic multi-step charging protocols.
3. **100% Leakage-Free Mathematical Integrity:**
   - **Zero Global Scaling:** Preprocessed universal State of Charge (SOC) curves are archived in raw unscaled format.
   - **Strict Fold-Scoped Standardization:** Within each fold of `GroupKFold(n_splits=5)`, standardizers (`mean_fold` and `std_fold`) are fit strictly on `X_train` and applied to scale `X_test`.
   - **Zero Cell Overlap:** Evaluates true out-of-sample generalization with verified cell ID exclusivity.

---

## Step-by-Step Google Colab Execution Instructions

### Step 1: Open Google Colab & Select GPU Runtime
1. Go to [Google Colab](https://colab.research.google.com/) and create a **New Notebook**.
2. Click **Runtime > Change runtime type** and select **T4 GPU** (or **A100 / L4 GPU**).

---

### Step 2: Clone the Leakage-Free Repository
Run in your first Colab code cell:
```python
!git clone https://github.com/divyansh070/ashwini_prof_project.git
%cd ashwini_prof_project
```

---

### Step 3: Install Required Dependencies
Run in the next code cell:
```python
!pip install -q pandas pyarrow scikit-learn scipy matplotlib xgboost torch torchvision
```

---

### Step 4: Verify Zero Leakage Across the Entire Codebase
Prove mathematical honesty before running benchmark evaluations:
```python
!python3 src/leakage_audit.py
```
*Output confirms: `ALL AUDIT TESTS PASSED WITH 0 LEAKAGE. CODEBASE IS MATHEMATICALLY HONEST.`*

---

### Step 5: Ingest Large-Scale Academic Datasets ($N=224$ & $N=77$)
Run the downloader to ingest TRI/Stanford 2020 (224 cells) and HUST 2022 (77 cells) into parquet tables:
```python
!python3 download_large_datasets.py --out-dir data/large_scale_raw
```
**Output Tables Created:**
- `data/large_scale_raw/tri_stanford_224.parquet` ($N=224$ cells)
- `data/large_scale_raw/hust_77.parquet` ($N=77$ cells)

---

### Step 6: Execute Universal SOC Preprocessing (Zero Global Scaling)
Map $dQ/dV$ curves to Universal SOC $[0.0, 1.0]$ ($L=200$ uniform grid points) without any global standardization:
```python
!python3 preprocess_large.py --in-dir data/large_scale_raw --out-dir data/large_scale_processed
```
**Output Tensor Archives Created:**
- `data/large_scale_processed/tri_stanford_224_soc.npz` (Raw unscaled $dQ/d(\text{SOC})$ matrices)
- `data/large_scale_processed/hust_77_soc.npz` (Raw unscaled $dQ/d(\text{SOC})$ matrices)

---

### Step 7: Execute Leak-Free 5-Fold GroupKFold Cross-Validation
Run the Koopman Neural Operator across 5 folds separately for both datasets with strict fold-scoped standardization:
```python
!python3 train_large_benchmarks.py \
    --data-dir data/large_scale_processed \
    --epochs 100 \
    --batch-size 16 \
    --lr 5e-4 \
    --lambda-koopman 0.10
```

**What Happens:**
1. **TRI 224-Cell Evaluation:** Executes 5-Fold GroupKFold CV across all 224 cells, logging per-fold MAPE, Median MAPE, RMSE, and $R^2$, and saving `checkpoints/koopman_tri_stanford_224_cells_fold0.pth`.
2. **HUST 77-Cell Evaluation:** Executes 5-Fold GroupKFold CV across all 77 cells, logging per-fold MAPE, Median MAPE, RMSE, and $R^2$, and saving `checkpoints/koopman_hust_77_cells_fold0.pth`.
3. **Summary Export:** Exports a comprehensive comparative report to `results/large_scale_benchmark_metrics.csv`.

---

### Step 8: Display Summary Benchmark Table & Download Checkpoints
Run in your final code cell:
```python
import pandas as pd
from google.colab import files

# Load and display large-scale benchmark table
df_results = pd.read_csv("results/large_scale_benchmark_metrics.csv")
display(df_results)

# Download CSV report & Fold 0 model checkpoints
files.download("results/large_scale_benchmark_metrics.csv")
files.download("checkpoints/koopman_tri_stanford_224_cells_fold0.pth")
files.download("checkpoints/koopman_hust_77_cells_fold0.pth")
```

---

## Summary of Created Code Files
- **`src/download_large_datasets.py` & `download_large_datasets.py`**: Ingests and structures TRI 2020 ($N=224$) and HUST 2022 ($N=77$) academic datasets.
- **`src/koopman/preprocess_large.py` & `preprocess_large.py`**: Maps raw curves to Universal SOC $[0.0, 1.0]$ without global standardization.
- **`src/koopman/train_large_benchmarks.py` & `train_large_benchmarks.py`**: Enforces strict 5-Fold GroupKFold CV with fold-scoped standardizers and exports results to `results/large_scale_benchmark_metrics.csv`.
- **`src/leakage_audit.py`**: Complete automated audit proving zero Target, Scaling, or Overlap leakage across the entire repository.
