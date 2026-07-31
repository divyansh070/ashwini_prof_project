# Step-by-Step Google Colab Execution Guide: Multi-Dataset Transfer Learning (PatchTST)

This guide provides exact, step-by-step instructions to upload, preprocess, and execute the **Patch Time Series Transformer (PatchTST)** transfer learning framework across **Stanford/MIT (LFP)**, **Oxford (LCO)**, and **CALCE (NMC)** battery datasets on Google Colab GPU.

---

## 1. Which Files to Zip & Upload (or Git Clone)

You have two simple options to get the files onto Google Colab:

### Option A: Automatic Git Clone in Colab (Recommended - No Zipping Required!)
Since all files have been pushed to GitHub, you do **not** need to zip anything manually. You can clone directly in Colab.

### Option B: Manual Zip & Upload
If you prefer to zip and upload manually from your local machine, compress the following files/folders into `patchtst_project.zip`:
```
patchtst_project.zip
├── download_datasets.py
├── preprocess.py
├── patchtst_model.py
├── train_colab.py
├── src/
│   ├── __init__.py
│   └── patchtst/
│       ├── __init__.py
│       ├── download_datasets.py
│       ├── preprocess.py
│       ├── patchtst_model.py
│       └── train_colab.py
└── data/
    └── processed/
        ├── battery_summary.parquet      # (Optional: speeds up LFP load)
        └── engineered_features.parquet  # (Optional: speeds up LFP load)
```

---

## 2. Google Colab Terminal Commands to Execute

Open a new **Google Colab Notebook**, go to **Runtime > Change runtime type**, and select **T4 GPU** (or **A100 / L4 GPU**).

### Step 1: Clone Repository (or Unzip Uploaded Archive)
In a notebook code cell, run:

```python
# Option A: Clone directly from GitHub
!git clone https://github.com/divyansh070/ashwini_prof_project.git
%cd ashwini_prof_project
```

*(If using Option B: upload `patchtst_project.zip` using the left sidebar in Colab, then run `!unzip patchtst_project.zip` and `%cd patchtst_project`)*.

---

### Step 2: Install Required Dependencies
Ensure PyTorch, XGBoost, Pandas, PyArrow, Scikit-Learn, and SciPy are updated:

```python
!pip install -q pandas pyarrow scikit-learn scipy matplotlib xgboost torch torchvision
```

---

### Step 3: Run Multi-Source Dataset Download & Structuring (`download_datasets.py`)
This script fetches or structures the Stanford LFP, Oxford LCO, and CALCE NMC datasets into standardized Parquet files under `data/patchtst_raw/`:

```python
!python3 download_datasets.py --out-dir data/patchtst_raw
```

**What happens:**
- Loads Stanford LFP dataset ($V \in [2.05\text{ V}, 3.50\text{ V}]$).
- Generates/loads Oxford LCO dataset ($V \in [2.70\text{ V}, 4.20\text{ V}]$).
- Generates/loads CALCE NMC dataset ($V \in [2.70\text{ V}, 4.20\text{ V}]$).

---

### Step 4: Run SOD Normalization & PatchTST Patching (`preprocess.py`)
This script resolves the chemistry cross-compatibility challenge by normalizing distinct voltage plateaus into a unified **Normalized State of Discharge (SOD) grid ($u \in [0.0, 1.0]$ with $L=200$ points)**, calculates Savitzky-Golay smoothed $dQ/du$, and segments sequences into temporal-spatial patches (`patch_len=16`, `stride=8`):

```python
!python3 preprocess.py --in-dir data/patchtst_raw --out-dir data/patchtst_processed
```

**What happens:**
- Outputs 3 compressed tensor archives to `data/patchtst_processed/`:
  - `stanford_lfp_patches.npz`
  - `oxford_lco_patches.npz`
  - `calce_nmc_patches.npz`

---

### Step 5: Execute Multi-Dataset Transfer Learning (`train_colab.py`)
Run the GPU-accelerated PatchTST training and transfer learning fine-tuning loop:

```python
!python3 train_colab.py \
    --data-dir data/patchtst_processed \
    --epochs-source 100 \
    --epochs-transfer 50 \
    --batch-size 16 \
    --lr-source 5e-4 \
    --lr-transfer 1e-4
```

**What happens:**
1. **Phase 1 (Source Domain Training):** Trains the 4-layer PatchTST model on Stanford LFP from scratch and saves `checkpoints/patchtst_stanford_source.pth`.
2. **Phase 2 (Zero-Shot Evaluation):** Evaluates the frozen Stanford model directly on unseen Oxford LCO and CALCE NMC chemistries without fine-tuning.
3. **Phase 3 (Transfer Learning Fine-Tuning):**
   - Freezes all Transformer self-attention and projection layers (`model.freeze_encoder()`).
   - Fine-tunes **only the RUL Regression Head** on Oxford LCO (`checkpoints/patchtst_oxford_lco_transfer.pth`).
   - Freezes encoder and fine-tunes **only the RUL Regression Head** on CALCE NMC (`checkpoints/patchtst_calce_nmc_transfer.pth`).
4. **Phase 4 (Quantitative Report):** Prints and saves a full benchmark table to `results/transfer_learning_metrics.csv`.

---

## 3. How to Download Your Results from Colab

After training finishes, run this cell in Colab to inspect and download your results and checkpoints:

```python
import pandas as pd
from google.colab import files

# Display transfer learning comparison table
df_results = pd.read_csv("results/transfer_learning_metrics.csv")
display(df_results)

# Download CSV table and trained model weights
files.download("results/transfer_learning_metrics.csv")
files.download("checkpoints/patchtst_stanford_source.pth")
files.download("checkpoints/patchtst_oxford_lco_transfer.pth")
files.download("checkpoints/patchtst_calce_nmc_transfer.pth")
```

---

## Summary of Key Architectural Features
- **Normalized SOD Grid ($u \in [0.0, 1.0]$):** Aligns $\text{LiFePO}_4$ ($3.3\text{ V}$), $\text{LiCoO}_2$ ($3.9\text{ V}$), and $\text{LiNiMnCoO}_2$ ($3.7\text{ V}$) plateaus into an identical 200-token feature space.
- **RevIN (Reversible Instance Normalization):** Normalizes distribution shifts across cells and chemistries.
- **Encoder Freezing (`model.freeze_encoder()`):** Ensures zero catastrophic forgetting of electrochemical temporal-spatial embeddings while adapting the regression head to new chemistry lifespans.
