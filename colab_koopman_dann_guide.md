# Google Colab Execution Guide: Koopman Neural Operator (KNO) & Domain Adversarial Transfer Learning (DANN)

This authoritative guide provides exact, step-by-step instructions to upload, normalize, and execute the **Koopman Neural Operator (KNO)** combined with **Domain-Adversarial Neural Networks (DANN)** across **Stanford/MIT (LFP)**, **Oxford (LCO)**, and **CALCE (NMC)** datasets on Google Colab GPU.

---

## Why This Architecture Solves Mode Collapse
1. **Universal Domain Normalization (`preprocess_v2.py`):**
   * Normalizes the x-axis from Absolute Voltage to **State of Charge (SOC) $s \in [0.0, 1.0]$** ($L=200$ uniform bins).
   * Computes **$dQ/d(\text{SOC})$** so that invariant electrochemical phase-transition staging peaks align across $\text{LiFePO}_4$, $\text{LiCoO}_2$, and $\text{LiNiMnCoO}_2$ chemistries.
2. **Koopman Neural Operator (`koopman_model.py`):**
   * Embeds non-linear capacity fade curves into an invariant linear subspace where temporal dynamics evolve linearly ($\mathbf{z}_{k+1} = \mathbf{K} \mathbf{z}_k$).
   * A **Physics-Informed Koopman Linearity Loss** ($\mathcal{L}_{\text{KNO}}$) regularizes the latent space to prevent Transformer mode collapse.
3. **Explicit Domain Adaptation via DANN (`train_da_colab.py`):**
   * Employs a **Gradient Reversal Layer (GRL)** and a Domain Discriminator to align the latent probability distributions of Stanford (Source) and Oxford/CALCE (Target) domains during transfer learning.

---

## 1. How to Load the Code on Google Colab

You have two options to load the files onto a Colab GPU instance:

### Option A: Automatic Git Clone (Recommended - No Zip Required!)
Since all files are pushed to GitHub, you can clone directly inside a Colab notebook cell.

### Option B: Manual Zip & Upload
If working offline, compress the following files into `koopman_dann_project.zip`:
```
koopman_dann_project.zip
├── download_datasets.py
├── preprocess_v2.py
├── koopman_model.py
├── train_da_colab.py
├── src/
│   ├── __init__.py
│   └── koopman/
│       ├── __init__.py
│       ├── preprocess_v2.py
│       ├── koopman_model.py
│       └── train_da_colab.py
└── data/
    └── patchtst_raw/       # (Optional: previously downloaded parquet files)
```

---

## 2. Step-by-Step Google Colab Commands

Open a new **Google Colab Notebook**, click **Runtime > Change runtime type**, and select **T4 GPU** (or **A100 / L4 GPU**).

### Step 1: Clone Repository (or Unzip Archive)
Run in the first code cell:
```python
# Option A: Clone from GitHub
!git clone https://github.com/divyansh070/ashwini_prof_project.git
%cd ashwini_prof_project
```
*(If using Option B: upload `koopman_dann_project.zip` via the left sidebar and run `!unzip koopman_dann_project.zip && %cd koopman_dann_project`)*.

---

### Step 2: Install PyTorch & Dependencies
```python
!pip install -q pandas pyarrow scikit-learn scipy matplotlib xgboost torch torchvision
```

---

### Step 3: Ensure Multi-Source Datasets Are Present (`download_datasets.py`)
If you have not already fetched the Stanford LFP, Oxford LCO, and CALCE NMC parquet tables, run:
```python
!python3 download_datasets.py --out-dir data/patchtst_raw
```

---

### Step 4: Run Universal Domain Normalization (`preprocess_v2.py`)
Transform Absolute Voltage into **State of Charge (SOC) in [0, 1]** and calculate aligned **$dQ/d(\text{SOC})$** matrices across all three chemistries:
```python
!python3 preprocess_v2.py --in-dir data/patchtst_raw --out-dir data/koopman_processed
```
**Output:** Saves 3 universal SOC-normalized archives:
- `data/koopman_processed/stanford_lfp_soc.npz`
- `data/koopman_processed/oxford_lco_soc.npz`
- `data/koopman_processed/calce_nmc_soc.npz`

---

### Step 5: Execute Koopman DANN Domain-Adversarial Transfer Learning (`train_da_colab.py`)
Run the GPU-accelerated Koopman Neural Operator training and Domain-Adversarial adaptation loop:
```python
!python3 train_da_colab.py \
    --data-dir data/koopman_processed \
    --epochs-source 100 \
    --epochs-dann 60 \
    --batch-size 16 \
    --lr-source 5e-4 \
    --lr-dann 2e-4 \
    --lambda-koopman 0.10 \
    --lambda-dann 0.50
```

**What Happens:**
1. **Phase 1 (Source Domain Training):** Trains the Koopman Neural Operator on Stanford LFP with Physics-Informed linearity regularized by $\lambda_{\text{Koopman}} = 0.10$, saving `checkpoints/koopman_dann_stanford_source.pth`.
2. **Phase 2 (Zero-Shot Evaluation):** Tests baseline generalization of the source Koopman operator on Oxford LCO and CALCE NMC.
3. **Phase 3 (DANN Adversarial Adaptation):**
   - Employs a Gradient Reversal Layer (GRL) with dynamic adaptation weight $\alpha$.
   - Simultaneously minimizes RUL prediction MSE and Koopman linearity error while **maximizing Domain Discriminator confusion** ($\lambda_{\text{DANN}} = 0.50$).
   - Saves `checkpoints/koopman_dann_oxford_lco_transfer.pth` and `checkpoints/koopman_dann_calce_nmc_transfer.pth`.
4. **Phase 4 (Quantitative Report):** Prints and exports a comparison table to `results/domain_adversarial_metrics.csv`.

---

### Step 6: Inspect Benchmark Table & Download Checkpoints
Run in your final code cell:
```python
import pandas as pd
from google.colab import files

# Display Domain Adversarial benchmark table
df_results = pd.read_csv("results/domain_adversarial_metrics.csv")
display(df_results)

# Download CSV report & DANN checkpoints
files.download("results/domain_adversarial_metrics.csv")
files.download("checkpoints/koopman_dann_stanford_source.pth")
files.download("checkpoints/koopman_dann_oxford_lco_transfer.pth")
files.download("checkpoints/koopman_dann_calce_nmc_transfer.pth")
```
