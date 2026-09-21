# HybridoNet-Adapt — Paper Replication

Reproduces the Table 2 "All" / Table 3 setting of Tran et al. (2025), arXiv:2503.21392v2.
Self-contained: [`model_paper.py`](model_paper.py) + [`train_paper.py`](train_paper.py) located in `src/hybridonet-adapt/`.

---

## Run on Google Colab

### Step 1: Clone or Pull Repo & Mount Google Drive
```python
from google.colab import drive
import os

drive.mount('/content/drive')

# Clone if not already present, or pull latest main
if not os.path.exists('/content/ashwini_prof_project'):
    !git clone https://github.com/divyansh070/ashwini_prof_project.git /content/ashwini_prof_project
%cd /content/ashwini_prof_project
!git pull origin main
```

### Step 2: Ensure Processed Data is Available
Ensure your processed NPZ files are linked or copied to `data/hybridonet/processed/`:
```bash
mkdir -p data/hybridonet/processed

# Example: If your data is stored in your Google Drive
# cp /content/drive/MyDrive/battery_data/*.npz data/hybridonet/processed/

# Verify that both source (MATR/TRI) and target (HUST/LHP) files exist:
ls -lh data/hybridonet/processed/
```

### Step 3: Fast Smoke Run (1 Run, 1 Epoch ~ 30 seconds)
Verify GPU setup, cell parsing, and data flow:
```bash
python3 src/hybridonet-adapt/train_paper.py \
    --source data/hybridonet/processed/MATR_raw_features.npz \
    --target data/hybridonet/processed/HUST_raw_features.npz \
    --epochs 1 \
    --num-runs 1
```

### Step 4: Quick Check for Under-Training (~2-3 minutes)
The model was previously undertrained because stride 10 yields only 20 gradient steps per epoch (200 steps total).
Train for 100 epochs to give ~2,000 gradient steps and verify if RMSE drops toward 153 cycles:
```bash
# 1-run quick check:
python3 src/hybridonet-adapt/train_paper.py \
    --source data/hybridonet/processed/MATR_raw_features.npz \
    --target data/hybridonet/processed/HUST_raw_features.npz \
    --epochs 100 \
    --num-runs 1 \
    --log-interval 10

# Full 10-run ensemble with 100 epochs:
python3 src/hybridonet-adapt/train_paper.py \
    --source data/hybridonet/processed/MATR_raw_features.npz \
    --target data/hybridonet/processed/HUST_raw_features.npz \
    --epochs 100 \
    --num-runs 10 \
    --log-interval 10 \
    --checkpoint-dir checkpoints/paper_ep100 \
    --results-json results/paper_replication_ep100.json
```

### Step 5: Paper-Faithful Dense Windows (Stride 1)
The paper plots dense predictions along cell life (Fig. 9), indicating **stride 1** (~26,800 source windows, giving ~2,090 gradient steps under the paper's original 10 epochs):
```bash
# 1. Preprocess with stride 1 into a separate folder:
python3 src/hybridonet-adapt/preprocess_hybrido.py \
    --stride 1 \
    --output-dir data/hybridonet/processed_stride1

# 2. Train with default 10 epochs on stride 1 data:
python3 src/hybridonet-adapt/train_paper.py \
    --source data/hybridonet/processed_stride1/MATR_raw_features.npz \
    --target data/hybridonet/processed_stride1/HUST_raw_features.npz \
    --epochs 10 \
    --num-runs 10 \
    --checkpoint-dir checkpoints/paper_stride1 \
    --results-json results/paper_replication_stride1.json
```

All defaults match the paper:
- 10 independent runs averaged into an ensemble
- 10 epochs, batch 128
- AdamW at fixed LR 5e-4
- Hidden dim 64, 2-layer LSTM
- Multi-head attention (second-to-last timestep)
- RK4 Neural ODE
- BatchNorm predictors
- Free unconstrained $\theta$ scalars
- Source head loss on source head alone ($Y_S$)

---

## Log Checklist (Verify During Execution)

| Check | Expected Log Output |
|---|---|
| Device | `[Paper] device: cuda` |
| Target Split | `target: 22 paper test cells, 55 training cells (paper: 22 / 55)` |
| Source Split | `source: 41 cells via 'severson-train' (paper: 41)` |
| Scaling | No `outside [-0.5, 1.5]` warnings |
| Prediction Spread | `val pred [min, max]` shows realistic spread (not collapsed `min ≈ max`) |

---

## Metric Comparison

The paper's headline numbers (**RMSE 153.24, R² 0.88, MAPE 7.30%**) are the **mean of per-cell metrics over the 22 test channels** (Table 3 "Mean" row), not pooled over every single window.
The script automatically prints both:
- **Paper Macro** (`mean of per-cell`): Direct comparison against paper Table 2 & Table 3.
- **Pooled** (`all windows`): Standard data-science pooling across all time windows.
- **Table 3 side-by-side comparison**: Per-channel breakdown against the paper's reported values.

---

## Source Cells Configuration

`severson-train` programmatically rebuilds Severson et al.'s training split (odd indices across batches 1 & 2). If your MATR mirror has a different cell ordering or prunes cells, you can provide an explicit list of cell IDs:

```bash
python3 src/hybridonet-adapt/train_paper.py \
    --source data/hybridonet/processed/MATR_raw_features.npz \
    --target data/hybridonet/processed/HUST_raw_features.npz \
    --source-cells file \
    --source-cells-file tri_train_cells.txt
```

---

## Architecture & Hyperparameter Flags (Unspecified in Paper)

| Hyperparameter / Choice | Default | Alternative Flags |
|---|---|---|
| Predictor Normalization | `batchnorm` | `--predictor-norm layernorm`, `--predictor-norm none` |
| NODE Output Concatenation | `concat` (h(0), h(1) $\to$ 128) | `--node-output last` (h(1) $\to$ 64) |
| Validation Domain Split | `target` (10% of target adapt) | `--val-domain source` (10% of source train) |
| MMD Bandwidth $\sigma$ | Median heuristic ($\le 0$) | `--mmd-sigma 1.0` (fixed bandwidth) |
| AdamW Weight Decay | `0.01` (PyTorch default) | `--weight-decay 1e-4` |
| Attention Heads | `4` | `--num-heads 8` |
| Residual & LayerNorm | Off | `--attn-residual-norm`, `--post-node-norm` |
| Trade-off Mode | `free` (unconstrained) | `--theta-mode softmax` |

