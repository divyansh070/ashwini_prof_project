# Stage 0: Cross-Chemistry Extension (Differential Rate Features)

## Background & Diagnosis

Our diagnostic evaluation of the HybridoNet-Adapt baseline on MATR $\rightarrow$ HUST revealed a systematic failure mode: **the model regresses to the mean**.
- On blind test cells, every predicted RUL trajectory starts in a narrow 1400–2000 cycle band regardless of the cell's true lifespan (true lifespans range 950–2280 cycles).
- Per-cell RMSE simply tracks how close a cell's true life is to the default guess (85 cycles for the closest cells, 428 cycles for the furthest).
- The error-bias curve independently confirms this: flat near zero below RUL 1000, and dropping to $-300$ cycles above RUL 1000.

**Physical Root Cause:**
All 18 baseline features are *within-cycle* statistics (mean, std, min, max, variance, median of $V, I, Q$). The network observes the battery's instantaneous state but never its **degradation rate** ($\Delta Q/\Delta t$), which is the primary physical quantity distinguishing a short-lived cell from a long-lived cell. Furthermore, absolute current features ($I$) suffered scaling collapse across domains due to disparate charging protocol definitions, whereas differential rate features are scale-relative and protocol-invariant (Zhu et al., 2025, *Energy*).

---

## Hypothesis & Arm Structure

**Stage 0 Hypothesis:** Appending window-relative cross-cycle differential features ($\Delta = \text{cell\_tensor} - \text{cell\_tensor}[0:1, :, :]$) provides explicit degradation rate information and scale-relative invariance, fixing mean-reversion without changing the core neural architecture.

We evaluate three controlled experimental arms with identical seeds (42..46) and cell partitions:

| Arm | Features | Delta Formulation | Notes |
| :--- | :--- | :--- | :--- |
| **Arm A** | **18-D** | None | Validated baseline ($R^2 \approx 0.863$, RMSE $\approx 197.5$ cyc) |
| **Arm B** | **30-D** | $\Delta Q, \Delta I$ (12 rate features) | Drops unselected $V$ deltas ($V$ features contributed $\approx 0$ in permutation test) |
| **Arm C** | **36-D** | $\Delta V, \Delta I, \Delta Q$ (18 rate features) | Full rate tensor ablation |

---

## Pass Criteria

Stage 0 succeeds if, compared to the 18-feature baseline:
1. **Quantitative**: Mean Test $R^2 > 0.863$ over 5 seeds, and the improvement exceeds the seed standard deviation ($R^2_{\text{arm}} - R^2_{\text{base}} > \sigma_{\text{base}}$).
2. **Qualitative (Trajectory Spread)**: In `figures/stage0_<arm>/A_trajectories.png`, predicted initial values spread with true lifespan (950–2280 cycles) instead of clustering at 1400–2000 cycles.
3. **Qualitative (Error Flattening)**: In `figures/stage0_<arm>/F_error_anatomy.png`, the bias curve flattens above RUL 1000 cycles.

---

## Google Colab Quickstart (Copy-Pasteable Cell Block)

Run the following cell in your Google Colab notebook (working directory: `/content/ashwini_prof_project`):

```python
# ==============================================================================
# COLAB STAGE 0 EXECUTION CELL
# ==============================================================================
import os

# 1. Navigate to repo root and pull latest commits
%cd /content/ashwini_prof_project
!git pull origin main

# 2. Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# 3. Verify data availability
data_raw = "data/hybridonet/raw"
if not os.path.exists(data_raw) or not any(os.scandir(data_raw)):
    # Fallback to mounted Google Drive if present
    drive_data = "/content/drive/MyDrive/battery_project_data/raw"
    if os.path.exists(drive_data):
        print(f"Linking Drive data: {drive_data} -> {data_raw}")
        os.makedirs("data/hybridonet", exist_ok=True)
        !ln -sfn "{drive_data}" "{data_raw}"

# 4. Run Stage 0 Benchmark (5 runs per arm, fully resumable)
# Anchored to validated baseline: 20 epochs, early-stop-patience 0, mmd-weight 0.05
# Expected wall-clock time on T4 GPU: ~8-12 mins total
!bash src/cross-chemistry/run_stage0.sh --num-runs 5 --epochs 20

# 5. Display comparison results
with open("results/stage0_comparison.md") as f:
    print(f.read())
```

---

## Expected Wall-Clock Times

| Stage / Component | Colab T4 GPU | Local CPU |
| :--- | :--- | :--- |
| **Preprocessing** (Arms A, B, C) | ~1–2 min per arm | ~2–3 min per arm |
| **Training** (20 epochs × 5 runs) | ~2–3 min per arm | ~6–8 min per arm |
| **Visualization** (Figures A–F) | ~30 sec per arm | ~45 sec per arm |
| **Total Benchmark** (3 arms) | **~8–12 minutes** | **~25–35 minutes** |

*Note on Resumability:* If a Colab session disconnects during training, simply rerun `bash src/cross-chemistry/run_stage0.sh`. Completed `.npz` files and arm checkpoints (`checkpoints/stage0_arm*.pt`) will be detected and skipped automatically. Pass `--force` to re-run from scratch.

---

## Standalone CLI Commands

```bash
# Feature preprocessing with custom delta signals
python3 src/cross-chemistry/preprocess_xchem.py \
  --data-dir data/hybridonet/raw \
  --output-dir data/xchem/processed_armB \
  --stride 10 \
  --use-deltas \
  --delta-signals Q,I

# Train individual model session
python3 src/cross-chemistry/train_xchem.py \
  --source data/xchem/processed_armB/MATR_raw_features.npz \
  --target data/xchem/processed_armB/HUST_raw_features.npz \
  --epochs 60 \
  --num-runs 5 \
  --norm-type layernorm \
  --rul-ceiling 2500 \
  --severson-only \
  --checkpoint-path checkpoints/stage0_armB.pt \
  --results-json results/stage0_armB.json

# Generate presentation figures
python3 src/cross-chemistry/visualize_xchem.py \
  --checkpoint checkpoints/stage0_armB.pt \
  --source data/xchem/processed_armB/MATR_raw_features.npz \
  --target data/xchem/processed_armB/HUST_raw_features.npz \
  --severson-only \
  --out-dir figures/stage0_armB
```
