#!/usr/bin/env bash
# ==============================================================================
# Stage 0 Cross-Chemistry Benchmark Runner (MATR -> HUST)
#
# Tests the differential rate feature hypothesis to address model mean-reversion:
#   Arm A: Baseline 18 features (within-cycle stats, generated via preprocess_xchem.py)
#   Arm B: 30 features (18 base + 12 deltas for Q and I, dropped V deltas)
#   Arm C: 36 features (18 base + 18 deltas for V, I, Q)
#
# Fully resumable on Google Colab or local environment:
# - Preprocessing for all arms uses the SAME preprocess_xchem.py script for strict control.
# - Skips preprocessing and training if artifacts already exist (unless --force).
# - Saves per-arm JSON results immediately upon completion.
# - Automatically generates presentation figures for each arm:
#     figures/stage0_armA/
#     figures/stage0_armB/
#     figures/stage0_armC/
# - Produces final comparison table in results/stage0_comparison.md and .json.
# ==============================================================================

set -eo pipefail

NUM_RUNS=5
EPOCHS=20
EARLY_STOP_PATIENCE=0
MMD_WEIGHT=0.05
FORCE=0
RUN_VIZ=1
DATA_DIR="data/hybridonet/raw"
SEED=42

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --num-runs)
      NUM_RUNS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --early-stop-patience)
      EARLY_STOP_PATIENCE="$2"
      shift 2
      ;;
    --mmd-weight)
      MMD_WEIGHT="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --no-viz)
      RUN_VIZ=0
      shift
      ;;
    -h|--help)
      echo "Usage: bash src/cross-chemistry/run_stage0.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --num-runs N            Number of multi-seed repetitions per arm (default: 5)"
      echo "  --epochs N              Number of training epochs per run (default: 20)"
      echo "  --early-stop-patience N Stop patience; 0 disables early stopping (default: 0)"
      echo "  --mmd-weight W          MMD loss weight multiplier (default: 0.05)"
      echo "  --data-dir PATH         Root directory of parquet datasets (default: data/hybridonet/raw)"
      echo "  --seed N                Starting random seed (default: 42)"
      echo "  --force                 Force rerun all preprocessing and training from scratch"
      echo "  --no-viz                Skip generating diagnostic figures for each arm"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "======================================================================"
echo "STAGE 0: CROSS-CHEMISTRY RATE FEATURE BENCHMARK (MATR -> HUST)"
echo "Config: num_runs=${NUM_RUNS}, epochs=${EPOCHS}, patience=${EARLY_STOP_PATIENCE},"
echo "        mmd_weight=${MMD_WEIGHT}, force=${FORCE}, seed=${SEED}"
echo "======================================================================"

mkdir -p data/xchem/processed_armA
mkdir -p data/xchem/processed_armB
mkdir -p data/xchem/processed_armC
mkdir -p checkpoints
mkdir -p results
mkdir -p figures

# ------------------------------------------------------------------------------
# STEP 1: Feature Preprocessing (Resumable, Identical Preprocessor)
# ------------------------------------------------------------------------------
echo ""
echo "--- [1/3] CHECKING PREPROCESSED DATASETS ---"

# Arm A: 18-D baseline (Strictly regenerated using preprocess_xchem.py for perfect control)
if [[ -f "data/xchem/processed_armA/MATR_raw_features.npz" && -f "data/xchem/processed_armA/HUST_raw_features.npz" && $FORCE -eq 0 ]]; then
  echo "[Arm A] Found existing 18-D datasets in data/xchem/processed_armA/. Skipping."
else
  echo "[Arm A] Preprocessing baseline 18-D features (no deltas, using preprocess_xchem.py)..."
  python3 src/cross-chemistry/preprocess_xchem.py \
    --data-dir "${DATA_DIR}" \
    --output-dir data/xchem/processed_armA \
    --stride 10 \
    --no-use-deltas
fi

# Arm B: 30-D with Q, I deltas (dropped V delta)
if [[ -f "data/xchem/processed_armB/MATR_raw_features.npz" && -f "data/xchem/processed_armB/HUST_raw_features.npz" && $FORCE -eq 0 ]]; then
  echo "[Arm B] Found existing 30-D datasets in data/xchem/processed_armB/. Skipping."
else
  echo "[Arm B] Preprocessing 30-D features (Q, I rate deltas)..."
  python3 src/cross-chemistry/preprocess_xchem.py \
    --data-dir "${DATA_DIR}" \
    --output-dir data/xchem/processed_armB \
    --stride 10 \
    --use-deltas \
    --delta-signals "Q,I"
fi

# Arm C: 36-D with all deltas (V, I, Q)
if [[ -f "data/xchem/processed_armC/MATR_raw_features.npz" && -f "data/xchem/processed_armC/HUST_raw_features.npz" && $FORCE -eq 0 ]]; then
  echo "[Arm C] Found existing 36-D datasets in data/xchem/processed_armC/. Skipping."
else
  echo "[Arm C] Preprocessing 36-D features (V, I, Q all deltas)..."
  python3 src/cross-chemistry/preprocess_xchem.py \
    --data-dir "${DATA_DIR}" \
    --output-dir data/xchem/processed_armC \
    --stride 10 \
    --use-deltas \
    --delta-signals "V,I,Q"
fi

# ------------------------------------------------------------------------------
# STEP 2: Model Training & Evaluation (Resumable per Arm)
# ------------------------------------------------------------------------------
echo ""
echo "--- [2/3] RUNNING STAGE 0 BENCHMARK ARMS ---"

# ARM A: 18-D Baseline
if [[ -f "results/stage0_armA.json" && -f "checkpoints/stage0_armA.pt" && $FORCE -eq 0 ]]; then
  echo "[Arm A] Found existing results in results/stage0_armA.json. Skipping training."
else
  echo ""
  echo ">>> [ARM A] Training 18-D Baseline (${NUM_RUNS} runs, seed=${SEED}) <<<"
  python3 src/cross-chemistry/train_xchem.py \
    --source data/xchem/processed_armA/MATR_raw_features.npz \
    --target data/xchem/processed_armA/HUST_raw_features.npz \
    --epochs "${EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --mmd-weight "${MMD_WEIGHT}" \
    --num-runs "${NUM_RUNS}" \
    --seed "${SEED}" \
    --norm-type layernorm \
    --rul-ceiling 2500 \
    --severson-only \
    --checkpoint-path checkpoints/stage0_armA.pt \
    --results-json results/stage0_armA.json
fi

# ARM B: 30-D Rate Features (Q, I deltas)
if [[ -f "results/stage0_armB.json" && -f "checkpoints/stage0_armB.pt" && $FORCE -eq 0 ]]; then
  echo "[Arm B] Found existing results in results/stage0_armB.json. Skipping training."
else
  echo ""
  echo ">>> [ARM B] Training 30-D Q,I Rate Model (${NUM_RUNS} runs, seed=${SEED}) <<<"
  python3 src/cross-chemistry/train_xchem.py \
    --source data/xchem/processed_armB/MATR_raw_features.npz \
    --target data/xchem/processed_armB/HUST_raw_features.npz \
    --epochs "${EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --mmd-weight "${MMD_WEIGHT}" \
    --num-runs "${NUM_RUNS}" \
    --seed "${SEED}" \
    --norm-type layernorm \
    --rul-ceiling 2500 \
    --severson-only \
    --checkpoint-path checkpoints/stage0_armB.pt \
    --results-json results/stage0_armB.json
fi

# ARM C: 36-D All Rate Features (V, I, Q deltas)
if [[ -f "results/stage0_armC.json" && -f "checkpoints/stage0_armC.pt" && $FORCE -eq 0 ]]; then
  echo "[Arm C] Found existing results in results/stage0_armC.json. Skipping training."
else
  echo ""
  echo ">>> [ARM C] Training 36-D All-Deltas Model (${NUM_RUNS} runs, seed=${SEED}) <<<"
  python3 src/cross-chemistry/train_xchem.py \
    --source data/xchem/processed_armC/MATR_raw_features.npz \
    --target data/xchem/processed_armC/HUST_raw_features.npz \
    --epochs "${EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --mmd-weight "${MMD_WEIGHT}" \
    --num-runs "${NUM_RUNS}" \
    --seed "${SEED}" \
    --norm-type layernorm \
    --rul-ceiling 2500 \
    --severson-only \
    --checkpoint-path checkpoints/stage0_armC.pt \
    --results-json results/stage0_armC.json
fi

# ------------------------------------------------------------------------------
# STEP 3: Generate Visualizations & Summary Comparison
# ------------------------------------------------------------------------------
if [[ $RUN_VIZ -eq 1 ]]; then
  echo ""
  echo "--- [3/3] GENERATING DIAGNOSTIC FIGURES ---"

  mkdir -p figures/stage0_armA
  mkdir -p figures/stage0_armB
  mkdir -p figures/stage0_armC

  if [[ -f "checkpoints/stage0_armA.pt" ]]; then
    echo "[Visualizer] Generating Arm A figures into figures/stage0_armA/ ..."
    python3 src/cross-chemistry/visualize_xchem.py \
      --checkpoint checkpoints/stage0_armA.pt \
      --source data/xchem/processed_armA/MATR_raw_features.npz \
      --target data/xchem/processed_armA/HUST_raw_features.npz \
      --severson-only \
      --out-dir figures/stage0_armA
  fi

  if [[ -f "checkpoints/stage0_armB.pt" ]]; then
    echo "[Visualizer] Generating Arm B figures into figures/stage0_armB/ ..."
    python3 src/cross-chemistry/visualize_xchem.py \
      --checkpoint checkpoints/stage0_armB.pt \
      --source data/xchem/processed_armB/MATR_raw_features.npz \
      --target data/xchem/processed_armB/HUST_raw_features.npz \
      --severson-only \
      --out-dir figures/stage0_armB
  fi

  if [[ -f "checkpoints/stage0_armC.pt" ]]; then
    echo "[Visualizer] Generating Arm C figures into figures/stage0_armC/ ..."
    python3 src/cross-chemistry/visualize_xchem.py \
      --checkpoint checkpoints/stage0_armC.pt \
      --source data/xchem/processed_armC/MATR_raw_features.npz \
      --target data/xchem/processed_armC/HUST_raw_features.npz \
      --severson-only \
      --out-dir figures/stage0_armC
  fi
fi

# Aggregate and print comparison table
python3 src/cross-chemistry/stage0_summary.py --results-dir results --out-dir results

echo "Stage 0 run complete. Inspect results/stage0_comparison.md and figures/stage0_*/"
