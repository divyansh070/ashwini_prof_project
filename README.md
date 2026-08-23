# Battery Life Prediction via Physics-Informed Domain Adaptation

This repository contains the codebase for predicting lithium-ion battery End of Life (EOL) using a physics-informed **Koopman Neural Operator** combined with a **Domain-Adversarial Neural Network (DANN)**. The goal is to train a model on one battery chemistry (e.g., LFP) and perfectly transfer its predictions to completely unseen chemistries (e.g., NMC, LCO).

**Crucial Note:** This project relies *exclusively* on genuine, physical sensor data. **Zero synthetic data interpolation** (e.g., `np.linspace`) is permitted anywhere in the data pipelines. 

---

## The Generalization Crisis
Standard machine learning models trained on Lithium Iron Phosphate (LFP) batteries fail catastrophically when evaluated on Nickel Manganese Cobalt (NMC) batteries due to fundamental differences in absolute voltage bounds and thermodynamic physics. 

We solve this using:
1. **Universal Physics Preprocessing:** We convert all raw voltage/capacity curves into State of Charge (SOC)-normalized differential capacity ($dQ/dSOC$) embeddings.
2. **Koopman Operator:** We mathematically force the neural network's latent space to evolve linearly over time and enforce thermodynamic monotonicity (batteries cannot heal).
3. **Domain-Adversarial Transfer (DANN):** Using a Gradient Reversal Layer (GRL), we actively blind the encoder to the specific battery chemistry, forcing the features of LFP and NMC batteries to statistically align in the latent space.

---

## The 5 Genuine Datasets
We evaluate our zero-shot transfer across five massive, open-source datasets containing distinct cell chemistries:

1. **Stanford/MIT (LFP)**: 124 cells (Source Domain)
2. **CALCE (NMC/LCO)**: 13 cells (Target Domain)
3. **HUST (LFP)**: 77 cells (Target Domain)
4. **SNL (LFP/NCA/NMC)**: 145 cells (Target Domain)
5. **RWTH (NMC)**: 48 cells (Target Domain)

---

## How to Run the Pipeline (Colab Instructions)

All modeling code is intuitively organized within the `src/` directory.

### 1. Acquire & Preprocess the Data
First, download the raw data from Zenodo and HuggingFace, and execute the physics preprocessing.
```bash
# 1. Download Source Data
python src/downloads/download_real_data.py

# 2. Download Target Data
python src/downloads/download_batterylife_data.py

# 3. Extract Thermodynamics (dQ/dSOC)
python src/preprocess/preprocess_real_data.py
```

### 2. The Forensic Data Leakage Audit
Before training any ML model, you must prove there is zero leakage of future capacity data. The following script will scan all parquet files and assert that no features extracted between Cycle 10 and 100 contain data from Cycle 101+.
```bash
python src/audit/leakage_audit.py
```

### 3. Train the DANN Pipeline
Execute the main Colab script. This script will:
1. Train the Koopman network on the **Stanford LFP** dataset using a strict 5-Fold GroupKFold CV.
2. Iterate through the four target datasets (CALCE, HUST, SNL, RWTH).
3. Evaluate Zero-Shot performance.
4. Execute DANN adversarial adaptation to align the physics of the Source and Target datasets.
```bash
python src/training/train_da_colab.py --epochs-source 50 --epochs-dann 30
```

Results are saved to `results/domain_adversarial_metrics.csv` and PyTorch model checkpoints are saved to the `checkpoints/` directory.
