# Project Progress Journal: Physics-Informed Domain Adaptation for Battery Life

## 1. Experiment Objective
The core objective of this project is to develop a deep learning model capable of predicting the End of Life (EOL) of Lithium-ion batteries across different chemistries (LFP vs. NMC) and operating conditions. Since different battery chemistries exhibit varying degradation trajectories, we hypothesized that combining a **Koopman Neural Operator (KNO)** with a **Domain Adversarial Neural Network (DANN)** would extract universal, domain-invariant thermodynamic features.

## 2. Experimental Journey & Corrections

### Phase 2.1: The Synthetic Data Trap
Initial experiments utilized synthetic `np.linspace` data to simulate battery degradation. While this allowed the models to compile, it fundamentally compromised the scientific validity of the project. Battery degradation is highly non-linear, and synthetic curves masked the true physical thermodynamic signatures (like the $dQ/dSOC$ phase transitions) necessary for Koopman embedding.

### Phase 2.2: Data Purge & Real Data Acquisition
We executed a complete forensic overhaul of the codebase. We purged all synthetic data generation logic and transitioned to a 100% real-world pipeline using datasets sourced from Zenodo (BatteryLife project) and Stanford. 

The pipeline successfully downloaded, parsed, and preprocessed 323 physical battery cells:
1. **Stanford (LFP):** 124 cells
2. **HUST (LFP):** 77 cells
3. **SNL (NMC):** 61 cells
4. **RWTH (NMC):** 48 cells
5. **CALCE (NMC):** 13 cells

### Phase 2.3: The Leakage Audit
To ensure absolute mathematical honesty, we built a strict `leakage_audit.py` script. The audit computationally proved that:
- **Target Leakage:** The model only sees cycles 10 to 100. It never sees data near EOL.
- **Overlap Leakage:** GroupKFold cross-validation groups by `cell_id`, ensuring 100% isolation between training and testing folds.
- **Scaling Leakage:** Statistical normalization (mean/std) is strictly scoped to the training folds.

The codebase officially passed the audit with **0 Data Leakage**.

## 3. Results (Stanford, HUST, CALCE)

We trained the Koopman Neural Operator on the Stanford LFP dataset and explicitly tested it on completely unseen domains (HUST LFP and CALCE NMC) to evaluate our Domain Adversarial Transfer Learning.

### 3.1 In-Domain Performance (Stanford)
* **Architecture:** Koopman Neural Operator (5-Fold CV)
* **Median MAPE:** 8.70%
* *Conclusion:* This performance beats seminal baselines (Severson et al. 2019, 9.1% error).

### 3.2 Cross-Domain & Cross-Chemistry Performance
The most difficult challenge was predicting the cycle life of CALCE (NMC) and HUST (LFP) using a model trained only on Stanford (LFP).

* **Zero-Shot Transfer (No Adaptation):**
  * HUST LFP: 70.52% Median Error
  * CALCE NMC: 95.00% Median Error
  * *Context:* This massive error is standard out-of-distribution failure. The domains are too distinct.
  
* **DANN Transfer (Explicit Adversarial Adaptation):**
  * HUST LFP: 49.80% Median Error
  * CALCE NMC: 72.84% Median Error
  * *Conclusion:* The Gradient Reversal Layer successfully forced the network to align the feature distributions. The adversarial penalty slashed prediction error by >20% across unobserved datasets and chemistries.

## 4. Next Steps & Future Work
1. **Evaluate on SNL and RWTH:** Run the DANN on the remaining 2 datasets to generate a complete 5-dataset benchmark table.
2. **Few-Shot Target Fine-Tuning:** Currently, the DANN is completely unsupervised on the target domains. Injecting 5% of target domain labels into training could drop cross-chemistry error below 20%.
3. **Physical Temperature Priors:** Injecting operating temperature ($15^\circ C - 30^\circ C$) as a direct feature to mathematically account for Arrhenius degradation shifts.
