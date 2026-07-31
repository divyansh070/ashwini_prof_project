"""
PatchTST Multi-Dataset Transfer Learning Package for Battery RUL Estimation.
Implements:
  - Multi-source data acquisition (Stanford LFP, Oxford LCO, CALCE NMC)
  - Normalized SOD grid patching & Savitzky-Golay dQ/du preprocessing
  - Patch Time Series Transformer (PatchTST) PyTorch model
  - Source training & frozen-encoder transfer learning fine-tuning
"""
