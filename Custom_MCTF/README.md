# Custom MCTF Implementation & Experiments

This directory contains our clean, modular reimplementation of the CVPR 2024 paper *"Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers"*.

## Architecture Modules
* **`mctf_core.py`**: Pure PyTorch implementation of Multi-Criteria Scoring (Similarity + Informativeness + Size) and Bipartite Soft Matching.
* **`models.py`**: `MCTFVisionTransformer` wrapper that intercepts forward attention passes, computes pre-fusion attention $A^{(l+1)}$, aggregates it to $A_n$, and bypasses redundant $Q \times K^T$ matrix multiplications in subsequent transformer blocks.
* **`training_history.json`**: Cached metric trajectories allowing instant plot re-rendering without retraining.

---

## Experiments

### 1. EuroSAT Experiment (`train_experiments.ipynb`)
Our primary experimental benchmark evaluates Baseline DeiT-Tiny vs. Custom MCTF-ViT on the **EuroSAT** satellite imagery dataset (10 classes, 27,000 images):
* Baseline DeiT-Tiny: **97.78% Peak Test Acc** (1.258 GFLOPs)
* Custom MCTF-ViT: **97.57% Peak Test Acc** (1.175 GFLOPs)
* **Verdict:** Preserved 99.8% of model accuracy while reducing token sequence length from 197 down to 107 tokens.

### 2. Multi-Dataset Collaborative Study
To fulfill the multi-dataset requirement across our group:
* **EuroSAT:** Evaluated in this repository (`train_experiments.ipynb`).
* **CIFAR-100:** Evaluated and contributed by teammate.
