# Custom MCTF Implementation & Experiments

This directory contains our clean, modular reimplementation of the CVPR 2024 paper *"Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers"*.

## Architecture Modules
* **`mctf_core.py`**: Pure PyTorch implementation of Multi-Criteria Scoring (Similarity + Informativeness + Size) and Bipartite Soft Matching.
* **`models.py`**: `MCTFVisionTransformer` wrapper that intercepts forward attention passes, computes pre-fusion attention $A^{(l+1)}$, aggregates it to $A_n$, and bypasses redundant $Q \times K^T$ matrix multiplications in subsequent transformer blocks.
* **`training_history.json`**: Cached metric trajectories allowing instant plot re-rendering without retraining.

---

## Experiments & Multi-Dataset Benchmarks

### 1. EuroSAT Land-Use Classification (`EuroSAT_nb.ipynb`)
Our primary experimental benchmark evaluates Baseline DeiT-Tiny vs. Custom MCTF-ViT on the **EuroSAT** satellite imagery dataset (10 classes, 27,000 images):
* **Baseline DeiT-Tiny:** 97.78% Peak Test Acc (1.258 GFLOPs, 197 tokens)
* **Custom MCTF-ViT:** 97.57% Peak Test Acc (1.175 GFLOPs, 107 tokens)
* **Verdict:** Preserved 99.8% of model accuracy ($\Delta = -0.21\%$) while reducing sequence length by 45.7% and saving 6.62% GFLOPs.

### 2. CIFAR-100 Fine-Grained Classification (`cifar10_nb.ipynb`)
Evaluates the model under dense 100-class label complexity:
* **Baseline DeiT-Tiny:** 78.88% Peak Val Acc | **78.72% Final Test Acc** (Early stopped at epoch 15)
* **Custom MCTF-ViT:** 75.98% Peak Val Acc | **76.52% Final Test Acc** (Early stopped at epoch 12)
* **Verdict:** Robust convergence within 2.2% of baseline while reducing quadratic self-attention FLOPs.

### 3. SVHN Street View House Numbers (`SVHN_nb.ipynb`)
Evaluates token fusion behavior on cropped house digits (10 classes):
* **Baseline DeiT-Tiny:** 95.71% Peak Val Acc | **96.25% Final Test Acc** (Early stopped at epoch 13)
* **Custom MCTF-ViT:** 95.10% Peak Val Acc | **95.56% Final Test Acc** (Early stopped at epoch 19)
* **Verdict:** Preserved 99.3% of baseline performance ($\Delta = -0.69\%$), showing informativeness scores successfully protect sharp digit edges.

