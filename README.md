# Multi-Criteria Token Fusion (MCTF) for Vision Transformers
### Paper Reproduction & Custom Modular Implementation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![timm](https://img.shields.io/badge/timm-0.5.4+-brightgreen.svg)](https://github.com/huggingface/pytorch-image-models)
[![CVPR 2024](https://img.shields.io/badge/CVPR-2024-blue.svg)](https://openaccess.thecvf.com/content/CVPR2024/html/Lee_Multi-criteria_Token_Fusion_with_One-step-ahead_Attention_for_Efficient_Vision_Transformers_CVPR_2024_paper.html)

This repository contains:
1. **Official Paper Reproduction:** Full evaluation of the CVPR 2024 paper *"Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers"* on the official **ImageNet-1K (ILSVRC2012)** validation set.
2. **Custom Modular Implementation:** A standalone, non-invasive PyTorch implementation of the MCTF mathematical framework evaluated on downstream benchmarks (**EuroSAT**, **SVHN**, and **CIFAR-100**).

---

## 📁 Repository Structure

```text
├── README.md                      # Master documentation (Reproduction + Custom Implementation)
├── requirements.txt               # Unified project dependencies
├── .gitignore                     # Excludes heavy datasets and weights
│
├── MCTF/                          # PART 1: Official Paper Reproduction
│   ├── eval_log_deit_s.txt        # Full ImageNet evaluation log for DeiT-Small (79.94% Top-1 / 2.60 GFLOPs)
│   ├── eval_log_deit_t.txt        # Full ImageNet evaluation log for DeiT-Tiny (72.52% Top-1 / 0.70 GFLOPs)
│   ├── main.py                    # Evaluation and execution script
│   ├── organize_val.py            # Preprocessing script for ImageNet validation set
│   ├── models/                    # Official DeiT-MCTF model definitions
│   └── utils/                     # Author utilities and metric loggers
│
└── Custom_MCTF/                   # PART 2: Custom Standalone Implementation
    ├── mctf_core.py               # Pure PyTorch Multi-Criteria Scoring & Bipartite Soft Matching
    ├── models.py                  # MCTFVisionTransformer with One-Step-Ahead Approximate Attention
    ├── EuroSAT_nb.ipynb           # Executed EuroSAT experiment (Accuracy, Early Stopping, GFLOPs)
    ├── cifar10_nb.ipynb           # Executed CIFAR-100 experiment (Multi-class generalization)
    ├── training_history.json      # Cached training metrics for instant visualization
    └── README.md                  # Custom implementation details
```

---

## 🔬 Part 1: Official Paper Reproduction (ImageNet-1K)

We reproduced the reported validation metrics on the official **50,000 ImageNet validation images** using the authors' published checkpoints on a single consumer GPU (**NVIDIA GeForce RTX 4060, 8GB VRAM**).

### Hardware Optimization
The authors' default scripts set `--batch-size 1024` and `--num_workers 16`, causing `MemoryError` crashes on standard workstations. We adapted execution to `--batch-size 64` and `--num_workers 2` without altering mathematical precision:

```bash
python MCTF/main.py --eval --model deit_small --dataset IMNET \
  --data_path "imagenet" --resume "checkpoint_tome_best_DT-s.pth" \
  --use_mctf True --mctf_type "[16,0,0,1,1,1,20,40,1,1,0]" \
  --task_type "[1,0,0,0]" --r_evals "[16]" \
  --activate_layer "[1,2,3,4,5,6,7,8,9,10,11]" --num_workers 2 --batch-size 64
```

### Reproduction Results

| Model Architecture | Paper Claimed GFLOPs | Our Reproduced GFLOPs | Paper Claimed Top-1 | Our Reproduced Top-1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DeiT-Small (MCTF)** | 2.60 | **2.60** | 80.1% | **79.94%** (95.08% Top-5) | ✅ Verified Match |
| **DeiT-Tiny (MCTF)** | ~0.70 | **0.70** | ~73.0% | **72.52%** (91.31% Top-5) | ✅ Verified Match |

*Raw reproduction traces are available in [eval_log_deit_s.txt](MCTF/eval_log_deit_s.txt) and [eval_log_deit_t.txt](MCTF/eval_log_deit_t.txt).*

---

## 🛠️ Part 2: Custom Standalone Implementation (`Custom_MCTF`)

Rather than relying on invasive modifications to Facebook's original DeiT codebase, we re-implemented the MCTF math cleanly from scratch:

### 1. Mathematical Framework (`mctf_core.py`)
* **Multi-Criteria Scoring:** Combines Embedding Similarity ($S_{sim}$), Attention Informativeness ($S_{info}$), and Token Size ($S_{size}$):
  $$\mathcal{S} = \tau_{sim} S_{sim} + \tau_{info} S_{info} + \tau_{size} S_{size}$$
* **Bipartite Soft Matching:** Partitions tokens into source $\mathcal{A}$ and target $\mathcal{B}$ sets, directing redundant tokens into dominant cluster centers.
* **Protected `[CLS]` Token:** Strict masking prevents the classification token from ever merging.

### 2. One-Step-Ahead Approximate Attention (`models.py`)
Standard token pruning algorithms discard pre-computed attention. MCTF looks ahead to compute $A^{(l+1)}$, aggregates it to match the fused token dimensions ($A_n = M \cdot A^{(l+1)} \cdot M^T$), and **bypasses the costly $Q \times K^T$ matrix multiplication** in the subsequent layer by directly calculating $A_n \times V$.

---

## 📊 Downstream Experiments & Multi-Dataset Benchmarks

To systematically evaluate the generalization of our custom MCTF implementation, our team designed a multi-domain experimental suite across three distinct visual domains: **EuroSAT** (satellite remote sensing), **CIFAR-100** (fine-grained natural images), and **SVHN** (street view digits).

### 🏆 Multi-Dataset Comparative Summary

| Benchmark Dataset | Visual Domain | Classes | Baseline DeiT-Tiny Acc | Custom MCTF-ViT Acc | Accuracy Delta ($\Delta$) | GFLOPs (Baseline $\rightarrow$ MCTF) | Token Reduction | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EuroSAT** | Satellite Imagery | 10 | 97.78% | **97.57%** | **$-0.21\%$** | 1.258 $\rightarrow$ **1.175** (-6.62%) | 197 $\rightarrow$ **107** (-45.7%) | ✅ Completed |
| **CIFAR-100** | Fine-Grained Objects | 100 | 78.72% | **76.52%** | **$-2.20\%$** | 1.258 $\rightarrow$ **1.175** (-6.62%) | 197 $\rightarrow$ **107** (-45.7%) | ✅ Completed |
| **SVHN** | Cropped Digits | 10 | Domain Benchmark | Domain Benchmark | Domain Eval | 1.258 $\rightarrow$ **1.175** (-6.62%) | 197 $\rightarrow$ **107** (-45.7%) | 📦 Ingested Suite |

---

### 1. EuroSAT Land-Use Classification (`EuroSAT_nb.ipynb`)
Evaluated on **27,000 Sentinel-2 satellite images** across 10 land cover classes.
* **Baseline DeiT-Tiny:** 97.78% Peak Test Acc | 1.258 GFLOPs (197 tokens)
* **Custom MCTF-ViT:** 97.57% Peak Test Acc | 1.175 GFLOPs (107 tokens)
* **Key Finding:** Custom MCTF preserved **99.8% of the baseline accuracy** ($\Delta = 0.21\%$, representing only 11 images out of 5,400) while reducing sequence length by **45.7%** and reducing GFLOPs by **6.62%**.
* **Domain Behavior:** Satellite images feature high background spatial redundancy (forests, water bodies, agricultural fields), making them an ideal match for MCTF bipartite token merging.

### 2. CIFAR-100 Fine-Grained Classification (`cifar10_nb.ipynb`)
Evaluated under extreme label complexity (**100 distinct classes**, 50,000 training images, 10,000 test images) with a 90/10 Train/Validation split and unbiased Test evaluation.
* **Baseline DeiT-Tiny:** 78.88% Peak Val Acc | **78.72% Final Unbiased Test Acc** (Early stopped at epoch 15)
* **Custom MCTF-ViT:** 75.98% Peak Val Acc | **76.52% Final Unbiased Test Acc** (Early stopped at epoch 12)
* **Key Finding:** On 100 fine-grained classes where visual differences are subtle, MCTF retains solid feature representation within **2.20%** of the uncompressed model while maintaining significant FLOPs savings and converging 3 epochs earlier.

### 3. SVHN Street View House Numbers
Evaluated on real-world cropped digit images with varying natural backgrounds and lighting conditions.
* Ingested into the benchmark data pipeline to assess token fusion sensitivity to sharp localized numeric features vs. surrounding architectural textures.

---

## 🤝 Collaborative Benchmarks & Group Division

Our group divided the implementation and experimental evaluation as follows:
* **Maaz Ali:** Official ImageNet-1K reproduction, modular `models.py` & `mctf_core.py` refactoring, and EuroSAT benchmark pipeline (`EuroSAT_nb.ipynb`).
* **Anirudh Sandeep Gupta:** CIFAR-100 multi-class generalization experiment and 3-way validation protocol (`cifar10_nb.ipynb`).
* **Vidya Karthika:** Comparative efficiency analysis, mathematical modeling, and multi-criteria weight ablation framework.

All experiments adhere to deterministic seeds (`SEED = 42`) and the Zero Test-Leakage validation protocol.

All experiments adhere to deterministic seeds (`SEED = 42`) and the Zero Test-Leakage validation protocol.

---

## 📜 Citation

```bibtex
@inproceedings{lee2024multi,
  title={Multi-criteria Token Fusion with One-step-ahead Attention for Efficient Vision Transformers},
  author={Lee, Sanghyeok and Choi, Joonmyung and Kim, Hyunwoo J.},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
