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
    ├── train_experiments.ipynb    # Executed EuroSAT experiment (Accuracy, Early Stopping, GFLOPs)
    ├── train_template.ipynb       # Template for SVHN & CIFAR-100 (Train / Val / Test protocol)
    ├── training_history.json      # Cached training metrics for instant visualization
    ├── MCTF_Group_Pack.zip        # Portable starter bundle for group experiments
    └── README.md                  # Teammate experiment instructions
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

## 📊 Downstream Experiments: EuroSAT Land-Use Classification

We evaluated our custom `MCTFVisionTransformer` side-by-side with a standard `deit_tiny_patch16_224` baseline on **EuroSAT** (10 classes, 27,000 satellite images) with Early Stopping (`patience=5`):

| Model | Parameters | Complexity (GFLOPs) | Peak Test Accuracy | Final Token Count | GFLOPs Saved |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline DeiT-Tiny** | 5.53 M | 1.258 | **97.78%** | 197 tokens | Reference (0%) |
| **Custom MCTF-ViT** | 5.53 M | **1.175** | **97.57%** | 107 tokens | **6.62% Reduction** |

> **Key Takeaway:** Custom MCTF preserved **99.8% of the baseline accuracy** ($\Delta = 0.21\%$, representing only 11 images out of 5,400) while reducing sequence length from 197 to 107 tokens and decreasing GFLOPs.

---

## 🚀 Teammate Experiments (SVHN & CIFAR-100)

For team collaboration, [Custom_MCTF/train_template.ipynb](Custom_MCTF/train_template.ipynb) provides a production-grade template implementing the **Zero Test-Leakage Protocol**:
* **Train Set (90%):** Gradient backpropagation.
* **Validation Set (10%):** Early Stopping and checkpoint selection (`best_weights = copy.deepcopy(model.state_dict())`).
* **Test Set:** Evaluated **strictly once at the end** on the optimal validation checkpoint.
* **Reproducibility:** Seed locked (`SEED = 42`) across Python, NumPy, PyTorch, and CUDA.

### Running Teammate Experiments:
1. Extract [Custom_MCTF/MCTF_Group_Pack.zip](Custom_MCTF/MCTF_Group_Pack.zip)
2. Run `pip install -r requirements.txt`
3. Open `train_template.ipynb` and select assigned dataset in Cell 2:
   ```python
   DATASET_NAME = 'SVHN'       # For Street View House Numbers (10 classes)
   # DATASET_NAME = 'CIFAR100'  # For Fine-Grained Object Classification (100 classes)
   ```
4. Click **Run All**.

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
