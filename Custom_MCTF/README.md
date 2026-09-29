# MCTF Group Experiment Guide

## 1. Files in this Package
1. `models.py`: DeiT-Tiny baseline and custom MCTF-ViT model implementations.
2. `mctf_core.py`: Pure PyTorch token fusion, bipartite matching, and similarity clustering logic.
3. `requirements.txt`: Python package dependencies.
4. `train_template.ipynb`: Production-grade experimental notebook for SVHN & CIFAR-100.
5. `README.md`: This guide.

---

## 2. Dataset Division (Group of 3)
* **Teammate 1 (You):** `EuroSAT` (Satellite land-use classification - 10 classes) -> **Done!**
* **Teammate 2:** `SVHN` (Street View House Numbers - 10 classes)
* **Teammate 3:** `CIFAR100` (Fine-grained object classification - 100 classes)

---

## 3. Academic Experimental Protocol
* **Full Reproducibility:** All seeds are locked (`SEED = 42`) across Python, NumPy, PyTorch, and CUDA deterministic backends.
* **Zero Test Leakage (Train / Val / Test):**
  * **Train Set:** Used strictly for backpropagation.
  * **Validation Set:** Used per-epoch for model checkpointing and Early Stopping.
  * **Test Set:** Kept completely isolated and evaluated **once at the end** on the optimal checkpoint to report unbiased final test accuracy.

---

## 4. How Your Teammates Run It
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Open `train_template.ipynb` in VS Code or Jupyter.**
3. **Select Assigned Dataset in Cell 2:**
   ```python
   # For Teammate 2 (SVHN):
   DATASET_NAME = 'SVHN'

   # For Teammate 3 (CIFAR-100):
   DATASET_NAME = 'CIFAR100'
   ```
4. **Click "Run All"**:
   * Downloads the dataset.
   * Trains both models with Early Stopping guided by the validation set.
   * Evaluates the best checkpoint on the test set.
   * Outputs the curves and GFLOPs complexity analysis.
