import json
import os

path = os.path.join('Custom_MCTF', 'train_experiments.ipynb')
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 1: Add seed
c1_source = [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torchvision import datasets, transforms\n",
    "from torch.utils.data import DataLoader\n",
    "import matplotlib.pyplot as plt\n",
    "import time, random\n",
    "import numpy as np\n",
    "from tqdm import tqdm\n",
    "\n",
    "# 1. Guarantee Full Reproducibility\n",
    "SEED = 42\n",
    "random.seed(SEED)\n",
    "np.random.seed(SEED)\n",
    "torch.manual_seed(SEED)\n",
    "if torch.cuda.is_available():\n",
    "    torch.cuda.manual_seed_all(SEED)\n",
    "    torch.backends.cudnn.deterministic = True\n",
    "\n",
    "# Import our custom modules!\n",
    "from models import MCTFVisionTransformer, create_baseline_vit\n",
    "\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f\"Using device: {device} | Reproducibility SEED: {SEED}\")\n"
]
nb['cells'][1]['source'] = c1_source

# Cell 2: Add protocol note
c2_source = [
    "## 1. Dataset Preprocessing\n",
    "\n",
    "> **Note on Evaluation Protocol:**\n",
    "> In this baseline demonstration on EuroSAT, an 80/20 train/test partition was employed with test evaluation logged per epoch for rapid iteration. For large-scale benchmarks (such as SVHN and CIFAR-100 in our companion template), the three-way Train/Validation/Test protocol is strictly enforced, where early stopping is decided purely on an isolated validation split, and the test split is evaluated only once on the optimal checkpoint to eliminate data-snooping bias.\n"
]
nb['cells'][2]['source'] = c2_source

# Cell 3: Ensure random_split uses deterministic seed generator
c3_lines = nb['cells'][3]['source']
for i, line in enumerate(c3_lines):
    if 'random_split(eurosat_dataset, [train_size, test_size])' in line:
        c3_lines[i] = "eurosat_train, eurosat_test = torch.utils.data.random_split(eurosat_dataset, [train_size, test_size], generator=torch.Generator().manual_seed(SEED))\n"

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully updated EuroSAT notebook with reproducibility seeds and protocol note!")
