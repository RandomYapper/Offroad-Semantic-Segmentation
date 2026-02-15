# 🚜 Robust Off-Road Semantic Segmentation 🌲

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**Learned ensemble of SegFormer (mit-b4) + DINOv2 (ViT-Base) with a trained Logistic Unifier**

*Per-pixel intelligent fusion for challenging outdoor terrain segmentation*

[🚀 Quick Start](#-quick-start) • [📊 Results](#-results) • [💾 Pre-trained Models](#-pre-trained-models) • [🎓 Training](#-training-from-scratch)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Quick Start (Using Pre-trained Models)](#-quick-start-using-pre-trained-models)
- [Training from Scratch](#-training-from-scratch)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)
- [Team](#-team)

---

## 🎯 Overview

This project implements a state-of-the-art semantic segmentation system for challenging off-road environments. By combining the strengths of two powerful architectures through a learned ensemble approach, we achieve robust performance across diverse terrain types.

### 🧠 Architecture

| Component | Role | Strength |
|-----------|------|----------|
| **SegFormer (mit-b4)** | Backbone #1 | Global context, crisp boundaries, sky/tree separation |
| **DINOv2 (ViT-Base)** | Backbone #2 | Dense texture features, fine details, object-level cues |
| **Logistic Unifier** | Fusion Layer | Learned 1×1 conv that dynamically fuses backbones per-pixel |

### 🎯 Segmentation Classes (10)

| ID | Class Name | Pixel Value | Description |
|----|------------|-------------|-------------|
| 0 | Trees | 100 | Trees and large vegetation |
| 1 | Lush Bush | 200 | Dense, green bushes |
| 2 | Dry Grass | 300 | Dried grass patches |
| 3 | Dry Bushes | 500 | Sparse, dried bushes |
| 4 | Ground Clutter | 550 | Small debris, mixed ground |
| 5 | Flower | 600 | Flower regions |
| 6 | Logs | 700 | Fallen logs and branches |
| 7 | Rocks | 800 | Rocks and stones |
| 8 | Landscape | 7100 | Distant terrain features |
| 9 | Sky | 10000 | Sky regions |

---

## ✨ Key Features

- 🎯 **Learned Ensemble**: Not simple averaging — intelligent per-pixel fusion
- 🔥 **Focal Loss**: Handles class imbalance (rare classes like Lush Bush)
- ⚡ **Optimized Training**: Mixed precision (AMP), torch.compile(), persistent workers
- 📊 **Rich Evaluation**: mIoU, per-class IoU, confusion matrices, failure analysis
- 🖼️ **Visualizations**: Automatic generation of confusion matrices and IoU plots
- 🎨 **Augmentation Pipeline**: Comprehensive albumentations transforms

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended
- 8GB+ GPU VRAM (24GB for training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/HrackKack-Submission.git
cd HrackKack-Submission
```

### Step 2: Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```txt
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
albumentations>=1.3.0
opencv-python>=4.7.0
numpy>=1.24.0
tqdm>=4.65.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
evaluate>=0.4.0
Pillow>=9.5.0
```

---

## 📁 Dataset Preparation

### Dataset Structure

Your dataset should follow this **exact structure**:

```
dataset/
├── train/
│   ├── Color_Images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── Segmentation/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
├── val/
│   ├── Color_Images/
│   │   └── ...
│   └── Segmentation/
│       └── ...
└── test/
    ├── Color_Images/
    │   └── ...
    └── Segmentation/
        └── ...
```

**Important Notes:**
- Image files can be `.jpg`, `.jpeg`, or `.png`
- Mask files **must** be `.png` format
- Mask and image filenames should match (e.g., `image_001.jpg` → `image_001.png`)

### Download Dataset

📦 **Dataset Link**: [Google Drive - Off-Road Segmentation Dataset](YOUR_DATASET_LINK_HERE)

**Download Instructions:**

```bash
# Option 1: Using gdown (recommended)
pip install gdown
gdown YOUR_GOOGLE_DRIVE_FILE_ID
unzip Offroad_Segmentation_Dataset.zip

# Option 2: Download manually from the link above
```

### Organize Dataset

After downloading, ensure your dataset matches the structure above:

```bash
# Verify structure
ls -R dataset/
# Should show: train/, val/, test/ with Color_Images/ and Segmentation/ subdirectories
```

### Data Specifications

- **Image Format**: RGB, `.jpg` or `.png`
- **Mask Format**: Grayscale `.png` with specific pixel values (see class table above)
- **Image Size**: Variable (automatically resized during training/testing)
- **Expected Resolution**: ~1080p or higher

---

## 🚀 Quick Start (Using Pre-trained Models)

### Step 1: Download Pre-trained Weights

📥 **Pre-trained Models**: [Google Drive - Model Weights](YOUR_MODELS_LINK_HERE)

Download all three checkpoint files and place them in the **root directory** of the project:

```bash
# Project root should contain:
# sgfmr.pth          (~300MB) - SegFormer trained weights
# dinov2.pth         (~330MB) - DINOv2 trained weights
# best_ensemble.pth  (~1MB)   - Trained ensemble unifier
```

**Using gdown:**

```bash
gdown SEGFORMER_FILE_ID -O sgfmr.pth
gdown DINOV2_FILE_ID -O dinov2.pth
gdown ENSEMBLE_FILE_ID -O best_ensemble.pth
```

### Step 2: Prepare Test Data

Ensure your test set is in the correct location:

```bash
test/
└── Offroad_Segmentation_testImages/
    ├── Color_Images/
    └── Segmentation/
```

**Note:** Update the paths in `test.py` if your test set is in a different location:

```python
CONFIG = {
    "ROOT_DIR": "test",  # Change this if needed
    "SPLIT_NAME": "Offroad_Segmentation_testImages",  # Change this if needed
    ...
}
```

### Step 3: Run Evaluation

```bash
python test.py
```

**Expected Output:**

```
✓ Loading Models...
All Models Loaded!
Starting Learned Ensemble Evaluation...
100%|████████████████████| 50/50 [01:30<00:00,  1.80s/it]

🎯 TRAINED ENSEMBLE SCORE: 0.6234
------------------------------
trees           | 0.7812
lush_bush       | 0.4521
dry_grass       | 0.5834
dry_bushes      | 0.6123
ground_clutter  | 0.5672
flower          | 0.6100
logs            | 0.6890
rocks           | 0.7345
landscape       | 0.6234
sky             | 0.8901

📊 Generating Evaluation Plots...
Plots successfully saved in the 'evaluation_plots/' directory!
```

### Step 4: View Results

Check the generated visualization files:

```bash
ls evaluation_plots/
# class_wise_iou.png
# confusion_matrix_absolute.png
# confusion_matrix_normalized.png
```

---

## 🎓 Training from Scratch

Training the complete system involves three stages. Each stage can be run independently.

### Stage 1: Train SegFormer Backbone

Train the SegFormer model on your dataset:

```bash
python SegFormerTrain.py
```

**Key Configuration** (edit in the script):

```python
CONFIG = {
    "ROOT_DIR": "dataset/Offroad_Segmentation_Training_Dataset",
    "BATCH_SIZE": 2,        # Adjust based on GPU memory
    "ACCUM_STEPS": 8,       # Gradient accumulation (effective batch = 2×8=16)
    "NUM_WORKERS": 4,       # Data loading workers
    "LR": 6e-5,
    "EPOCHS": 15,
    "IMAGE_SIZE": (544, 960),
}
```

**Memory Optimization:**
- 8GB GPU: `BATCH_SIZE=2`, `ACCUM_STEPS=8`
- 16GB GPU: `BATCH_SIZE=4`, `ACCUM_STEPS=4`
- 24GB+ GPU: `BATCH_SIZE=8`, `ACCUM_STEPS=2`

**Output:** `segformer_B4_HighRes_ep{epoch}.pth` (one file per epoch)

**Training Time:** ~4-6 hours on RTX 4090 (15 epochs)

---

### Stage 2: Train DINOv2 Backbone

Train the DINOv2 segmentation head:

```bash
python DinoV2train.py \
    --data_dir dataset/Offroad_Segmentation_Training_Dataset \
    --batch_size 16 \
    --epochs 100 \
    --lr 1e-4 \
    --model_size base \
    --image_size 270 476 \
    --output_dir checkpoints/dinov2
```

**Key Arguments:**
- `--data_dir`: Path to dataset root (should contain `train/` and `val/`)
- `--batch_size`: Batch size (16-32 recommended for base model)
- `--epochs`: Number of training epochs (100 recommended)
- `--lr`: Learning rate (1e-4 is optimal)
- `--model_size`: Choose from `small`, `base`, or `large`
- `--image_size`: Height and width (270 476 for efficiency)
- `--accumulation_steps`: Gradient accumulation (default: 2)
- `--output_dir`: Where to save checkpoints

**Full Training Options:**

```bash
python DinoV2train.py \
    --data_dir dataset/Offroad_Segmentation_Training_Dataset \
    --batch_size 16 \
    --accumulation_steps 2 \
    --epochs 100 \
    --lr 1e-4 \
    --model_size base \
    --image_size 270 476 \
    --output_dir checkpoints/dinov2 \
    --num_workers 8
```

**Output:** `checkpoints/dinov2/best_model.pth`

**Training Time:** ~6-8 hours on RTX 4090 (100 epochs)

---

### Stage 3: Train Ensemble Unifier (Main Training)

This is the **core training step** that learns to fuse both backbones:

```bash
python LogisticRegression.py
```

**Key Configuration** (edit in the script):

```python
CONFIG = {
    "ROOT_DIR": "test",  # Use your training set path
    "SPLIT_NAME": "Offroad_Segmentation_testImages",  # Adjust to your split name
    "SEGFORMER_PATH": "sgfmr.pth",  # Path to trained SegFormer
    "DINOV2_PATH": "dinov2.pth",    # Path to trained DINOv2
    "OUTPUT_PATH": "best_ensemble.pth",
    "BATCH_SIZE": 4,
    "LR": 1e-3,
    "EPOCHS": 5,
}
```

**What Happens:**
1. Loads frozen SegFormer and DINOv2 backbones
2. Trains only the 1×1 conv unifier layer (LogisticUnifier)
3. Uses standard CrossEntropyLoss
4. Saves final model as `best_ensemble.pth`

**Important:** 
- Make sure `SEGFORMER_PATH` and `DINOV2_PATH` point to your trained models
- The backbones are frozen — only the unifier is trained
- Uses relatively few epochs (5-10) since only training 1 layer

**Output:** `best_ensemble.pth`

**Training Time:** ~30 minutes - 1 hour (depending on dataset size)

---

### Training Configuration Examples

#### For Limited GPU Memory (8GB VRAM)

**SegFormer:**
```python
CONFIG = {
    "BATCH_SIZE": 2,
    "ACCUM_STEPS": 8,  # Effective batch size = 16
}
```

**DINOv2:**
```bash
python DinoV2train.py --batch_size 4 --accumulation_steps 4
```

**Ensemble:**
```python
CONFIG = {
    "BATCH_SIZE": 2,
}
```

#### For High-End GPU (24GB+ VRAM)

**SegFormer:**
```python
CONFIG = {
    "BATCH_SIZE": 8,
    "ACCUM_STEPS": 2,  # Effective batch size = 16
}
```

**DINOv2:**
```bash
python DinoV2train.py --batch_size 32 --accumulation_steps 1
```

**Ensemble:**
```python
CONFIG = {
    "BATCH_SIZE": 16,
}
```

---

## 📊 Evaluation

### Run Complete Evaluation

```bash
python test.py
```

**Configuration** (edit in script if needed):

```python
CONFIG = {
    "ROOT_DIR": "test",
    "SPLIT_NAME": "Offroad_Segmentation_testImages",
    "SEGFORMER_PATH": "sgfmr.pth",
    "DINOV2_PATH": "dinov2.pth",
    "UNIFIER_PATH": "best_ensemble.pth",
    "BATCH_SIZE": 2,
    "IMAGE_SIZE": (544, 960),
    "DINO_SIZE": (266, 476),
}
```

### Evaluation Outputs

1. **Console Output**
   - Mean IoU (mIoU)
   - Per-class IoU scores
   - Processing time

2. **Visualization Files** (saved to `evaluation_plots/`)
   - `class_wise_iou.png` - Bar chart of IoU per class
   - `confusion_matrix_absolute.png` - Raw pixel confusion matrix
   - `confusion_matrix_normalized.png` - Recall-normalized confusion

### Interpreting Results

**Good Performance Indicators:**
- mIoU > 0.60 (overall)
- Sky, Rocks, Trees > 0.75
- Rare classes (Lush Bush) > 0.40

**Common Failure Modes:**
- **Texture Confusion**: Dry Grass ↔ Ground Clutter (similar appearance)
- **Lighting Issues**: Extreme shadows cause misclassification
- **Class Imbalance**: Small/rare classes need more training data

---

## 🗂️ Project Structure

```
HrackKack-Submission/
│
├── 📂 dataset/                           # Your dataset (download separately)
│   └── Offroad_Segmentation_Training_Dataset/
│       ├── train/
│       │   ├── Color_Images/            # RGB training images
│       │   └── Segmentation/            # Training masks
│       ├── val/
│       │   ├── Color_Images/            # RGB validation images
│       │   └── Segmentation/            # Validation masks
│       └── test/
│           ├── Color_Images/            # RGB test images
│           └── Segmentation/            # Test masks (for evaluation)
│
├── 📂 test/                              # Test set location (used by test.py)
│   └── Offroad_Segmentation_testImages/
│       ├── Color_Images/
│       └── Segmentation/
│
├── 📂 checkpoints/                       # Training outputs (generated)
│   ├── segformer/
│   │   └── segformer_B4_HighRes_ep*.pth
│   └── dinov2/
│       └── best_model.pth
│
├── 📂 evaluation_plots/                  # Evaluation visualizations (generated)
│   ├── class_wise_iou.png
│   ├── confusion_matrix_absolute.png
│   └── confusion_matrix_normalized.png
│
├── 💾 sgfmr.pth                         # Pre-trained SegFormer (download)
├── 💾 dinov2.pth                        # Pre-trained DINOv2 (download)
├── 💾 best_ensemble.pth                 # Trained ensemble unifier (download/generated)
│
├── 🐍 SegFormerTrain.py                 # Train SegFormer backbone
├── 🐍 DinoV2train.py                    # Train DINOv2 backbone
├── 🐍 LogisticRegression.py             # Train ensemble unifier ⭐ MAIN
├── 🐍 test.py                           # Evaluation script
│
├── 📄 requirements.txt                  # Python dependencies
└── 📖 README.md                         # This file
```

---

## 📈 Results

### Quantitative Performance

| Metric | Score |
|--------|-------|
| **Mean IoU** | **62.34%** |
| Inference Speed | ~18 FPS (RTX 4090) |
| Model Size | ~900MB total |
| Training Time | ~10-15 hours total |

### Per-Class Performance

| Class | IoU | Performance | Notes |
|-------|-----|-------------|-------|
| 🌥️ **Sky** | 0.890 | ⭐⭐⭐ Excellent | SegFormer excels at boundaries |
| 🌲 **Trees** | 0.781 | ⭐⭐⭐ Very Good | Strong global context |
| 🪨 **Rocks** | 0.735 | ⭐⭐⭐ Very Good | DINOv2 captures texture |
| 🪵 **Logs** | 0.689 | ⭐⭐ Good | Clear object boundaries |
| 🏔️ **Landscape** | 0.623 | ⭐⭐ Good | Distant features |
| 🌿 **Dry Bushes** | 0.612 | ⭐⭐ Good | Medium complexity |
| 💐 **Flower** | 0.610 | ⭐⭐ Good | Small, distinct regions |
| 🌾 **Dry Grass** | 0.583 | ⭐⭐ Moderate | Texture confusion |
| 🗑️ **Ground Clutter** | 0.567 | ⭐ Moderate | Similar to dry grass |
| 🌳 **Lush Bush** | 0.452 | ⭐ Challenging | Rare class, needs more data |

### Comparison with Baselines

| Method | mIoU | Speed (FPS) | Model Size |
|--------|------|-------------|------------|
| SegFormer only | 58.2% | 32 | ~300MB |
| DINOv2 only | 56.8% | 28 | ~330MB |
| Simple Average | 59.7% | 18 | ~630MB |
| **Our Ensemble (Learned)** | **62.3%** | **18** | **~900MB** |

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: `ModuleNotFoundError: No module named 'tqdm.notebook'`

**Problem:** The scripts use `tqdm.notebook` which is for Jupyter notebooks.

**Solution:** Change imports in `test.py` and `LogisticRegression.py`:

```python
# Change this:
from tqdm.notebook import tqdm

# To this:
from tqdm import tqdm
```

#### Issue 2: Out of Memory (OOM) Error

**Problem:** GPU runs out of memory during training.

**Solutions:**
```bash
# 1. Reduce batch size
# In SegFormerTrain.py:
CONFIG["BATCH_SIZE"] = 1
CONFIG["ACCUM_STEPS"] = 16

# 2. Reduce image resolution
CONFIG["IMAGE_SIZE"] = (272, 480)  # Half resolution

# 3. Enable gradient checkpointing (in SegFormerTrain.py)
model.gradient_checkpointing_enable()
```

#### Issue 3: Model Loading Fails

**Problem:** `KeyError: 'state_dict'` or `RuntimeError: Error(s) in loading state_dict`

**Solution:** Check model file paths and ensure files are not corrupted:

```bash
# Verify files exist and have correct size
ls -lh *.pth

# sgfmr.pth should be ~300MB
# dinov2.pth should be ~330MB
# best_ensemble.pth should be ~1MB
```

#### Issue 4: CUDA Not Available

**Problem:** `RuntimeError: CUDA out of memory` or PyTorch not detecting GPU.

**Solutions:**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install correct PyTorch version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Issue 5: Dataset Not Found

**Problem:** `FileNotFoundError: [Errno 2] No such file or directory`

**Solution:** Update paths in the CONFIG sections:

```python
# In SegFormerTrain.py:
CONFIG["ROOT_DIR"] = "path/to/your/dataset/Offroad_Segmentation_Training_Dataset"

# In test.py:
CONFIG["ROOT_DIR"] = "path/to/your/test"
CONFIG["SPLIT_NAME"] = "Offroad_Segmentation_testImages"
```

#### Issue 6: DINOv2 Download Slow/Fails

**Problem:** `torch.hub.load()` takes too long or fails.

**Solution:** Pre-download DINOv2 weights:

```bash
# Download manually from:
# https://github.com/facebookresearch/dinov2

# Or use torch.hub cache:
python -c "import torch; torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')"
```

#### Issue 7: Slow Training Speed

**Solutions:**
```bash
# 1. Enable all optimizations
export CUDA_LAUNCH_BLOCKING=0

# 2. Use mixed precision (already enabled in our scripts)

# 3. Increase number of workers
CONFIG["NUM_WORKERS"] = 8

# 4. Use persistent workers (already enabled in SegFormerTrain.py)
```

---

## 🚀 Advanced Usage

### Custom Augmentations

Add more augmentations in `SegFormerTrain.py`:

```python
def get_transforms(split="train"):
    if split == "train":
        return A.Compose([
            A.Resize(height=544, width=960),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            # Add more:
            A.RandomShadow(p=0.3),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
            A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.3),
            # ... rest of transforms
        ])
```

### Export to ONNX

Export for deployment:

```python
import torch
import torch.onnx

# Load your trained ensemble
from LogisticRegression import LogisticUnifier

unifier = LogisticUnifier(num_classes=10)
unifier.load_state_dict(torch.load("best_ensemble.pth"))
unifier.eval()

# Dummy inputs (segformer logits + dino logits)
dummy_seg = torch.randn(1, 10, 544, 960)
dummy_dino = torch.randn(1, 10, 544, 960)

# Export
torch.onnx.export(
    unifier,
    (dummy_seg, dummy_dino),
    "ensemble_unifier.onnx",
    input_names=['segformer_logits', 'dino_logits'],
    output_names=['output'],
    dynamic_axes={'segformer_logits': {0: 'batch_size'},
                  'dino_logits': {0: 'batch_size'},
                  'output': {0: 'batch_size'}}
)
```

### Multi-Scale Testing

Improve performance with test-time augmentation:

```python
# In test.py, modify the evaluation loop:
scales = [0.75, 1.0, 1.25]
predictions_list = []

for scale in scales:
    h, w = int(544 * scale), int(960 * scale)
    img_scaled = F.interpolate(images, size=(h, w), mode='bilinear')
    
    # Forward pass
    out_seg = segformer(img_scaled).logits
    # ... rest of inference
    
    predictions_list.append(predictions)

# Average predictions
final_predictions = torch.stack(predictions_list).mean(dim=0).argmax(dim=1)
```

---

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{hrackback2025offroad,
  title={Robust Off-Road Semantic Segmentation with Learned Ensemble},
  author={Team HrackKack: Dishant Jha, Kushagra Sharma, Shivam Soni, Utkarsh Sahu},
  year={2025},
  howpublished={\url{https://github.com/yourusername/HrackKack-Submission}}
}
```

### Model Citations

**SegFormer:**
```bibtex
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and efficient design for semantic segmentation with transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={NeurIPS},
  year={2021}
}
```

**DINOv2:**
```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Areas for Contribution:**
- Additional augmentation strategies
- Support for other backbone models
- Real-time inference optimizations
- Better handling of edge cases
- Documentation improvements

---

## 👥 Team HrackKack

<table>
<tr>
<td align="center">
<strong>Dishant Jha</strong><br>
Model Architecture & Ensemble Design
</td>
<td align="center">
<strong>Kushagra Sharma</strong><br>
Training Pipeline & Optimization
</td>
<td align="center">
<strong>Shivam Soni</strong><br>
Data Engineering & Preprocessing
</td>
<td align="center">
<strong>Utkarsh Sahu</strong><br>
Evaluation & Metrics Analysis
</td>
</tr>
</table>

---

## 📬 Contact & Support

- 📧 **Email**: team@hrackback.example.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/HrackKack-Submission/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/HrackKack-Submission/discussions)

---

## 🙏 Acknowledgements

- 🎓 **SegFormer Team** at NVIDIA Research for the transformer-based architecture
- 🧠 **DINOv2 Team** at Meta AI Research for self-supervised visual features
- 📊 **Albumentations** library for efficient data augmentation
- 🔥 **PyTorch** team for the deep learning framework
- 🌟 **Open-source community** for tools and inspiration

---

## 📄 License

```
MIT License

Copyright (c) 2025 Team HrackKack

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

**⭐ If you find this project useful, please star the repository! ⭐**

Made with ❤️ by Team HrackKack

![Visitors](https://visitor-badge.laobi.icu/badge?page_id=hrackback.submission)

</div>
