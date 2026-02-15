# 🚜 Robust Off-Road Semantic Segmentation 🌲

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**Learned ensemble of SegFormer (mit-b4) + DINOv2 (ViT-Base) with a trained Logistic Unifier**

*Per-pixel intelligent fusion for challenging outdoor terrain segmentation*

[🚀 Quick Start](#-quickstart--reproduce-our-results) • [📊 Performance](#-performance-notes) • [🔬 Analysis](#-how-we-analyze-failures) • [👥 Team](#-team)

</div>

---

## ✨ Highlights

| Feature | Description |
|---------|-------------|
| 🎯 **Learned Ensemble** | Not simple averaging — a 1×1 conv (Logistic Regression Unifier) dynamically fuses backbones per pixel |
| 🧠 **SegFormer (mit-b4)** | Crisp boundaries and global context understanding |
| 🧩 **DINOv2 (ViT-Base)** | Dense, texture-aware features for rocks, clutter, and fine details |
| ⚖️ **Focal Loss** | Combats rare-class imbalance (e.g., Lush Bush) |
| 📈 **Rich Outputs** | mIoU, per-class IoU, confusion matrix, and failure-case visualizations |

---

## 🗂️ Repository Structure

```
HrackKack-Submission/
│
├── 📁 Color_Images/              # (Dataset) Raw images
├── 📁 Segmentation/              # (Dataset) Ground truth masks
│
├── 💾 sgfmr.pth                  # Pre-trained SegFormer weights
├── 💾 dinov2.pth                 # Pre-trained DINOv2 weights
├── 💾 best_dual_logistic.pth     # (Generated) Trained ensemble weights
│
├── 🐍 SegFormerTrain.py          # Train SegFormer backbone (if needed)
├── 🐍 DinoV2train.py             # Train DINOv2 backbone (if needed)
├── 🚀 LogisticRegression.py      # MAIN: Trains the 1×1 logistic unifier
├── 📊 test.py                    # Evaluation: IoU, confusion matrix & failure cases
│
└── 📖 README.md                  # This file
```

---

## 🧩 Quickstart — Reproduce Our Results

### 1️⃣ Install Dependencies

```bash
pip install torch torchvision numpy opencv-python albumentations \
            transformers tqdm matplotlib seaborn scikit-learn evaluate
```

### 2️⃣ Prepare Dataset

Expected directory layout:

```
train/
  ├── Color_Images/
  └── Segmentation/
test/
  ├── Color_Images/
  └── Segmentation/
```

> 💡 **Tip:** You can symlink or copy — the scripts expect these paths.

### 3️⃣ Add Backbone Checkpoints

Place these files in the repository root:

- `sgfmr.pth` — SegFormer pretrained weights
- `dinov2.pth` — DINOv2 pretrained weights

> ⚡ The ensemble trainer **freezes both backbones** and trains only the 1×1 conv unifier.

### 4️⃣ Train the Unifier

```bash
python LogisticRegression.py
```

**What happens:**
- **Input:** `sgfmr.pth` + `dinov2.pth`
- **Loss:** Focal Loss (handles rare classes like Lush Bush)
- **Output:** `best_dual_logistic.pth`

### 5️⃣ Evaluate

```bash
python test.py
```

**Generates:**
- ✅ Per-class IoU scores
- ✅ Mean IoU (mIoU)
- ✅ `confusion_matrix.png`
- ✅ Failure-case visualizations for error analysis

---

## 🧠 Model & Design Choices

### Why This Ensemble?

<table>
<tr>
<td width="50%">

#### 🎨 SegFormer Strengths
- Excels at **boundaries** and **context**
- Great for Sky vs Tree separation
- Strong global understanding

</td>
<td width="50%">

#### 🔍 DINOv2 Strengths
- Excels at **fine textures**
- Object-level cues (e.g., Rocks)
- Dense feature representations

</td>
</tr>
</table>

**🎯 Logistic Unifier:** A learned 1×1 conv layer lets the network **learn where to trust each model** instead of hand-crafted averaging.

### 🔥 Loss Function

**Focal Loss** is used to:
- Reduce the impact of dominant classes
- Emphasize rare classes (e.g., Lush Bush)
- Improve overall balance

### 🔄 Inference Pipeline

```
Input Image
    ↓
[SegFormer] → Probability Map → Normalize
                                      ↓
                                 Concatenate
                                      ↓
[DINOv2]    → Probability Map → Normalize
    ↓
1×1 Conv (Logistic Unifier)
    ↓
Softmax
    ↓
Final Segmentation Mask
```

---

## 📊 Performance Notes

| Class | Performance | Notes |
|-------|-------------|-------|
| 🌥️ **Sky** | ⭐⭐⭐ High IoU | SegFormer handles boundaries exceptionally |
| 🪨 **Rocks** | ⭐⭐⭐ High IoU | DINOv2 captures texture detail |
| 🌿 **Lush Bush** | ⭐⭐ Improved | Still challenging; focal loss + ensemble helps |
| 🌾 **Dry Grass** | ⭐⭐ Moderate | Confusion with Ground Clutter (similar texture/color) |

### 🚨 Common Failure Modes

1. **Texture Confusion:** Dry Grass ↔ Ground Clutter (similar appearance)
2. **Lighting Issues:** Extreme lighting/shadows cause misclassification
3. **Rare Classes:** Small, infrequent classes need more examples/augmentation

---

## 🔬 How We Analyze Failures

The `test.py` script automatically generates:

| Output | Description |
|--------|-------------|
| 📊 `confusion_matrix.png` | Per-class confusion visualization |
| 🖼️ **Failure visualizations** | Side-by-side: Input → Ground Truth → Prediction |

> 💡 **Tip:** Use more aggressive augmentations (color jitter, random shadows) or synthetic minority oversampling for rare classes.

---

## 🚀 Tips to Improve Further

### 🎯 Training Strategies

- [ ] **Fine-tune backbones** selectively if you have enough labeled data
- [ ] **Class-balanced sampling** for rare classes like Lush Bush
- [ ] **Stronger augmentations** (color jitter, random shadows, cutout)
- [ ] **Mixed-precision training** (AMP) to fit larger batches

### 🔧 Architecture Enhancements

- [ ] **Multi-scale inference** for better boundary detection
- [ ] **CRF post-processing** for sharper, more coherent masks
- [ ] **Attention mechanisms** in the unifier layer
- [ ] **Test-time augmentation** (TTA) for robust predictions

### 📊 Data Improvements

- [ ] **Synthetic data generation** for rare classes
- [ ] **Hard negative mining** for challenging examples
- [ ] **Active learning** to identify and label informative samples

---

## 👥 Team HrackKack

<table>
<tr>
<td align="center">
<img src="https://via.placeholder.com/100" width="100px;" alt=""/><br />
<sub><b>Member 1</b></sub><br />
<sub>Role/Specialty</sub>
</td>
<td align="center">
<img src="https://via.placeholder.com/100" width="100px;" alt=""/><br />
<sub><b>Member 2</b></sub><br />
<sub>Role/Specialty</sub>
</td>
<td align="center">
<img src="https://via.placeholder.com/100" width="100px;" alt=""/><br />
<sub>Role/Specialty</sub>
</td>
<td align="center">
<img src="https://via.placeholder.com/100" width="100px;" alt=""/><br />
<sub><b>Member 4</b></sub><br />
<sub>Role/Specialty</sub>
</td>
</tr>
</table>

> *Replace with actual names, photos, and roles as desired.*

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@misc{hrackack2025offroad,
  title={Robust Off-Road Semantic Segmentation with Learned Ensemble},
  author={Team HrackKack},
  year={2025},
  howpublished={\url{https://github.com/yourusername/HrackKack-Submission}}
}
```

### 📚 Model Citations

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
@inproceedings{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  booktitle={TMLR},
  year={2023}
}
```

---

## 🤝 Acknowledgements

- 🙏 Thanks to the **SegFormer** and **DINOv2** teams for their foundational models
- 🎓 Dataset contributors and annotators
- 💰 [Add funding sources if applicable]
- 🌟 Open-source community

---

## 📬 Contact & Support

<div align="center">

**Questions? Issues? Collaboration ideas?**

[![GitHub Issues](https://img.shields.io/badge/Issues-Report%20Bug-red?logo=github)](https://github.com/yourusername/HrackKack-Submission/issues)
[![Email](https://img.shields.io/badge/Email-Contact%20Team-blue?logo=gmail)](mailto:team@hrackback.example.com)

</div>

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

[Full MIT License text...]
```

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

Made with ❤️ by Team HrackKack

</div>
