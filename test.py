import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
from tqdm.notebook import tqdm
import evaluate 

# --- NEW IMPORTS FOR VISUALIZATION ---
import matplotlib.pyplot as plt
import seaborn as sns
# -------------------------------------

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    #  PATHS
    "ROOT_DIR": "test", 
    "SPLIT_NAME": "Offroad_Segmentation_testImages", 
    
    #  MODEL PATHS
    "SEGFORMER_PATH": "sgfmr.pth", 
    "DINOV2_PATH": "dinov2.pth",
    "UNIFIER_PATH": "best_ensemble.pth", # <--- YOUR NEW TRAINED MODEL
    
    # SETTINGS
    "NUM_CLASSES": 10,
    "BATCH_SIZE": 2,      
    "IMAGE_SIZE": (544, 960), 
    "DINO_SIZE": (266, 476), 
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DINO_MODEL_TYPE": "base"
}

ID2LABEL = {
    0: "trees", 1: "lush_bush", 2: "dry_grass", 3: "dry_bushes",
    4: "ground_clutter", 5: "flower", 6: "logs", 7: "rocks",
    8: "landscape", 9: "sky"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ==========================================
# 2. MODEL CLASSES (Must be defined to load)
# ==========================================
class LogisticUnifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mixer = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1)
        
    def forward(self, seg_logits, dino_logits):
        x = torch.cat([seg_logits, dino_logits], dim=1)
        return self.mixer(x)

class ImprovedSegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH, hidden_dim=256):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.initial_conv = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, 1), nn.BatchNorm2d(hidden_dim), nn.GELU())
        self.refine1 = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim), nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim), nn.GELU())
        self.refine2 = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim), nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.BatchNorm2d(hidden_dim), nn.GELU())
        self.classifier = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1), nn.BatchNorm2d(hidden_dim // 2), nn.GELU(), nn.Conv2d(hidden_dim // 2, out_channels, 1))
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.initial_conv(x)
        x = x + self.refine1(x)
        x = x + self.refine2(x)
        return self.classifier(x)

class DINOv2Wrapper(nn.Module):
    def __init__(self, model_size, num_classes, image_size):
        super().__init__()
        backbone_archs = {"small": "vits14", "base": "vitb14", "large": "vitl14"}
        self.backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=f"dinov2_{backbone_archs[model_size]}")
        self.head = ImprovedSegmentationHead(768, num_classes, image_size[1]//14, image_size[0]//14)
    def forward(self, x):
        return self.head(self.backbone.forward_features(x)["x_norm_patchtokens"])

# ==========================================
# 3. UTILS & DATASET
# ==========================================
def align_dino_to_segformer(dino_logits):
    new_logits = torch.full_like(dino_logits, -100.0) 
    mapping = {1:0, 2:1, 3:2, 4:3, 5:4, 6:6, 7:7, 8:8, 9:9}
    for src, dst in mapping.items():
        new_logits[:, dst] = dino_logits[:, src]
    return new_logits

class OffRoadDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        self.transform = transform
        self.img_dir = os.path.join(root_dir, split, "Color_Images")
        self.mask_dir = os.path.join(root_dir, split, "Segmentation")
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self): return len(self.images)

    def convert_mask(self, mask):
        new_mask = np.zeros_like(mask, dtype=np.uint8)
        mapping = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}
        for k, v in mapping.items(): new_mask[mask == k] = v
        return new_mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, img_name)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, img_name.replace(".jpg", ".png")), cv2.IMREAD_UNCHANGED)
        mask = self.convert_mask(mask) if mask is not None else np.zeros(image.shape[:2], dtype=np.uint8)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']
        return image, mask.long()

def clean_sd(sd):
    return {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in sd.items()}

# ==========================================
# 4. EVALUATION LOOP
# ==========================================
def run_evaluation():
    # 1. Load Data
    tf = A.Compose([A.Resize(CONFIG["IMAGE_SIZE"][0], CONFIG["IMAGE_SIZE"][1]), A.Normalize(), ToTensorV2()])
    dataset = OffRoadDataset(CONFIG["ROOT_DIR"], CONFIG["SPLIT_NAME"], transform=tf)
    loader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2)

    # 2. Load Models
    print(" Loading Models...")
    segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b4", num_labels=10, ignore_mismatched_sizes=True).to(CONFIG["DEVICE"])
    ckpt = torch.load(CONFIG["SEGFORMER_PATH"], map_location=CONFIG["DEVICE"], weights_only=False)
    segformer.load_state_dict(clean_sd(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt), strict=False)

    dinov2 = DINOv2Wrapper(CONFIG["DINO_MODEL_TYPE"], CONFIG["NUM_CLASSES"], CONFIG["DINO_SIZE"]).to(CONFIG["DEVICE"])
    dkpt = torch.load(CONFIG["DINOV2_PATH"], map_location=CONFIG["DEVICE"], weights_only=False)
    if isinstance(dkpt, dict) and 'backbone_state_dict' in dkpt:
        dinov2.backbone.load_state_dict(clean_sd(dkpt['backbone_state_dict']))
        dinov2.head.load_state_dict(clean_sd(dkpt['head_state_dict']))
    else:
        dinov2.load_state_dict(clean_sd(dkpt.state_dict() if hasattr(dkpt, 'state_dict') else dkpt), strict=False)

    # 3. Load Trained Unifier
    unifier = LogisticUnifier(CONFIG["NUM_CLASSES"]).to(CONFIG["DEVICE"])
    unifier.load_state_dict(torch.load(CONFIG["UNIFIER_PATH"], map_location=CONFIG["DEVICE"], weights_only=False))
    print("All Models Loaded!")

    segformer.eval(); dinov2.eval(); unifier.eval()

    # 4. Run Loop
    metric = evaluate.load("mean_iou")
    print("Starting Learned Ensemble Evaluation...")
    
    with torch.no_grad():
        for images, masks in tqdm(loader):
            images, masks = images.to(CONFIG["DEVICE"]), masks.to(CONFIG["DEVICE"])
            
            # Forward Passes
            out_seg = segformer(images).logits
            out_seg = F.interpolate(out_seg, size=CONFIG["IMAGE_SIZE"], mode='bilinear', align_corners=False)
            
            img_dino = F.interpolate(images, size=CONFIG["DINO_SIZE"], mode='bilinear', align_corners=False)
            out_dino = dinov2(img_dino)
            
            # Align & Resize
            out_dino = align_dino_to_segformer(out_dino)
            out_dino = F.interpolate(out_dino, size=CONFIG["IMAGE_SIZE"], mode='bilinear', align_corners=False)
            
            #  USE THE TRAINED BRAIN
            final_logits = unifier(out_seg, out_dino)
            predictions = final_logits.argmax(dim=1)
            
            # Fix Clutter -> Rocks
            # predictions[predictions == 4] = 7 
            
            metric.add_batch(predictions=predictions.cpu().numpy(), references=masks.cpu().numpy())

    # 5. Results
    results = metric.compute(num_labels=10, ignore_index=255, reduce_labels=False)
    print(f"\n TRAINED ENSEMBLE SCORE: {results['mean_iou']:.4f}")
    print("-" * 30)
    for i, iou in enumerate(results['per_category_iou']):
        print(f"{ID2LABEL[i]:<15} | {iou:.4f}")

    # ==========================================
    # 6. VISUALIZATION PLOTS (NEW ADDITION)
    # ==========================================
    print("\n Generating Evaluation Plots...")
    plot_dir = "evaluation_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    categories = [ID2LABEL[i] for i in range(CONFIG["NUM_CLASSES"])]
    
    # A. Class-wise IoU Bar Chart
    ious = results['per_category_iou']
    plt.figure(figsize=(12, 6))
    sns.barplot(x=categories, y=ious, hue=categories, palette="viridis", legend=False)
    plt.title("Class-wise Intersection over Union (IoU)")
    plt.ylabel("IoU Score")
    plt.xlabel("Classes")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    for i, v in enumerate(ious):
        if not np.isnan(v):
            plt.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "class_wise_iou.png"), dpi=300)
    plt.close()

    # Get Confusion Matrix from evaluate metric
    conf_matrix = results['confusion_matrix']

    # B. Absolute Confusion Matrix (Pixel Counts)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=False, fmt='.0f', cmap="Blues", 
                xticklabels=categories, yticklabels=categories)
    plt.title("Confusion Matrix (Absolute Pixel Count)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "confusion_matrix_absolute.png"), dpi=300)
    plt.close()

    # C. Normalized Confusion Matrix (Recall / Class Accuracy)
    # Normalize by the sum of each row (True labels)
    row_sums = conf_matrix.sum(axis=1)[:, np.newaxis]
    # Avoid division by zero warnings for classes that might not exist in the test set
    conf_matrix_norm = np.divide(conf_matrix.astype('float'), row_sums, out=np.zeros_like(conf_matrix, dtype=float), where=row_sums!=0)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap="magma", 
                xticklabels=categories, yticklabels=categories)
    plt.title("Normalized Confusion Matrix (Recall per Class)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "confusion_matrix_normalized.png"), dpi=300)
    plt.close()

    print(f"Plots successfully saved in the '{plot_dir}/' directory!")

if __name__ == "__main__":
    run_evaluation()