"""
MAX PERFORMANCE Segmentation Training Script using DINOv2
Optimized for 24GB VRAM + 32 Core CPU

Optimizations:
1. Mixed Precision (AMP) for 2x speedup
2. torch.compile() for graph optimization
3. TF32 Precision for Ampere/Ada GPUs
4. Persistent Data Workers
5. Fixed Albumentations Tensor Bug
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import multiprocessing

# ---------------- HARDWARE OPTIMIZATIONS ---------------- #
# 1. Enable CuDNN Benchmarking (Best for fixed image sizes)
torch.backends.cudnn.benchmark = True

# 2. Enable TF32 for Ampere+ GPUs (RTX 30xx/40xx, A100, A10)
torch.set_float32_matmul_precision('medium')

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')

# ============================================================================
# Mask Conversion
# ============================================================================

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}
n_classes = len(value_map)

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return new_arr

# ============================================================================
# Dataset with Augmentation (FIXED)
# ============================================================================

class MaskDatasetAugmented(Dataset):
    def __init__(self, data_dir, image_size=(270, 476), augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.data_ids = os.listdir(self.image_dir)
        self.image_size = image_size
        self.augment = augment
        
        if augment:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.OneOf([
                    A.OpticalDistortion(distort_limit=0.1, p=1.0),
                    A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
                ], p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        transformed = self.transform(image=image, mask=mask)
        
        # 'image' is already a Tensor because of ToTensorV2()
        image = transformed['image']
        
        # FIX: Just cast it to long, do not use from_numpy()
        mask = transformed['mask'].long()

        return image, mask

# ============================================================================
# Improved Segmentation Head
# ============================================================================

class ImprovedSegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH, hidden_dim=256):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        self.refine1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        self.refine2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.GELU(),
            nn.Conv2d(hidden_dim // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        
        x = self.initial_conv(x)
        x = x + self.refine1(x)
        x = x + self.refine2(x)
        x = self.classifier(x)
        
        return x

# ============================================================================
# Combined Loss
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight=None, dice_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return (1 - self.dice_weight) * ce + self.dice_weight * dice

# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10):
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        pred_inds = pred == class_id
        target_inds = target == class_id

        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10):
    iou_scores = []
    all_class_iou = []

    model.eval()
    backbone.eval()
    
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            # OPTIMIZATION: Use Autocast during inference too
            with torch.amp.autocast('cuda'):
                output = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = model(output)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)

            iou, class_iou = compute_iou(outputs, labels, num_classes=num_classes)
            iou_scores.append(iou)
            all_class_iou.append(class_iou)

    model.train()
    return np.mean(iou_scores), np.nanmean(all_class_iou, axis=0)

# ============================================================================
# Plotting
# ============================================================================

def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Training and Validation mIoU')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    if 'final_class_iou' in history:
        valid_iou = [iou if not np.isnan(iou) else 0 for iou in history['final_class_iou']]
        plt.bar(range(n_classes), valid_iou)
        plt.xticks(range(n_classes), class_names, rotation=45, ha='right')
        plt.ylabel('IoU')
        plt.title('Final Per-Class IoU')
        plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train segmentation model with DINOv2 (Max Performance)')
    parser.add_argument('--data_dir', type=str, default='/home/teaching/dlv/Training/Offroad_Segmentation_Training_Dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./train_stats_dinov2_optimized',
                        help='Output directory')
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'],
                        help='DINOv2 model size')
    
    # INCREASED BATCH SIZE DEFAULT
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (Try 16 or 24 for 24GB VRAM)')
    # ADJUSTED ACCUMULATION
    parser.add_argument('--accumulation_steps', type=int, default=2, help='Steps to reach effective BS of 32+')
    
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, nargs=2, default=[266, 476], 
                        help='Image size (height width) - must be divisible by 14')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Using device: {device}")
    
    # ---------------- OPTIMIZATION: Check Cores ---------------- #
    num_cores = multiprocessing.cpu_count()
    # Use 16 workers to keep CPU pipe full (leave some overhead)
    num_workers = min(16, num_cores) 
    print(f" Using {num_workers} CPU workers")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate patch dimensions
    patch_size = 14
    h_patches = args.image_size[0] // patch_size
    w_patches = args.image_size[1] // patch_size

    # Create datasets
    print("Loading datasets...")
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    
    trainset = MaskDatasetAugmented(train_dir, image_size=tuple(args.image_size), augment=True)
    valset = MaskDatasetAugmented(val_dir, image_size=tuple(args.image_size), augment=False)
    
    # OPTIMIZATION: persistent_workers=True avoids respawning overhead
    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    
    print(f"Training samples: {len(trainset)}")
    print(f"Validation samples: {len(valset)}")

    # Load DINOv2 backbone
    print(f"Loading DINOv2-{args.model_size}...")
    backbone_archs = {"small": "vits14", "base": "vitb14", "large": "vitl14"}
    backbone_name = f"dinov2_{backbone_archs[args.model_size]}"
    backbone = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone = backbone.to(device)
    
    # Get embedding dimension
    with torch.no_grad():
        # Use Autocast here to match training dtype
        with torch.amp.autocast('cuda'):
            sample_img = torch.randn(1, 3, args.image_size[0], args.image_size[1]).to(device)
            output = backbone.forward_features(sample_img)["x_norm_patchtokens"]
            embed_dim = output.shape[2]
    
    print(f"Embedding dimension: {embed_dim}")

    # Create segmentation head
    segmentation_head = ImprovedSegmentationHead(
        in_channels=embed_dim,
        out_channels=n_classes,
        tokenW=w_patches,
        tokenH=h_patches,
        hidden_dim=256
    ).to(device)
    
    # ---------------- OPTIMIZATION: Compile Models ---------------- #
    # PyTorch 2.0+ Compiler for extra speed
    print(" Compiling models for speed (this takes a minute at start)...")
    try:
        backbone = torch.compile(backbone)
        segmentation_head = torch.compile(segmentation_head)
        print("✅ Models compiled successfully!")
    except Exception as e:
        print(f"⚠️ Compilation failed (ignoring): {e}")

    # Calculate class weights
    print("Computing class weights...")
    class_counts = np.zeros(n_classes)
    # Use a simplified loader for counting to be fast
    count_loader = DataLoader(trainset, batch_size=32, num_workers=num_workers, shuffle=False)
    
    for _, mask in tqdm(count_loader, desc="Counting classes"):
        unique, counts = np.unique(mask.numpy(), return_counts=True)
        for cls, cnt in zip(unique, counts):
            class_counts[cls] += cnt
    
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * n_classes
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Loss, Optimizer, Scaler
    criterion = CombinedLoss(weight=class_weights, dice_weight=0.4)
    
    # OPTIMIZATION: foreach=True for faster optimizer steps
    optimizer = optim.AdamW([
        {'params': backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': segmentation_head.parameters(), 'lr': args.lr}
    ], weight_decay=0.01, foreach=True)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # OPTIMIZATION: Initialize GradScaler for Mixed Precision
    scaler = torch.amp.GradScaler('cuda')

    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': []}
    best_val_iou = 0.0
    
    # Training loop
    print("\nStarting High-Performance Training...")
    print("=" * 80)

    for epoch in range(args.epochs):
        backbone.train()
        segmentation_head.train()
        train_losses = []
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for batch_idx, (imgs, labels) in enumerate(pbar):
            imgs, labels = imgs.to(device), labels.to(device)

            # ---------------- OPTIMIZATION: Mixed Precision ---------------- #
            with torch.amp.autocast('cuda'):
                output = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = segmentation_head(output)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                
                loss = criterion(outputs, labels)
                loss = loss / args.accumulation_steps
            
            # Scaler Backward
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.accumulation_steps == 0:
                # Unscale before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(segmentation_head.parameters(), max_norm=1.0)
                
                # Scaler Step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_losses.append(loss.item() * args.accumulation_steps)
            pbar.set_postfix({'loss': f"{loss.item() * args.accumulation_steps:.4f}"})

        # Validation phase
        val_losses = []
        backbone.eval()
        segmentation_head.eval()
        
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)

                # Validation also benefits from Autocast speed
                with torch.amp.autocast('cuda'):
                    output = backbone.forward_features(imgs)["x_norm_patchtokens"]
                    logits = segmentation_head(output)
                    outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                    loss = criterion(outputs, labels)
                
                val_losses.append(loss.item())

        # Metrics
        train_iou, _ = evaluate_metrics(segmentation_head, backbone, train_loader, device, num_classes=n_classes)
        val_iou, val_class_iou = evaluate_metrics(segmentation_head, backbone, val_loader, device, num_classes=n_classes)

        # Update history
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"  Train IoU:  {train_iou:.4f} | Val IoU:  {val_iou:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'head_state_dict': segmentation_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
                'class_iou': val_class_iou
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (Val IoU: {val_iou:.4f})")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'backbone_state_dict': backbone.state_dict(),
                'head_state_dict': segmentation_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    history['final_class_iou'] = val_class_iou
    save_training_plots(history, args.output_dir)
    
    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()