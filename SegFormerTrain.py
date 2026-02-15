import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
from torch import nn
from tqdm import tqdm
import evaluate 
import torch.nn.functional as F 
from torch.cuda.amp import autocast, GradScaler # KEY: Mixed Precision

# ==========================================
# 1. CONFIGURATION (The "Sweet Spot")
# ==========================================
CONFIG = {
    "ROOT_DIR": "folder/Offroad_Segmentation_Training_Dataset", 
    "NUM_CLASSES": 10,
    
    # --- PERFORMANCE TUNING ---
    "BATCH_SIZE": 2,        # High enough for speed, low enough to avoid OOM
    "ACCUM_STEPS": 8,       # 2 * 8 = Effective Batch Size of 16 (Professional Standard)
    "NUM_WORKERS": 4,       # 4 is optimal for Batch Size 2 (Prevents CPU overhead)
    
    "LR": 6e-5,
    "EPOCHS": 15,     
    "IMAGE_SIZE": (544, 960), 
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WEIGHT_DECAY": 0.01
}

ID2LABEL = {
    0: "trees", 1: "lush_bush", 2: "dry_grass", 3: "dry_bushes",
    4: "ground_clutter", 5: "flower", 6: "logs", 7: "rocks",
    8: "landscape", 9: "sky"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ==========================================
# 2. CUSTOM LOSS FUNCTION (Dice + CE)
# ==========================================
class DiceCELoss(nn.Module):
    def __init__(self, num_classes):
        super(DiceCELoss, self).__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        # 1. Cross Entropy
        ce_loss = self.ce(inputs, targets)

        # 2. Dice Loss
        inputs_soft = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (inputs_soft * targets_one_hot).sum(dim=(2, 3))
        union = inputs_soft.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice_score = (2. * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1 - dice_score.mean()

        return 0.5 * ce_loss + 0.5 * dice_loss

# ==========================================
# 3. DATASET CLASS
# ==========================================
class OffRoadDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform
        self.img_dir = os.path.join(root_dir, split, "Color_Images")
        self.mask_dir = os.path.join(root_dir, split, "Segmentation")
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.images)

    def convert_mask(self, mask):
        new_mask = np.zeros_like(mask, dtype=np.uint8) 
        new_mask[mask == 100]   = 0  
        new_mask[mask == 200]   = 1  
        new_mask[mask == 300]   = 2  
        new_mask[mask == 500]   = 3  
        new_mask[mask == 550]   = 4  
        new_mask[mask == 600]   = 5  
        new_mask[mask == 700]   = 6  
        new_mask[mask == 800]   = 7  
        new_mask[mask == 7100]  = 8  
        new_mask[mask == 10000] = 9  
        return new_mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = img_name.replace(".jpg", ".png").replace(".jpeg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if mask is None: raise ValueError(f"Mask not found: {mask_path}")
        mask = self.convert_mask(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()

# ==========================================
# 4. ROBUST AUGMENTATIONS
# ==========================================
def get_transforms(split="train"):
    if split == "train":
        return A.Compose([
            A.Resize(height=CONFIG["IMAGE_SIZE"][0], width=CONFIG["IMAGE_SIZE"][1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=1, p=0.3), # Critical for generalization
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=CONFIG["IMAGE_SIZE"][0], width=CONFIG["IMAGE_SIZE"][1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

# ==========================================
# 5. HIGH-PERFORMANCE TRAINING LOOP
# ==========================================
def train_and_validate():
    train_ds = OffRoadDataset(CONFIG["ROOT_DIR"], split="train", transform=get_transforms("train"))
    val_ds = OffRoadDataset(CONFIG["ROOT_DIR"], split="val", transform=get_transforms("val"))

    # Optimized Dataloaders: Pin Memory ON, Persistent Workers ON
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, 
        num_workers=CONFIG["NUM_WORKERS"], pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, 
        num_workers=CONFIG["NUM_WORKERS"], pin_memory=True, persistent_workers=True
    )

    metric = evaluate.load("mean_iou")

    print("[INFO] Loading SegFormer B4...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b4",
        num_labels=CONFIG["NUM_CLASSES"], 
        id2label=ID2LABEL, 
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model.to(CONFIG["DEVICE"])
    
    # OPTIONAL: If this STILL crashes, uncomment the line below. It saves memory at the cost of slight speed.
    # model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"], weight_decay=CONFIG["WEIGHT_DECAY"])
    
    # Scheduler: Decay LR over TOTAL STEPS (Epochs * Batches)
    total_steps = (len(train_loader) // CONFIG["ACCUM_STEPS"]) * CONFIG["EPOCHS"]
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=1.0)

    loss_fct = DiceCELoss(num_classes=CONFIG["NUM_CLASSES"])
    scaler = GradScaler() # Initializes Mixed Precision Scaler

    print(f"[INFO] STARTING TRAINING")
    print(f"| Physical Batch: {CONFIG['BATCH_SIZE']}")
    print(f"| Accumulation:   {CONFIG['ACCUM_STEPS']}")
    print(f"| Effective Batch: {CONFIG['BATCH_SIZE'] * CONFIG['ACCUM_STEPS']}")
    print(f"| Mode:           Mixed Precision (AMP)")

    for epoch in range(CONFIG["EPOCHS"]):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} [Train]")
        
        for step, (images, masks) in enumerate(pbar):
            images, masks = images.to(CONFIG["DEVICE"]), masks.to(CONFIG["DEVICE"])
            
            # 1. Forward Pass (AutoCast handles float16/float32 mixing)
            with autocast():
                outputs = model(images)
                logits = torch.nn.functional.interpolate(
                    outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = loss_fct(logits, masks)
                loss = loss / CONFIG["ACCUM_STEPS"] # Normalize loss

            # 2. Backward Pass (Scaled)
            scaler.scale(loss).backward()
            
            # 3. Optimizer Step (Accumulated)
            if (step + 1) % CONFIG["ACCUM_STEPS"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                lr_scheduler.step()
            
            train_loss += loss.item() * CONFIG["ACCUM_STEPS"]
            pbar.set_postfix(loss=loss.item() * CONFIG["ACCUM_STEPS"], lr=lr_scheduler.get_last_lr()[0])

        # --- VALIDATE ---
        model.eval()
        print(f"Validating...")
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} [Val]"):
                images, masks = images.to(CONFIG["DEVICE"]), masks.to(CONFIG["DEVICE"])
                
                # AMP for Validation speeds up inference significantly
                with autocast():
                    outputs = model(images)
                    logits = torch.nn.functional.interpolate(
                        outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                
                predictions = logits.float().argmax(dim=1)
                metric.add_batch(predictions=predictions.detach().cpu().numpy(), references=masks.detach().cpu().numpy())

        # --- METRICS & SAVE ---
        metrics = metric.compute(num_labels=CONFIG["NUM_CLASSES"], ignore_index=255, reduce_labels=False)
        
        print(f"\n--- EPOCH {epoch+1} RESULTS ---")
        print(f"Mean IoU: {metrics['mean_iou']:.4f}")
        for i, iou in enumerate(metrics['per_category_iou']):
            print(f"  {ID2LABEL[i]:<15}: {iou:.4f}")
        
        save_path = f"segformer_B4_HighRes_ep{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train_and_validate
