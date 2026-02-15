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

# ==========================================
# 1. CONFIGURATION
# ==========================================
CONFIG = {
    #  PATHS
    "ROOT_DIR": "test",  # Use your test folder if you want to train on it (or change to 'train')
    "SPLIT_NAME": "Offroad_Segmentation_testImages", 
    
    #  MODEL PATHS
    "SEGFORMER_PATH": "sgfmr.pth", 
    "DINOV2_PATH": "dinov2.pth",
    "OUTPUT_PATH": "best_ensemble.pth",
    
    # SETTINGS
    "NUM_CLASSES": 10,
    "BATCH_SIZE": 4,      
    "LR": 1e-3,
    "EPOCHS": 5,
    "IMAGE_SIZE": (544, 960), 
    "DINO_SIZE": (266, 476), 
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "DINO_MODEL_TYPE": "base"
}

# ==========================================
# 2. THE UNIFIER (Logistic Regression)
# ==========================================
class LogisticUnifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Input: 20 Channels (10 from SegFormer + 10 from DINO)
        # Output: 10 Channels (Final Predictions)
        self.mixer = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1)
        
    def forward(self, seg_logits, dino_logits):
        x = torch.cat([seg_logits, dino_logits], dim=1)
        return self.mixer(x)

# ==========================================
# 3. HELPER: ALIGNMENT FUNCTION
# ==========================================
def align_dino_to_segformer(dino_logits):
    """
    Shifts DINO logits to match SegFormer class mapping.
    DINO (1=Trees) -> SegFormer (0=Trees)
    """
    new_logits = torch.full_like(dino_logits, -100.0) 
    
    # DINO -> SegFormer Mapping based on your configs
    mapping = {1:0, 2:1, 3:2, 4:3, 5:4, 6:6, 7:7, 8:8, 9:9}
    
    for src, dst in mapping.items():
        new_logits[:, dst] = dino_logits[:, src]
    
    return new_logits

# ==========================================
# 4. DATASET & MODELS
# ==========================================
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
        mask_name = img_name.replace(".jpg", ".png")
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_UNCHANGED)
        
        mask = self.convert_mask(mask)
        mask[mask == 4] = 7  # Force Clutter -> Rocks during training
        
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']
        return image, mask.long()

def clean_sd(sd):
    return {k[10:] if k.startswith('_orig_mod.') else k: v for k, v in sd.items()}

# ==========================================
# 5. TRAINING LOOP
# ==========================================
def train():
    # 1. Load Data
    print(" Loading Data...")
    tf = A.Compose([A.Resize(CONFIG["IMAGE_SIZE"][0], CONFIG["IMAGE_SIZE"][1]), A.Normalize(), ToTensorV2()])
    dataset = OffRoadDataset(CONFIG["ROOT_DIR"], CONFIG["SPLIT_NAME"], transform=tf)
    loader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True, num_workers=2)

    # 2. Load Models (Frozen)
    print(" Loading Backbones...")
    segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b4", num_labels=10, ignore_mismatched_sizes=True).to(CONFIG["DEVICE"])
    
    #  FIX: Robust Loading Logic
    ckpt = torch.load(CONFIG["SEGFORMER_PATH"], map_location=CONFIG["DEVICE"], weights_only=False)
    # Check if 'state_dict' key exists, otherwise assume the checkpoint IS the state_dict
    sd = ckpt['state_dict'] if (isinstance(ckpt, dict) and 'state_dict' in ckpt) else ckpt
    segformer.load_state_dict(clean_sd(sd), strict=False)
    print(" SegFormer Loaded!")

    dinov2 = DINOv2Wrapper(CONFIG["DINO_MODEL_TYPE"], CONFIG["NUM_CLASSES"], CONFIG["DINO_SIZE"]).to(CONFIG["DEVICE"])
    dkpt = torch.load(CONFIG["DINOV2_PATH"], map_location=CONFIG["DEVICE"], weights_only=False)
    
    if isinstance(dkpt, dict) and 'backbone_state_dict' in dkpt:
        dinov2.backbone.load_state_dict(clean_sd(dkpt['backbone_state_dict']))
        dinov2.head.load_state_dict(clean_sd(dkpt['head_state_dict']))
    else:
        dinov2.load_state_dict(clean_sd(dkpt.state_dict() if hasattr(dkpt, 'state_dict') else dkpt), strict=False)
    print(" DINOv2 Loaded!")

    # Freeze them! We only want to train the mixer.
    segformer.eval(); dinov2.eval()
    for p in segformer.parameters(): p.requires_grad = False
    for p in dinov2.parameters(): p.requires_grad = False

    # 3. Initialize Unifier
    print(" Initializing Logistic Regression Unifier...")
    unifier = LogisticUnifier(CONFIG["NUM_CLASSES"]).to(CONFIG["DEVICE"])
    optimizer = torch.optim.AdamW(unifier.parameters(), lr=CONFIG["LR"])
    criterion = nn.CrossEntropyLoss()

    # 4. Training Loop
    for epoch in range(CONFIG["EPOCHS"]):
        unifier.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']}")
        
        for images, masks in pbar:
            images, masks = images.to(CONFIG["DEVICE"]), masks.to(CONFIG["DEVICE"])
            
            with torch.no_grad():
                # A. Get SegFormer Logits
                out_seg = segformer(images).logits
                out_seg = F.interpolate(out_seg, size=CONFIG["IMAGE_SIZE"], mode='bilinear', align_corners=False)
                
                # B. Get DINOv2 Logits
                img_dino = F.interpolate(images, size=CONFIG["DINO_SIZE"], mode='bilinear', align_corners=False)
                out_dino = dinov2(img_dino)
                
                #  C. ALIGN DINO TO SEGFORMER
                out_dino = align_dino_to_segformer(out_dino)
                
                # D. Resize DINO to match SegFormer
                out_dino = F.interpolate(out_dino, size=CONFIG["IMAGE_SIZE"], mode='bilinear', align_corners=False)

            # E. Train the Mixer
            optimizer.zero_grad()
            final_logits = unifier(out_seg, out_dino)
            loss = criterion(final_logits, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    # 5. Save
    torch.save(unifier.state_dict(), CONFIG["OUTPUT_PATH"])
    print(f" Trained Unifier saved to {CONFIG['OUTPUT_PATH']}")

if __name__ == "__main__":
    train()
