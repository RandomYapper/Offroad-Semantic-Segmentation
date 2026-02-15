import os
import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm
import evaluate 

# ==========================================
# 1. CONFIGURATION (Must Match Training!)
# ==========================================
CONFIG = {
    # CHANGE THIS to your actual path if needed
    "ROOT_DIR": "folder", 
    "NUM_CLASSES": 10,
    "BATCH_SIZE": 2,      
    "IMAGE_SIZE": (544, 960), 
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "MODEL_PATH": "segformer_B4_HighRes_ep8.pth" # <--- YOUR SAVED MODEL
}

ID2LABEL = {
    0: "trees", 1: "lush_bush", 2: "dry_grass", 3: "dry_bushes",
    4: "ground_clutter", 5: "flower", 6: "logs", 7: "rocks",
    8: "landscape", 9: "sky"
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# ==========================================
# 2. DATASET CLASS (Identical to Training)
# ==========================================
class OffRoadDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.transform = transform
        self.img_dir = os.path.join(root_dir, split, "Color_Images")
        self.mask_dir = os.path.join(root_dir, split, "Segmentation")
        
        # Check if folder exists to avoid crash if 'test' folder is missing
        if not os.path.exists(self.img_dir):
            print(f"WARNING: Directory not found: {self.img_dir}")
            self.images = []
        else:
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
        
        # Safe handling for missing masks
        if mask is None:
            # If mask missing, return dummy mask (for inference only)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = self.convert_mask(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.long()

# ==========================================
# 3. EVALUATION FUNCTION
# ==========================================
def run_evaluation(model, dataloader, dataset_name="Validation"):
    metric = evaluate.load("mean_iou")
    model.eval()
    
    print(f"\n[INFO] Starting evaluation on {dataset_name} set...")
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
            images, masks = images.to(CONFIG["DEVICE"]), masks.to(CONFIG["DEVICE"])
            
            # Forward Pass
            outputs = model(images)
            
            # Upscale logits to match original mask size
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            
            predictions = logits.argmax(dim=1)
            

            # ### <<< MODIFICATION START >>> ###
            # Convert Ground Clutter (ID 4) to Rocks (ID 7)
            predictions[predictions == 4] = 7 
            # ### <<< MODIFICATION END >>> ###


            # Add batch to metrics
            metric.add_batch(
                predictions=predictions.detach().cpu().numpy(), 
                references=masks.detach().cpu().numpy()
            )

    # Compute Metrics
    metrics = metric.compute(num_labels=CONFIG["NUM_CLASSES"], ignore_index=255, reduce_labels=False)
    
    print(f"\n--- {dataset_name.upper()} RESULTS ---")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean Accuracy: {metrics['mean_accuracy']:.4f}")
    print("-" * 30)
    print(f"{'Class':<15} | {'IoU':<10} | {'Accuracy':<10}")
    print("-" * 30)
    for i, iou in enumerate(metrics['per_category_iou']):
        acc = metrics['per_category_accuracy'][i]
        print(f"{ID2LABEL[i]:<15} | {iou:.4f}     | {acc:.4f}")
    print("-" * 30)

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
def main():
    # A. Define Transforms (Only Normalization/Resize for Testing)
    test_transform = A.Compose([
        A.Resize(height=CONFIG["IMAGE_SIZE"][0], width=CONFIG["IMAGE_SIZE"][1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # B. Initialize Datasets
    # Note: Ensure you have a folder named 'test' or change split="val" to whatever you want
    val_ds = OffRoadDataset(CONFIG["ROOT_DIR"], split="val", transform=test_transform)
    test_ds = OffRoadDataset(CONFIG["ROOT_DIR"], split="test", transform=test_transform) # Assuming 'test' folder exists

    val_loader = DataLoader(val_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=2)

    # C. Load Model Structure
    print("[INFO] Loading Model Architecture...")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b4",
        num_labels=CONFIG["NUM_CLASSES"], 
        id2label=ID2LABEL, 
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )
    
    # D. Load Your Weights
    print(f"[INFO] Loading Weights from {CONFIG['MODEL_PATH']}...")
    if os.path.exists(CONFIG['MODEL_PATH']):
        # We use strict=False sometimes if there are minor mismatches, but usually True is good
        state_dict = torch.load(CONFIG['MODEL_PATH'], map_location=CONFIG["DEVICE"])
        model.load_state_dict(state_dict)
        print("[INFO] Weights loaded successfully!")
    else:
        print(f"[ERROR] Model file {CONFIG['MODEL_PATH']} not found!")
        return

    model.to(CONFIG["DEVICE"])

    # E. Run Evaluations
    if len(val_ds) > 0:
        run_evaluation(model, val_loader, dataset_name="Validation")
    else:
        print("Skipping Validation: Dataset empty.")

    if len(test_ds) > 0:
        run_evaluation(model, test_loader, dataset_name="Test")
    else:
        print("Skipping Test: Dataset empty or 'test' folder does not exist.")

if __name__ == "__main__":
    main()
