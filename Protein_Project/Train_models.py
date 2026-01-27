import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import recall_score, accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import os
import timm 

# ==========================================
# 🔧 CONFIGURATION 
# ==========================================
DATA_DIR = r"D:\Protein_Cancer\Protein_Project\dataset_images" 
SAVE_DIR = r"D:\Protein_Cancer\Protein_Project\models_3"
os.makedirs(SAVE_DIR, exist_ok=True)

# Full Model List (Inception Removed, ConvNeXt Added)
MODEL_LIST = ["efficientnet_b4", "densenet201", "seresnet50", "convnext_base"]

BATCH_SIZE = 16
EPOCHS = 30
IMG_SIZE = 299
ACCUM_STEPS = 2
LR = 2e-4  
EARLY_STOP_PATIENCE = 7

# 🚨 THRESHOLD = 0.75 (High Recall Strategy)
# Since Cancer=0 and Normal=1:
# We require >75% probability of "Normal" to classify as Normal.
# Anything less confident (e.g., 60% Normal) is treated as Cancer.
THRESHOLD = 0.75  

USE_FOCAL_LOSS = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🔥 Using: {device}")

# ==========================================
# 1️⃣ CANCER-BIASED FOCAL LOSS
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        pt = torch.exp(-bce_loss)
        
        # 🚨 WEIGHT FIX: 
        # Target 0 (Cancer) -> 0.85 weight (Heavy Penalty)
        # Target 1 (Normal) -> 0.15 weight (Light Penalty)
        alpha = torch.where(targets == 0, 0.85, 0.15)
        
        loss = alpha * (1-pt)**self.gamma * bce_loss
        return loss.mean()

criterion = FocalLoss() if USE_FOCAL_LOSS else nn.BCEWithLogitsLoss()

# ==========================================
# 2️⃣ AUGMENTATION (Prevent Overfitting)
# ==========================================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomApply([
        transforms.RandomRotation(90),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ColorJitter(0.3, 0.3)
    ], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ==========================================
# 3️⃣ 30x BIASED SAMPLER
# ==========================================
full_ds = datasets.ImageFolder(DATA_DIR)
train_size = int(0.8 * len(full_ds))
val_size = len(full_ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])

train_ds.dataset.transform = train_tf
val_ds.dataset.transform = val_tf

# --- WEIGHT CALCULATION ---
train_targets = np.array(full_ds.targets)[train_ds.indices]
class_counts = np.bincount(train_targets)
print(f"📊 Class Counts in Train: {class_counts} (0=Cancer, 1=Non-Cancer)")

# 1. Base Inverse Weighting (Math based)
weights = 1.0 / class_counts

# 2. 🔥 MANUAL BOOST (The "Hard Negative Mining" Fix) 🔥
# Multiply Cancer (Index 0) by 2.5x ON TOP of the math weighting.
# This results in Cancer being seen ~30x more often relative to count.
weights[0] = weights[0] * 2.5 

sample_weights = torch.tensor([weights[t] for t in train_targets], dtype=torch.double)
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=0, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)

# ==========================================
# MODEL FACTORY
# ==========================================
def create_model(name):
    print(f"🏗️ Building {name}...")

    if name == "efficientnet_b4":
        model = models.efficientnet_b4(weights="IMAGENET1K_V1")
        model.classifier[1] = nn.Linear(1792, 1)

    elif name == "densenet201":
        model = models.densenet201(weights="IMAGENET1K_V1")
        model.classifier = nn.Linear(1920, 1)

    elif name == "seresnet50":
        model = timm.create_model("seresnet50", pretrained=True, num_classes=1)
        
    elif name == "convnext_base":
        model = timm.create_model("convnext_base", pretrained=True, num_classes=1)

    return model.to(device)

# ==========================================
# TRAINING LOOP
# ==========================================
def train_model(name):
    model = create_model(name)

    # Specific LR logic
    lr = 1.5e-4 if name == "efficientnet_b4" else LR
    if name == "convnext_base": lr = 5e-5 # SOTA model needs lower LR

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    best_recall = 0.0
    patience_counter = 0

    print(f"\n🚀 Training Started: {name}\n")

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"{name} | Epoch {epoch+1}/{EPOCHS}")
        optimizer.zero_grad()

        for i, (imgs, lbls) in enumerate(loop):
            # Float target for BCE/Focal Loss
            imgs, lbls = imgs.to(device), lbls.float().unsqueeze(1).to(device)

            with torch.cuda.amp.autocast():
                out = model(imgs)
                loss = criterion(out, lbls) / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            loop.set_postfix(loss=loss.item()*ACCUM_STEPS)

        scheduler.step()

        # === Validation === #
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                out = torch.sigmoid(model(imgs)).cpu().numpy()
                
                # 🚨 HIGH THRESHOLD LOGIC
                # Prob > 0.75 -> Normal (1)
                # Prob <= 0.75 -> Cancer (0)
                preds.extend(out > THRESHOLD)
                labels.extend(lbls.numpy())

        # Metrics
        acc = accuracy_score(labels, preds)
        
        # 🚨 Specific Cancer Recall (Class 0)
        labels_np = np.array(labels)
        preds_np = np.array(preds)
        cancer_mask = (labels_np == 0)
        
        if np.sum(cancer_mask) > 0:
            # We check how many Class 0 were correctly predicted as 0
            cancer_recall = np.mean(preds_np[cancer_mask] == 0)
        else:
            cancer_recall = 0.0

        print(f"📌 {name} | Acc: {acc:.4f} | 🛡️ Cancer Recall (Class 0): {cancer_recall:.4f}")

        # Save model based on Cancer Recall
        if cancer_recall > best_recall:
            best_recall = cancer_recall
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{name}_best.pth"))
            print(f"💾 Saved Best {name} | Recall: {best_recall:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("🛑 Early Stopping")
            break

    print(f"🎯 Final Best Cancer Recall for {name}: {best_recall:.4f}\n")

if __name__ == "__main__":
    for name in MODEL_LIST:
        train_model(name)