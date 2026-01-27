import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import timm
import os

# ================== CONFIG ==================
DATA_DIR = r"D:\Protein_Cancer\Protein_Project\dataset_images"
MODEL_DIR = r"D:\Protein_Cancer\Protein_Project\models_3"
IMG_SIZE = 299
BATCH_SIZE = 32
THRESHOLD = 0.30   # Same threshold used in training
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Evaluating on: {device}")

# ---------- DATALOADER ----------
def get_val_loader():
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_DIR)

    # Use SAME split as training
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    val_ds.dataset.transform = transform
    loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return loader, dataset.classes


# ---------- MODEL LOADERS ----------
def load_model(name):
    print(f"\n🏗 Loading Model: {name}...")

    if name == "efficientnet_b4":
        model = models.efficientnet_b4(weights=None)
        model.classifier[1] = nn.Linear(1792, 1)

    elif name == "densenet201":
        model = models.densenet201(weights=None)
        model.classifier = nn.Linear(1920, 1)

    elif name == "seresnet50":
        model = timm.create_model('seresnet50', pretrained=False, num_classes=1)

    else:
        print("❌ Unknown model name.")
        return None

    model_path = os.path.join(MODEL_DIR, f"{name}_best.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded weights from {model_path}")
    else:
        print(f"❌ Model weights not found: {model_path}")
        return None

    return model.to(device)


# ---------- EVALUATE FUNCTION ----------
def evaluate_model(model_name):
    model = load_model(model_name)
    if model is None:
        return

    loader, class_names = get_val_loader()
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    print(f"🚀 Evaluating {model_name}...")

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > THRESHOLD).astype(int)

            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, pos_label=0)
    f1 = f1_score(y_true, y_pred, pos_label=0)

    print("\n" + "="*40)
    print(f"📊 REPORT: {model_name}")
    print("="*40)
    print(f"Accuracy: {acc:.4f}")
    print(f"Recall (Cancer=0): {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()
    print(f"📌 Saved: {model_name}_confusion_matrix.png")

    # ROC Curve & AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_name}_roc_curve.png')
    plt.close()
    print(f"📌 Saved: {model_name}_roc_curve.png")
    print("✔ Done\n")

# ---------- MAIN ----------
if __name__ == "__main__":
    # Inception_v3 removed from this list
    for model_name in ["efficientnet_b4", "densenet201", "seresnet50"]:
        evaluate_model(model_name)

    print("🎉 All Models Evaluation Completed!")