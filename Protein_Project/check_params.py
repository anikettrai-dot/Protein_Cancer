import torch
import torch.nn as nn
from torchvision import models
import timm
import os

# ==== CONFIG ====
MODEL_DIR = r"D:\Protein_Cancer\Protein_Project\models_3"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_model(name):
    print(f"\n🔍 Checking {name}...")
    
    # 1. Rebuild Architecture (Must match training exactly)
    try:
        if name == "densenet201":
            model = models.densenet201(weights=None)
            model.classifier = nn.Linear(1920, 1)
        elif name == "efficientnet_b4":
            model = models.efficientnet_b4(weights=None)
            model.classifier[1] = nn.Linear(1792, 1)
        elif name == "seresnet50":
            model = timm.create_model("seresnet50", pretrained=False, num_classes=1)
        elif name == "convnext_base":
            model = timm.create_model("convnext_base", pretrained=False, num_classes=1)
        else:
            print("Unknown model name")
            return

        # 2. Load Weights (Optional validation)
        path = os.path.join(MODEL_DIR, f"{name}_best.pth")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"✅ Weights loaded from {path}")
        else:
            print(f"⚠️ Weights file not found (Counting initialized architecture only)")

        # 3. Count
        total_params = count_parameters(model)
        print(f"📊 Total Trainable Parameters: {total_params:,}")
        print(f"   ({total_params / 1e6:.2f} Million)")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_model("densenet201")
    check_model("efficientnet_b4")
    check_model("seresnet50")
    # check_model("convnext_base") # Uncomment if you used this