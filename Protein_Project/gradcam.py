import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms
from PIL import Image
import timm
import os
import random

# --- CONFIG ---
DATA_DIR = r"D:\Protein_Cancer\Protein_Project\dataset_images\Cancer"
MODEL_DIR = r"D:\Protein_Cancer\Protein_Project\models_3"

PATH_DENSENET = os.path.join(MODEL_DIR, "densenet201_best.pth")
PATH_EFFNET   = os.path.join(MODEL_DIR, "efficientnet_b4_best.pth")
PATH_SERESNET = os.path.join(MODEL_DIR, "seresnet50_best.pth")

IMG_SIZE = 299
THRESHOLD = 0.50 # Threshold for Consensus
# --------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Running on: {device}")

# --- CUSTOM TARGETS FOR BINARY CLASSIFICATION ---
class CancerTarget:
    """Highlights regions that push the model towards Class 0 (Cancer)"""
    def __call__(self, model_output):
        return -1 * model_output # Maximizing negative output = Minimizing Logit (Class 0)

class NormalTarget:
    """Highlights regions that push the model towards Class 1 (Non-Cancer)"""
    def __call__(self, model_output):
        return model_output # Maximizing output = Maximizing Logit (Class 1)

def load_model_architecture(name, path):
    print(f"🏗️ Loading {name}...")
    if not os.path.exists(path):
        print(f"   ⚠️ Weights not found at {path} (Skipping)")
        return None, None

    if name == "DenseNet201":
        model = models.densenet201(weights=None)
        model.classifier = torch.nn.Linear(1920, 1)
        target_layer = [model.features[-1]]
    elif name == "EfficientNetB4":
        model = models.efficientnet_b4(weights=None)
        model.classifier[1] = torch.nn.Linear(1792, 1)
        target_layer = [model.features[-1]]
    elif name == "SE-ResNet50":
        model = timm.create_model('seresnet50', pretrained=False, num_classes=1)
        target_layer = [model.layer4[-1]]
    else:
        return None, None

    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        return model, target_layer
    except Exception as e:
        print(f"   ❌ Error loading {name}: {e}")
        return None, None

def generate_visualizations(model, target_layers, input_tensor, img_float):
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 1. Prediction Score
    with torch.no_grad():
        logits = model(input_tensor)
        prob_normal = torch.sigmoid(logits).item()
        prob_cancer = 1.0 - prob_normal # Since 0=Cancer, 1=Normal
    
    # 2. Positive Evidence (Cancer - Class 0)
    targets = [CancerTarget()]
    grayscale_cancer = cam(input_tensor=input_tensor, targets=targets)[0, :]
    heatmap_cancer = show_cam_on_image(img_float, grayscale_cancer, use_rgb=True)
    
    # 3. Negative Evidence (Normal - Class 1)
    targets = [NormalTarget()]
    grayscale_normal = cam(input_tensor=input_tensor, targets=targets)[0, :]
    heatmap_normal = show_cam_on_image(img_float, grayscale_normal, use_rgb=True)
    
    return heatmap_cancer, heatmap_normal, prob_cancer

def main():
    # Pick Image
    if not os.path.exists(DATA_DIR):
        print("❌ Data folder not found.")
        return
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.png')]
    if not files:
        print("❌ No images found.")
        return
        
    random_file = random.choice(files)
    img_path = os.path.join(DATA_DIR, random_file)
    print(f"📸 Analyzing: {random_file}")

    # Preprocess
    img_pil = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_float = np.float32(img_pil) / 255.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # Models to process
    model_configs = [
        ("DenseNet201", PATH_DENSENET),
        ("EfficientNetB4", PATH_EFFNET),
        ("SE-ResNet50", PATH_SERESNET)
    ]

    # Plotting Setup (3 Rows, 4 Columns)
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    
    # --- ROW 1: ORIGINAL IMAGE ---
    axs[0, 0].imshow(img_pil)
    axs[0, 0].set_title("Original Input", fontsize=14, fontweight='bold')
    axs[0, 0].axis('off')
    
    # Hide empty slots in Row 1
    axs[0, 1].axis('off'); axs[0, 2].axis('off'); axs[0, 3].axis('off')

    # Data for Bar Chart
    model_names = []
    cancer_probs = []

    # --- PROCESS MODELS ---
    for i, (name, path) in enumerate(model_configs):
        col = i + 1
        model, layers = load_model_architecture(name, path)
        
        if model:
            # Generate Maps AND Prediction
            cancer_map, normal_map, p_cancer = generate_visualizations(model, layers, input_tensor, img_float)
            
            # Store for graph
            model_names.append(name.replace("EfficientNet", "EffNet").replace("DenseNet", "Dense"))
            cancer_probs.append(p_cancer)

            # Row 2: Cancer Evidence
            axs[1, col].imshow(cancer_map)
            axs[1, col].set_title(f"{name}\nProb Cancer: {p_cancer:.1%}", fontsize=10, color='darkred')
            axs[1, col].axis('off')
            
            # Row 3: Healthy Evidence
            axs[2, col].imshow(normal_map)
            axs[2, col].set_title(f"{name}\nHealthy Signal", fontsize=10, color='darkgreen')
            axs[2, col].axis('off')
        else:
            axs[1, col].text(0.5, 0.5, "Model Missing", ha='center')
            axs[2, col].text(0.5, 0.5, "Model Missing", ha='center')

    # --- ROW 2, COL 0: PROBABILITY BAR CHART ---
    if model_names:
        colors = ['red' if p > THRESHOLD else 'green' for p in cancer_probs]
        bars = axs[1, 0].bar(model_names, cancer_probs, color=colors, alpha=0.7)
        axs[1, 0].set_ylim(0, 1.0)
        axs[1, 0].set_ylabel("Cancer Probability")
        axs[1, 0].set_title("Model Confidence Scores", fontsize=12, fontweight='bold')
        axs[1, 0].axhline(y=THRESHOLD, color='gray', linestyle='--', linewidth=1, label='Threshold')
        
        # Add labels on bars
        for bar, p in zip(bars, cancer_probs):
            height = bar.get_height()
            axs[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{p:.1%}', ha='center', va='bottom', fontsize=9)
    else:
        axs[1, 0].text(0.5, 0.5, "No Models Loaded", ha='center')
        axs[1, 0].axis('off')

    # --- ROW 3, COL 0: CONSENSUS SUMMARY ---
    if cancer_probs:
        avg_prob = sum(cancer_probs) / len(cancer_probs)
        verdict = "CANCER" if avg_prob > THRESHOLD else "NORMAL"
        bg_color = 'mistyrose' if verdict == "CANCER" else 'honeydew'
        text_color = 'darkred' if verdict == "CANCER" else 'darkgreen'
        
        axs[2, 0].text(0.5, 0.5, 
                       f"ENSEMBLE VERDICT:\n{verdict}\n\nAvg Confidence:\n{avg_prob:.1%}", 
                       ha='center', va='center', fontsize=16, fontweight='bold', color=text_color)
        axs[2, 0].set_facecolor(bg_color)
        axs[2, 0].set_xticks([])
        axs[2, 0].set_yticks([])
    else:
        axs[2, 0].axis('off')

    plt.tight_layout()
    plt.savefig("advanced_gradcam_with_graphs.png")
    print("✅ Saved 'advanced_gradcam_with_graphs.png'")
    plt.show()

if __name__ == "__main__":
    main()