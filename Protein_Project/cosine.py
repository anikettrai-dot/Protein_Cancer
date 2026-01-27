import numpy as np
import matplotlib.pyplot as plt

# =============== Your Training Settings ===============
EPOCHS = 30
LR_DENSE = 2e-4
LR_EFF = 1.5e-4
LR_SE = 2e-4

# Cosine schedule formula (PyTorch behavior, eta_min = 0)
def cosine_lr(initial_lr, t, T_max):
    return initial_lr * (1 + np.cos(np.pi * t / T_max)) / 2

t = np.arange(EPOCHS)

lr_dense = cosine_lr(LR_DENSE, t, EPOCHS)
lr_eff   = cosine_lr(LR_EFF,   t, EPOCHS)
lr_se    = cosine_lr(LR_SE,    t, EPOCHS)

# =============== Plot Style ===============
plt.figure(figsize=(10,6))

# DenseNet
plt.scatter(t, lr_dense, c=lr_dense, cmap='viridis', s=60, label="DenseNet201 (2e-4)")
plt.plot(t, lr_dense, color='magenta', linewidth=2)

# EfficientNet
plt.scatter(t, lr_eff, c=lr_eff, cmap='plasma', s=60, label="EfficientNet-B4 (1.5e-4)")
plt.plot(t, lr_eff, color='orange', linewidth=2)

# SE-ResNet50
plt.scatter(t, lr_se, c=lr_se, cmap='cividis', s=60, label="SE-ResNet50 (2e-4)")
plt.plot(t, lr_se, color='green', linewidth=2)

# Labels & Title
plt.title("📉 Cosine Annealing Learning Rate Schedule", fontsize=18, weight='bold')
plt.xlabel("Epochs", fontsize=14)
plt.ylabel("Learning Rate", fontsize=14)

plt.colorbar(label="LR Magnitude")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()
