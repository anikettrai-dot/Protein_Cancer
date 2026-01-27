import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary

# ======================================================
# CNN MODEL (Keras-style Sequential Equivalent)
# ======================================================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # ---------------- Feature Extraction ----------------
        # Input: (3, 32, 32)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.relu3 = nn.ReLU()

        # ---------------- Classification ----------------
        self.flatten = nn.Flatten()

        # 64 × 4 × 4 = 1024
        self.fc1 = nn.Linear(1024, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.dropout(self.fc1(x))
        x = self.fc2(x)
        return x


# ======================================================
# MODEL INITIALIZATION
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

print("✅ Model Initialized Successfully\n")

# ======================================================
# KERAS-LIKE MODEL SUMMARY (THIS IS WHAT YOU WANT)
# ======================================================
summary(
    model,
    input_size=(1, 3, 32, 32),  # (batch, channels, height, width)
    col_names=("input_size", "output_size", "num_params"),
    col_width=20,
    row_settings=("var_names",)
)

# ======================================================
# LOSS & OPTIMIZER (Equivalent to compile())
# ======================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

print("\n✅ Model Compiled Successfully")

# ======================================================
# OUTPUT SHAPE CHECK
# ======================================================
dummy_input = torch.randn(1, 3, 32, 32).to(device)
output = model(dummy_input)

print("\n🔍 Shape Verification")
print(f"Input Shape : {dummy_input.shape}")
print(f"Output Shape: {output.shape}  (10 classes)")
