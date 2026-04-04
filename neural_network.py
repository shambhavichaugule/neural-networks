import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image
import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

# Load MNIST
dataset = load_dataset("ylecun/mnist")
train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()

# Convert images to pixel arrays
def image_to_pixels(image_dict):
    img = Image.open(io.BytesIO(image_dict['bytes']))
    return np.array(img).flatten()

print("Loading data...")
X_train = np.array([image_to_pixels(img) for img in train_df['image']])
X_test = np.array([image_to_pixels(img) for img in test_df['image']])
y_train = train_df['label'].values
y_test = test_df['label'].values

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"Training samples: {X_train.shape}")
print(f"Test samples: {X_test.shape}")

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# Create dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

print("Data ready!")

# ── NEURAL NETWORK ARCHITECTURE ──────────────────────────
print("\n" + "="*50)
print("FEEDFORWARD NEURAL NETWORK")
print("="*50)

class FeedforwardNN(nn.Module):
    def __init__(self):
        super(FeedforwardNN, self).__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 2
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Layer 3
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output layer
            nn.Linear(128, 10)  # 10 classes (digits 0-9)
        )

    def forward(self, x):
        return self.network(x)

# Initialize model
model = FeedforwardNN().to(device)
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nTraining...")
epochs = 10
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = epoch_loss / len(train_loader)
    accuracy = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_pred = test_outputs.argmax(dim=1).cpu().numpy()

test_accuracy = accuracy_score(y_test, test_pred)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, test_pred))

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')

ax2.plot(train_accuracies)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy')

plt.tight_layout()
plt.savefig("training_curves.png")
print("\nTraining curves saved!")

# Visualize what the network learned
# Show some correct and incorrect predictions
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predictions = outputs.argmax(dim=1).cpu().numpy()

# Find incorrect predictions
incorrect_idx = np.where(predictions != y_test)[0]
correct_idx = np.where(predictions == y_test)[0]

fig, axes = plt.subplots(2, 10, figsize=(15, 4))

# Show 10 correct predictions
for i in range(10):
    idx = correct_idx[i]
    axes[0, i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
    axes[0, i].set_title(f"✓{predictions[idx]}", fontsize=8)
    axes[0, i].axis('off')

# Show 10 incorrect predictions
for i in range(10):
    idx = incorrect_idx[i]
    axes[1, i].imshow(X_test[idx].reshape(28, 28), cmap='gray')
    axes[1, i].set_title(f"✗{predictions[idx]}\n({y_test[idx]})", fontsize=8)
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Correct', fontsize=10)
axes[1, 0].set_ylabel('Wrong', fontsize=10)
plt.suptitle('FNN Predictions — Correct vs Incorrect')
plt.tight_layout()
plt.savefig("predictions.png")
print("Predictions chart saved!")

print(f"\nTotal errors: {len(incorrect_idx)} out of 10,000")
print(f"Error rate: {len(incorrect_idx)/10000:.2%}")

# Compare activation functions
class FNN_Sigmoid(nn.Module):
    def __init__(self):
        super(FNN_Sigmoid, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.Sigmoid(),           # replaced ReLU with Sigmoid
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# Train sigmoid model
model_sigmoid = FNN_Sigmoid().to(device)
optimizer_sigmoid = optim.Adam(model_sigmoid.parameters(), lr=0.001)

print("\nTraining with Sigmoid activation...")
for epoch in range(10):
    model_sigmoid.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        optimizer_sigmoid.zero_grad()
        outputs = model_sigmoid(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_sigmoid.step()

        epoch_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/10 | Loss: {epoch_loss/len(train_loader):.4f} | Accuracy: {correct/total:.4f}")

model_sigmoid.eval()
with torch.no_grad():
    sigmoid_pred = model_sigmoid(X_test_tensor).argmax(dim=1).cpu().numpy()

acc_sigmoid = accuracy_score(y_test, sigmoid_pred)

print(f"\nReLU accuracy:    {test_accuracy:.4f}")
print(f"Sigmoid accuracy: {acc_sigmoid:.4f}")
print(f"Difference:       {test_accuracy - acc_sigmoid:.4f}")

class FNN_GELU(nn.Module):
    def __init__(self):
        super(FNN_GELU, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

model_gelu = FNN_GELU().to(device)
optimizer_gelu = optim.Adam(model_gelu.parameters(), lr=0.001)

print("\nTraining with GELU activation...")
for epoch in range(10):
    model_gelu.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        optimizer_gelu.zero_grad()
        outputs = model_gelu(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_gelu.step()

        epoch_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/10 | Loss: {epoch_loss/len(train_loader):.4f} | Accuracy: {correct/total:.4f}")

model_gelu.eval()
with torch.no_grad():
    gelu_pred = model_gelu(X_test_tensor).argmax(dim=1).cpu().numpy()

acc_gelu = accuracy_score(y_test, gelu_pred)

print(f"\nReLU accuracy:    {test_accuracy:.4f}")
print(f"Sigmoid accuracy: {acc_sigmoid:.4f}")
print(f"GELU accuracy:    {acc_gelu:.4f}")

# Model with Batch Normalisation
class FNN_BatchNorm(nn.Module):
    def __init__(self):
        super(FNN_BatchNorm, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),    # normalise after linear
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

model_bn = FNN_BatchNorm().to(device)
optimizer_bn = optim.Adam(model_bn.parameters(), lr=0.001)

print("\nTraining with Batch Normalisation...")
bn_losses = []

for epoch in range(10):
    model_bn.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        optimizer_bn.zero_grad()
        outputs = model_bn(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer_bn.step()

        epoch_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = epoch_loss / len(train_loader)
    bn_losses.append(avg_loss)

    if (epoch + 1) % 2 == 0:
        print(f"Epoch {epoch+1}/10 | Loss: {avg_loss:.4f} | Accuracy: {correct/total:.4f}")

model_bn.eval()
with torch.no_grad():
    bn_pred = model_bn(X_test_tensor).argmax(dim=1).cpu().numpy()

acc_bn = accuracy_score(y_test, bn_pred)

print(f"\nFinal Comparison:")
print(f"{'Model':<30} {'Accuracy':>10}")
print("-" * 42)
print(f"{'ReLU (baseline)':<30} {test_accuracy:>10.4f}")
print(f"{'Sigmoid':<30} {acc_sigmoid:>10.4f}")
print(f"{'GELU':<30} {acc_gelu:>10.4f}")
print(f"{'ReLU + BatchNorm':<30} {acc_bn:>10.4f}")
