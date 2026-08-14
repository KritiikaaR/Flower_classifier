import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
import os

# --- Config ---
DATA_DIR = "./data"
MODEL_DIR = "./model"
EPOCHS = 5
BATCH_SIZE = 32
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Data Loading ---
# These transforms normalize images the same way ResNet was originally trained
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Downloads automatically (~330MB)
train_data = datasets.Flowers102(DATA_DIR, split="train", transform=train_transforms, download=True)
val_data   = datasets.Flowers102(DATA_DIR, split="val",   transform=val_transforms,   download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data,   batch_size=BATCH_SIZE)

# --- Model ---
# Load pretrained ResNet18, replace the final layer for 102 flower classes
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 102)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
model = model.to(device)

# --- Training ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.3f} | Val Acc: {100*correct/total:.1f}%")

# Save model
torch.save(model.state_dict(), f"{MODEL_DIR}/flower_classifier.pth")
print("Model saved!")