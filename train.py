"""
Flower species classifier — ResNet18 transfer learning on Oxford Flowers-102.

Two-phase training:
  Phase 1  freeze the pretrained backbone, train only the new classification head.
           The head starts from random weights; letting it update the backbone
           while it is still random is what destroys the pretrained features.
  Phase 2  unfreeze everything and fine-tune at a much lower learning rate so
           the pretrained filters are nudged, not overwritten.

Model selection uses the val split. The test split is touched exactly once,
at the end, for the number you report.
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

# --- Config ---
DATA_DIR = "./data"
MODEL_DIR = "./model"
BATCH_SIZE = 32

HEAD_EPOCHS = 8           # phase 1: head only
FINETUNE_EPOCHS = 12      # phase 2: whole network
HEAD_LR = 1e-3
FINETUNE_LR = 1e-4        # 10x lower — this is the key change

os.makedirs(MODEL_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# --- Data ---
# Stronger augmentation than before. With only 10 training images per class,
# augmentation is doing a lot of the work.
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_data = datasets.Flowers102(DATA_DIR, split="train", transform=train_transforms, download=True)
val_data = datasets.Flowers102(DATA_DIR, split="val", transform=eval_transforms, download=True)
test_data = datasets.Flowers102(DATA_DIR, split="test", transform=eval_transforms, download=True)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=0)

print(f"train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

# --- Model ---
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 102)
model = model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def run_phase(name, epochs, optimizer, scheduler, best_acc):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()

        val_acc = evaluate(val_loader)
        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{MODEL_DIR}/flower_classifier.pth")
            marker = "  <- saved"
        print(f"[{name}] epoch {epoch+1}/{epochs} | loss {running_loss/len(train_loader):.3f} "
              f"| val acc {val_acc:.1f}%{marker}")
    return best_acc


# --- Phase 1: head only ---
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.fc.parameters(), lr=HEAD_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HEAD_EPOCHS)
best_acc = run_phase("head", HEAD_EPOCHS, optimizer, scheduler, 0.0)

# --- Phase 2: full fine-tune at low LR ---
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)
best_acc = run_phase("finetune", FINETUNE_EPOCHS, optimizer, scheduler, best_acc)

# --- Final held-out evaluation ---
model.load_state_dict(torch.load(f"{MODEL_DIR}/flower_classifier.pth", map_location=device))
test_acc = evaluate(test_loader)

print(f"\nBest val accuracy:  {best_acc:.1f}%")
print(f"Test accuracy:      {test_acc:.1f}%   <- this is the number for your resume")