"""
Scores the OLD model on the test split, so the resume can show a real
before-and-after instead of a guess.

Run this after training finishes (it uses the same data folder).
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

DATA_DIR = "./data"
OLD_MODEL = "./model/flower_classifier_old.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_data = datasets.Flowers102(DATA_DIR, split="test", transform=eval_transforms, download=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 102)
model.load_state_dict(torch.load(OLD_MODEL, map_location=device))
model = model.to(device)
model.eval()

correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        _, predicted = torch.max(model(images), 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Old model test accuracy: {100 * correct / total:.1f}%  ({correct}/{total} correct)")