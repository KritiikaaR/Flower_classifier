import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json

# Load class names (102 flower types)
# Flowers102 uses integer labels 0-101; we'll map them to names
FLOWER_NAMES = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood",
    "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle",
    "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower",
    "great masterwort", "siam tulip", "lenten rose", "barberton daisy", "daffodil",
    "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold",
    "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
    "orange dahlia", "pink-yellow dahlia", "cautleya spicata", "japanese anemone",
    "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus",
    "bearded iris", "windflower", "tree poppy", "gazania", "azalea", "water lily",
    "rose", "thorn apple", "morning glory", "passion flower", "lotus",
    "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine",
    "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily",
    "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"
]

# Load model
device = torch.device("cpu")
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 102)
model.load_state_dict(torch.load("model/flower_classifier.pth", map_location=device))
model.eval()

# Same normalization as training
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict(image):
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Return top 5 predictions
    top5_probs, top5_ids = torch.topk(probabilities, 5)
    return {FLOWER_NAMES[idx]: float(prob) for idx, prob in zip(top5_ids, top5_probs)}

# Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a flower image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="🌸 Flower Classifier",
    description="Upload any flower image and the model will identify it from 102 possible species. Trained using transfer learning on ResNet18.",
    examples=[]   # you can add example image paths here later
)

if __name__ == "__main__":
    demo.launch()