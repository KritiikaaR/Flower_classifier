"""
Flower species classifier — Streamlit interface.

Same model and preprocessing as app.py, rebuilt for Streamlit Community Cloud.
Run locally with:  streamlit run streamlit_app.py
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

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
    "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper",
    "blackberry lily",
]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
    """Loaded once and reused across reruns instead of on every interaction."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 102)
    model.load_state_dict(torch.load("model/flower_classifier.pth", map_location="cpu"))
    model.eval()
    return model


st.set_page_config(page_title="Flower Classifier", page_icon="🌸")

st.title("🌸 Flower Species Classifier")
st.write(
    "Upload a photo of a flower and the model will identify it from 102 species. "
    "Built by fine-tuning ResNet18 — 88.2% accuracy on 6,149 held-out test images."
)

model = load_model()

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")

    col_left, col_right = st.columns(2)
    with col_left:
        st.image(image, caption="Your image", use_container_width=True)

    with torch.no_grad():
        outputs = model(transform(image).unsqueeze(0))
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top5_probs, top5_ids = torch.topk(probabilities, 5)

    with col_right:
        st.subheader("Predictions")
        for prob, idx in zip(top5_probs, top5_ids):
            name = FLOWER_NAMES[int(idx)]
            st.write(f"**{name}** — {float(prob) * 100:.1f}%")
            st.progress(float(prob))

    st.caption(
        "Trained on the Oxford Flowers-102 dataset. Works best on a clear photo "
        "of a single bloom."
    )
else:
    st.info("Upload an image to get started.")
