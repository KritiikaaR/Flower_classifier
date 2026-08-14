"""
Flower species classifier — Streamlit interface.

Run locally with:  streamlit run streamlit_app.py
"""

from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

REPO_URL = "https://github.com/KritiikaaR/Flower_classifier"
EXAMPLES_DIR = Path("examples")
LOW_CONFIDENCE = 0.20

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
    """Loaded once and reused, instead of on every interaction."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 102)
    model.load_state_dict(torch.load("model/flower_classifier.pth", map_location="cpu"))
    model.eval()
    return model


def predict(image):
    with torch.no_grad():
        outputs = load_model()(transform(image).unsqueeze(0))
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_ids = torch.topk(probabilities, 5)
    return [(FLOWER_NAMES[int(i)], float(p)) for p, i in zip(top_probs, top_ids)]


st.set_page_config(page_title="Flower Classifier", page_icon="🌸")

st.title("🌸 Flower Species Classifier")
st.write(
    "Upload a photo of a flower and the model identifies it from 102 species. "
    "Built by fine-tuning ResNet18 — **88.2% accuracy** on 6,149 held-out test images."
)

# --- Input: examples or upload -------------------------------------------------
if "example_choice" not in st.session_state:
    st.session_state.example_choice = None

example_files = sorted(EXAMPLES_DIR.glob("*.jpg")) if EXAMPLES_DIR.is_dir() else []

if example_files:
    st.caption("Try an example:")
    columns = st.columns(len(example_files))
    for column, path in zip(columns, example_files):
        label = path.stem.replace("_", " ").title()
        if column.button(label, use_container_width=True):
            st.session_state.example_choice = path

uploaded = st.file_uploader("Or upload your own", type=["jpg", "jpeg", "png", "webp"])

# An upload always takes priority over a previously clicked example.
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.session_state.example_choice = None
elif st.session_state.example_choice is not None:
    image = Image.open(st.session_state.example_choice).convert("RGB")
else:
    image = None

# --- Output --------------------------------------------------------------------
if image is not None:
    results = predict(image)

    col_left, col_right = st.columns(2)
    with col_left:
        st.image(image, use_container_width=True)

    with col_right:
        st.subheader("Predictions")
        for name, probability in results:
            st.write(f"**{name}** — {probability * 100:.1f}%")
            st.progress(probability)

    if results[0][1] < LOW_CONFIDENCE:
        st.warning(
            f"The model isn't confident about this one (top guess at "
            f"{results[0][1] * 100:.1f}%). It works best on a clear, well-lit photo of a "
            "single bloom filling most of the frame. Screenshots, bouquets, and wide "
            "garden shots tend to confuse it, and species outside the 102 it was trained "
            "on will always come back uncertain."
        )
else:
    st.info("Pick an example above or upload an image to get started.")

# --- Explanation ---------------------------------------------------------------
with st.expander("How this works"):
    st.markdown(
        """
The model starts from **ResNet18** pretrained on ImageNet, with its final layer
replaced to output 102 flower classes instead of 1,000 general object classes.

Training runs in two stages:

1. **Head only** — the pretrained layers are frozen and only the new output layer
   trains. That layer starts from random weights, and letting it push large
   corrections back into the pretrained features damages them before it has
   learned anything useful.
2. **Full fine-tune** — every layer unfreezes and trains at one tenth of the
   first learning rate, so the pretrained features get adjusted rather than
   overwritten.

An earlier version trained all layers at a single high learning rate and reached
only **59%** accuracy, misclassifying even common flowers. The two-stage approach,
plus rotation, cropping, and color augmentation, raised that to **88.2%**.

Predictions rarely exceed about 30% confidence, which is expected here: the choice
is spread across 102 classes, and training used label smoothing, which deliberately
discourages the model from being overly certain. What matters is the gap between the
top prediction and the rest.

Trained on the [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
dataset — 8,189 images, 102 species.
        """
    )

st.caption(f"[View the code on GitHub]({REPO_URL})")