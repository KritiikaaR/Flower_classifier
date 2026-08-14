# Flower Species Classifier

An image classifier that identifies 102 flower species from a photo, built by fine-tuning a pretrained ResNet18. Upload an image and the app returns the five most likely species with confidence scores.

**Test accuracy: 88.2%** on 6,149 held-out images.

---

## How it works

The model starts from ResNet18 pretrained on ImageNet, with the final layer replaced to output 102 flower classes instead of 1,000 general object classes. Training happens in two stages:

**Stage 1 — head only.** The pretrained layers are frozen and only the new output layer trains. The new layer starts from random weights, and letting it push large corrections back into the pretrained features damages them before it has learned anything useful.

**Stage 2 — full fine-tune.** All layers unfreeze and train at one tenth of the stage 1 learning rate, so the pretrained features get adjusted rather than overwritten.

The dataset provides only 10 training images per class, so training applies random cropping, horizontal flips, rotation, and color jitter to get more variation out of a small set.

## Results

| Version | Test accuracy |
|---|---|
| First attempt — all layers, single learning rate, 5 epochs | 59.0% |
| Two-stage training, 20 epochs, expanded augmentation | 88.2% |

The first version updated every layer at a learning rate of 1e-3 from the start, which overwrote the pretrained ImageNet features it was supposed to be building on. Common flowers were being misclassified as a result. Freezing the backbone for the first stage and dropping the fine-tuning learning rate to 1e-4 fixed it.

Model selection uses the validation split. The test split is evaluated once, at the end, after training is complete.

## Dataset

[Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) — 8,189 images across 102 flower species common in the UK. Downloads automatically on first run via `torchvision.datasets.Flowers102`.

| Split | Images |
|---|---|
| Train | 1,020 |
| Validation | 1,020 |
| Test | 6,149 |

## Running it

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Train the model (downloads the dataset on first run):

```bash
python train.py
```

Launch the web app:

```bash
python app.py
```

## Files

| File | Purpose |
|---|---|
| `train.py` | Two-stage training, saves the best checkpoint by validation accuracy |
| `eval_old.py` | Scores a saved model against the test split |
| `app.py` | Gradio interface for uploading a photo and viewing predictions |
| `model/flower_classifier.pth` | Trained weights |

## Built with

Python, PyTorch, torchvision, Gradio
