# Bi-Temporal Change Detection — Siamese UNet with Shared ResNet-34 Encoder

Binary pixel-wise change detection from bi-temporal remote-sensing imagery.
A pre-event and post-event image pair is passed through a shared ResNet-34 encoder;
multi-scale absolute-difference features serve as decoder skip connections, and a
concatenation-and-projection bottleneck provides richer context at the deepest level.
Training uses a combined Focal + Dice loss to handle class imbalance, with threshold
tuning on the validation set to maximise F1.

---

## Requirements

- Python 3.13.2

```
datasets==4.8.5
huggingface_hub==1.14.0
matplotlib==3.10.9
numpy==2.4.4
Pillow==12.2.0
torch==2.11.0
torchvision==0.26.0
```

A `requirements.txt` is provided in the repository root.

---

## Environment Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate        # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

> If you see a PowerShell execution-policy error, run once:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## Dataset Structure

The dataset is downloaded automatically from Hugging Face Hub
(`doron333/change-detection-dataset`) on first run — no manual setup required.

The expected on-disk layout after the first run (cached by `huggingface_hub`):

```
~/.cache/huggingface/hub/
└── datasets--doron333--change-detection-dataset/
    └── snapshots/<hash>/
        ├── train.zip
        ├── val.zip
        └── test.zip
```

Each zip contains three sub-folders:

```
train.zip
├── train/pre-event/   *.tif   # RGB optical image before event
├── train/post-event/  *.tif   # SAR / optical image after event
└── train/target/      *.tif   # multi-class label mask
```

Labels are remapped to binary: classes 2 and 3 → change (1), all others → no-change (0).

---

## Training

```powershell
python tech_assn.py
```

Key hyper-parameters (edit at the top of `tech_assn.py`):

| Parameter | Default | Description |
|---|---|---|
| `NUM_EPOCHS` | 10 | Maximum training epochs |
| `LEARNING_RATE` | 3e-4 | Initial AdamW learning rate |
| `TRAIN_BATCH_SIZE` | 8 | Batch size |
| `EARLY_STOPPING_PATIENCE` | 3 | Epochs without val F1 improvement before stopping |
| `PRETRAINED_ENCODER` | True | Use ImageNet-pretrained ResNet-34 |
| `AUGMENT_TRAIN` | True | Enable paired spatial + photometric augmentation |
| `USE_CUDA_IF_AVAILABLE` | False | Set to `True` to train on GPU |

Training automatically:
- Downloads the dataset on first run
- Saves the best checkpoint (by validation F1) to `siamese_unet_best.pt`
- Tunes the classification threshold on the full validation set
- Reports IoU, Precision, Recall, F1, and confusion matrix on val and test splits

---

## Evaluation

Evaluation runs automatically at the end of `tech_assn.py`. To generate
qualitative prediction figures after training:

```powershell
# 3 success + 2 failure cases from the val split
python visualise.py

# Use the test split
python visualise.py --split test

# Override the classification threshold
python visualise.py --threshold 0.62

# Scan more samples to find better examples
python visualise.py --scan 80
```

Figures are saved to `figures/`:

```
figures/
├── success_1.png
├── success_2.png
├── success_3.png
├── failure_1.png
└── failure_2.png
```

Each figure is a four-panel row: **pre-event | post-event | ground truth | predicted probability map**.

---

## Model Weights

The best checkpoint is saved locally to `siamese_unet_best.pt` after training.

> **Public download:** _[Upload to HuggingFace Hub or Google Drive and paste the link here]_

To load the checkpoint manually:

```python
import torch
from tech_assn import SiameseUNet

model = SiameseUNet(pretrained_encoder=False)
model.load_state_dict(torch.load("siamese_unet_best.pt", map_location="cpu"))
model.eval()
```

---

## Results

Metrics reported at the threshold tuned on the validation set (τ = 0.62).

### Validation split

| IoU | Precision | Recall | F1 | Val Loss |
|---|---|---|---|---|
| 0.3173 | 0.3523 | 0.7615 | 0.4817 | 0.3991 |

**Confusion matrix (pixel counts):**

| TP | TN | FP | FN |
|---|---|---|---|
| 1,846,064 | 16,070,602 | 3,394,142 | 578,216 |

### Test split

| IoU | Precision | Recall | F1 |
|---|---|---|---|
| — | — | — | — |

> Test results pending completion of the full training run.

---

## Architecture

```
Pre-event  ──┐                                    ┌── |Δ| skip ──┐
             ├── Shared ResNet-34 (weights tied) ──               ├── Dec4 → Dec3 → Dec2 → Dec1 → Refine → Head
Post-event ──┘          └── Bottleneck (concat+proj 512ch) ──────┘
                              |Δ| skips at each resolution level
```

| Component | Detail |
|---|---|
| Encoder | ResNet-34, ImageNet pretrained, shared between both streams |
| Feature levels | Stem 64ch → Layer1 64ch → Layer2 128ch → Layer3 256ch → Layer4 512ch |
| Bottleneck | `concat(pre, post)` → Conv 1×1 → 512ch (richer than pure difference) |
| Skip connections | `\|pre_l − post_l\|` absolute difference at each level |
| Decoder | Bilinear upsample → concat skip → Conv-BN-ReLU × 2 + Dropout2d |
| Dropout rates | 0.3 → 0.2 → 0.1 → 0.1 (deep to shallow) |
| Loss | 0.5 × Focal (α=0.80, γ=2.0) + 0.5 × Dice |
| Optimiser | AdamW, weight decay 1e-4 |
| Scheduler | ReduceLROnPlateau (mode=max, patience=2, factor=0.5) |
| Threshold | Swept over [0.30, 0.71] on val set, best F1 selected |

---

## Citation / References

```
Daudt et al. (2018)     — Fully Convolutional Siamese Networks for Change Detection. ICIP.
He et al. (2016)        — Deep Residual Learning for Image Recognition. CVPR.
Ronneberger et al. (2015) — U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
Lin et al. (2017)       — Focal Loss for Dense Object Detection. ICCV.
Milletari et al. (2016) — V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV.
Loshchilov & Hutter (2018) — Decoupled Weight Decay Regularization. ICLR.
Dataset               — https://huggingface.co/datasets/doron333/change-detection-dataset
```
