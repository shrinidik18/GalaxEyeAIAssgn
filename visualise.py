"""
visualise.py — generate qualitative prediction figures for the report.

Run after training:
    python visualise.py                        # uses v2 checkpoint by default
    python visualise.py --model v1             # uses baseline checkpoint
    python visualise.py --split test           # visualise test set instead of val
    python visualise.py --threshold 0.62       # override threshold

Saves 5 figures to figures/:
    success_1.png, success_2.png, success_3.png
    failure_1.png, failure_2.png
Each figure is a 4-panel row: pre-event | post-event | ground truth | prediction.
"""

import argparse
from io import BytesIO
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from PIL import Image
import torchvision.transforms as T
from torchvision.models import ResNet34_Weights, resnet34

# ---------------------------------------------------------------------------
# Minimal model definitions (copied so this script is self-contained)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.encoder = nn.Sequential(*layers)  # matches checkpoint naming from tech_assn.py

    def forward(self, x):
        return self.encoder(x)


class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        bb = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu)
        self.maxpool = bb.maxpool
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


# ---------- Baseline decoder (no attention gates) ----------

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


# Original architecture: no bottleneck_proj, no attention gates
class SiameseUNetOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained=False)
        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128,  64,  64)
        self.decoder1 = DecoderBlock( 64,  64,  64)
        self.refine   = ConvBlock(64, 32)
        self.head     = nn.Conv2d(32, 1, 1)

    def forward(self, pre, post):
        pf = self.encoder(pre); qf = self.encoder(post)
        diff = [torch.abs(p - q) for p, q in zip(pf, qf)]
        x = self.decoder4(diff[4], diff[3])
        x = self.decoder3(x,       diff[2])
        x = self.decoder2(x,       diff[1])
        x = self.decoder1(x,       diff[0])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(self.refine(x))


# Updated v1: has bottleneck_proj, no attention gates
class SiameseUNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained=False)
        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(1024, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.decoder4 = DecoderBlock(512, 256, 256, dropout=0.3)
        self.decoder3 = DecoderBlock(256, 128, 128, dropout=0.2)
        self.decoder2 = DecoderBlock(128,  64,  64, dropout=0.1)
        self.decoder1 = DecoderBlock( 64,  64,  64, dropout=0.1)
        self.refine   = ConvBlock(64, 32)
        self.head     = nn.Conv2d(32, 1, 1)

    def forward(self, pre, post):
        pf = self.encoder(pre); qf = self.encoder(post)
        diff = [torch.abs(p - q) for p, q in zip(pf, qf)]
        bot  = self.bottleneck_proj(torch.cat([pf[4], qf[4]], dim=1))
        x = self.decoder4(bot,  diff[3])
        x = self.decoder3(x,    diff[2])
        x = self.decoder2(x,    diff[1])
        x = self.decoder1(x,    diff[0])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(self.refine(x))


# ---------- V2 decoder (attention gates) ----------

class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(g_ch,  inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.W_x = nn.Sequential(nn.Conv2d(x_ch,  inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g, x):
        g_up = F.interpolate(self.W_g(g), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * self.psi(F.relu(g_up + self.W_x(x), inplace=True))


class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.attn  = AttentionGate(in_ch, skip_ch, skip_ch // 2)
        self.block = ConvBlock(in_ch + skip_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        skip = self.attn(x, skip)
        x    = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class SiameseUNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained=False)
        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(1024, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.decoder4 = AttentionDecoderBlock(512, 256, 256, dropout=0.3)
        self.decoder3 = AttentionDecoderBlock(256, 128, 128, dropout=0.2)
        self.decoder2 = AttentionDecoderBlock(128,  64,  64, dropout=0.1)
        self.decoder1 = AttentionDecoderBlock( 64,  64,  64, dropout=0.1)
        self.refine   = ConvBlock(64, 32)
        self.head     = nn.Conv2d(32, 1, 1)

    def forward(self, pre, post):
        pf = self.encoder(pre); qf = self.encoder(post)
        diff = [torch.abs(p - q) for p, q in zip(pf, qf)]
        bot  = self.bottleneck_proj(torch.cat([pf[4], qf[4]], dim=1))
        x = self.decoder4(bot,  diff[3])
        x = self.decoder3(x,    diff[2])
        x = self.decoder2(x,    diff[1])
        x = self.decoder1(x,    diff[0])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(self.refine(x))


# ---------------------------------------------------------------------------
# Data helpers (minimal — no augmentation)
# ---------------------------------------------------------------------------

DATASET_ID = "doron333/change-detection-dataset"

def read_tiff_from_zip(archive_path, member_name):
    with zipfile.ZipFile(archive_path) as arc:
        with arc.open(member_name) as fh:
            return np.array(Image.open(BytesIO(fh.read())))


def remap_labels(mask):
    mask = np.asarray(mask)
    binary = np.zeros(mask.shape[:2] if mask.ndim == 3 else mask.shape, dtype=np.uint8)
    binary[np.isin(mask, [2, 3])] = 1
    if not np.any(binary) and np.any(mask > 0):
        binary[mask > 0] = 1
    return binary


def normalize_tensor(t, eps=1e-6):
    mean = t.mean(dim=(1, 2), keepdim=True)
    std  = t.std(dim=(1, 2),  keepdim=True).clamp_min(eps)
    return (t - mean) / std


def tensor_to_display(t):
    """Convert a normalised CHW tensor to a HWC uint8 array for display."""
    arr = t.permute(1, 2, 0).numpy()
    arr = arr - arr.min()
    denom = arr.max()
    if denom > 0:
        arr = arr / denom
    return (arr * 255).clip(0, 255).astype(np.uint8)


def build_split_index(split):
    archive_path = hf_hub_download(DATASET_ID, f"{split}.zip", repo_type="dataset")
    with zipfile.ZipFile(archive_path) as arc:
        members = arc.namelist()
        pre_f  = {Path(n).name: n for n in members if f"{split}/pre-event/"  in n and n.endswith(".tif")}
        post_f = {Path(n).name: n for n in members if f"{split}/post-event/" in n and n.endswith(".tif")}
        tgt_f  = {Path(n).name: n for n in members if f"{split}/target/"     in n and n.endswith(".tif")}
    ids = sorted(pre_f.keys() & post_f.keys() & tgt_f.keys())
    return [{"archive_path": archive_path, "pre": pre_f[i], "post": post_f[i], "label": tgt_f[i]} for i in ids]


def load_sample(info):
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    pre_arr  = read_tiff_from_zip(info["archive_path"], info["pre"]).astype(np.uint8)
    post_arr = read_tiff_from_zip(info["archive_path"], info["post"]).astype(np.uint8)
    label    = read_tiff_from_zip(info["archive_path"], info["label"])

    if post_arr.ndim == 2:
        post_arr = np.stack([post_arr, post_arr, post_arr], axis=-1)

    pre_pil  = Image.fromarray(pre_arr)
    post_pil = Image.fromarray(post_arr)

    pre_t  = normalize_tensor(transform(pre_pil))
    post_t = normalize_tensor(transform(post_pil))

    mask = remap_labels(label)
    mask_t = torch.from_numpy(
        np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST), dtype=np.float32)
    ).unsqueeze(0)

    return pre_t, post_t, mask_t, pre_arr, post_arr


# ---------------------------------------------------------------------------
# Metrics per sample
# ---------------------------------------------------------------------------

def sample_f1(prob: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    pred = (prob >= threshold).astype(np.uint8)
    tp = np.logical_and(pred == 1, gt == 1).sum()
    fp = np.logical_and(pred == 1, gt == 0).sum()
    fn = np.logical_and(pred == 0, gt == 1).sum()
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def save_prediction_figure(pre_arr, post_arr, gt_arr, prob_arr, threshold,
                            filename, title=""):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(title, fontsize=11, y=1.01)

    # Pre-event
    axes[0].imshow(pre_arr if pre_arr.ndim == 3 else pre_arr, cmap="gray" if pre_arr.ndim == 2 else None)
    axes[0].set_title("Pre-event", fontsize=10)
    axes[0].axis("off")

    # Post-event
    axes[1].imshow(post_arr if post_arr.ndim == 3 else post_arr, cmap="gray" if post_arr.ndim == 2 else None)
    axes[1].set_title("Post-event", fontsize=10)
    axes[1].axis("off")

    # Ground truth
    axes[2].imshow(gt_arr, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Ground Truth", fontsize=10)
    axes[2].axis("off")

    # Predicted probability map with threshold contour
    im = axes[3].imshow(prob_arr, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[3].contour(prob_arr, levels=[threshold], colors="blue", linewidths=0.8)
    axes[3].set_title(f"Prediction (τ={threshold:.2f})", fontsize=10)
    axes[3].axis("off")
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    # Overlay: show TP/FP/FN in a small legend
    pred_bin = (prob_arr >= threshold).astype(np.uint8)
    gt_bin   = gt_arr.astype(np.uint8)
    tp = int(np.logical_and(pred_bin == 1, gt_bin == 1).sum())
    fp = int(np.logical_and(pred_bin == 1, gt_bin == 0).sum())
    fn = int(np.logical_and(pred_bin == 0, gt_bin == 1).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    axes[3].set_xlabel(f"TP={tp:,}  FP={fp:,}  FN={fn:,}  F1={f1:.3f}  IoU={iou:.3f}",
                       fontsize=8)

    plt.tight_layout()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filename}  (F1={f1:.3f}, IoU={iou:.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="v2",  choices=["v1", "v2"])
    parser.add_argument("--split",     default="val", choices=["val", "test"])
    parser.add_argument("--threshold", default=None,  type=float)
    parser.add_argument("--n_success", default=3,     type=int)
    parser.add_argument("--n_failure", default=2,     type=int)
    parser.add_argument("--scan",      default=40,    type=int,
                        help="How many samples to scan when finding success/failure cases")
    args = parser.parse_args()

    # --- pick checkpoint file ---
    ckpt_path = "siamese_unet_v2_best.pt" if args.model == "v2" else "siamese_unet_best.pt"
    default_threshold = 0.50 if args.model == "v2" else 0.62
    threshold = args.threshold if args.threshold is not None else default_threshold

    if not Path(ckpt_path).exists():
        # Fall back to the other checkpoint if the requested one is missing
        fallback = "siamese_unet_best.pt" if args.model == "v2" else "siamese_unet_v2_best.pt"
        if Path(fallback).exists():
            print(f"Warning: {ckpt_path} not found, falling back to {fallback}")
            ckpt_path = fallback
        else:
            raise FileNotFoundError(
                f"No checkpoint found. Run tech_assn.py or tech_assn_v2.py first."
            )

    device = torch.device("cpu")
    state  = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Auto-detect architecture from checkpoint keys
    has_attention  = any("attn" in k for k in state)
    has_bottleneck = any("bottleneck_proj" in k for k in state)

    if has_attention:
        ModelClass = SiameseUNetV2
        arch_name  = "V2 (attention gates + bottleneck)"
    elif has_bottleneck:
        ModelClass = SiameseUNetV1
        arch_name  = "V1-updated (bottleneck, no attention)"
    else:
        ModelClass = SiameseUNetOriginal
        arch_name  = "Original (no bottleneck, no attention)"

    model = ModelClass().to(device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {ckpt_path}  [{arch_name}]  |  threshold={threshold}  |  split={args.split}")

    # --- load split index ---
    index = build_split_index(args.split)
    n_scan = min(args.scan, len(index))
    print(f"Scanning {n_scan} samples to find success/failure cases…")

    scored = []
    with torch.no_grad():
        for i in range(n_scan):
            pre_t, post_t, mask_t, pre_arr, post_arr = load_sample(index[i])
            logits = model(pre_t.unsqueeze(0).to(device),
                           post_t.unsqueeze(0).to(device))
            prob  = torch.sigmoid(logits).squeeze().cpu().numpy()
            gt    = mask_t.squeeze().numpy()
            f1    = sample_f1(prob, gt, threshold)
            n_pos = int(gt.sum())
            scored.append({"idx": i, "f1": f1, "n_pos": n_pos,
                           "pre_arr": pre_arr, "post_arr": post_arr,
                           "gt": gt, "prob": prob})
            if (i + 1) % 10 == 0:
                print(f"  scanned {i + 1}/{n_scan}")

    # Filter: only consider samples that have some positive pixels
    # (otherwise "no change" samples trivially get high F1 at nothing)
    has_change = [s for s in scored if s["n_pos"] > 50]
    no_change  = [s for s in scored if s["n_pos"] <= 50]

    has_change_sorted = sorted(has_change, key=lambda s: s["f1"], reverse=True)

    # Successes: highest F1 among samples that have change pixels
    successes = has_change_sorted[:args.n_success]
    # Failures: lowest F1 among samples that have change pixels
    failures  = has_change_sorted[-args.n_failure:]

    success_f1s = [f"{s['f1']:.3f}" for s in successes]
    failure_f1s = [f"{s['f1']:.3f}" for s in failures]
    print(f"\nTop {args.n_success} success cases (F1): {success_f1s}")
    print(f"Top {args.n_failure} failure cases (F1): {failure_f1s}")

    for i, s in enumerate(successes, 1):
        fname = f"figures/success_{i}.png"
        title = (f"Success case {i} — {args.split} sample #{s['idx']}  "
                 f"(F1={s['f1']:.3f}, {s['n_pos']:,} change pixels)")
        save_prediction_figure(
            s["pre_arr"], s["post_arr"], s["gt"], s["prob"],
            threshold, fname, title=title
        )

    for i, s in enumerate(failures, 1):
        fname = f"figures/failure_{i}.png"
        title = (f"Failure case {i} — {args.split} sample #{s['idx']}  "
                 f"(F1={s['f1']:.3f}, {s['n_pos']:,} change pixels)")
        save_prediction_figure(
            s["pre_arr"], s["post_arr"], s["gt"], s["prob"],
            threshold, fname, title=title
        )

    print(f"\nAll figures saved to figures/  — include them in the LaTeX report.")


if __name__ == "__main__":
    main()
