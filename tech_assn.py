from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from PIL import Image
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models import ResNet34_Weights, resnet34

DATASET_ID = "doron333/change-detection-dataset"
SPLITS = ("train", "val", "test")
FULL_PREPROCESS_TRAIN = True
FULL_PREPROCESS_VAL = True
FULL_PREPROCESS_TEST = True
PREVIEW_SAMPLES_PER_SPLIT = 8
AUGMENT_TRAIN = True
AUGMENT_VAL = False
AUGMENT_TEST = False
USE_CUDA_IF_AVAILABLE = False
THRESHOLD_TUNING_MAX_SAMPLES = None
PRETRAINED_ENCODER = True
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
TRAIN_BATCH_SIZE = 8
EARLY_STOPPING_PATIENCE = 3



def read_tiff_from_zip(archive_path: str, member_name: str) -> np.ndarray:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_name) as file_handle:
            image = Image.open(BytesIO(file_handle.read()))
            return np.array(image)


def build_split_index(split: str):
    archive_path = hf_hub_download(DATASET_ID, f"{split}.zip", repo_type="dataset")

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        pre_files = {
            Path(name).name: name
            for name in members
            if f"{split}/pre-event/" in name and name.lower().endswith(".tif")
        }
        post_files = {
            Path(name).name: name
            for name in members
            if f"{split}/post-event/" in name and name.lower().endswith(".tif")
        }
        target_files = {
            Path(name).name: name
            for name in members
            if f"{split}/target/" in name and name.lower().endswith(".tif")
        }

    common_ids = sorted(pre_files.keys() & post_files.keys() & target_files.keys())
    return [
        {
            "sample_id": sample_id,
            "archive_path": archive_path,
            "pre_image_path": pre_files[sample_id],
            "post_image_path": post_files[sample_id],
            "label_path": target_files[sample_id],
        }
        for sample_id in common_ids
    ]


def build_dataset_dict() -> DatasetDict:
    return DatasetDict(
        {
            split: Dataset.from_list(build_split_index(split))
            for split in SPLITS
        }
    )


def remap_labels(mask):

    mask = np.asarray(mask)
    binary = np.zeros(mask.shape[:2] if mask.ndim == 3 else mask.shape, dtype=np.uint8)
    binary[np.isin(mask, [2, 3])] = 1

    if not np.any(binary) and np.any(mask > 0):
        binary[mask > 0] = 1

    return binary


def normalize_modality_tensor(tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = tensor.mean(dim=(1, 2), keepdim=True)
    std = tensor.std(dim=(1, 2), keepdim=True).clamp_min(eps)
    return (tensor - mean) / std


def preprocess(example):
    pre_image = read_tiff_from_zip(example["archive_path"], example["pre_image_path"])
    post_image = read_tiff_from_zip(example["archive_path"], example["post_image_path"])
    label = read_tiff_from_zip(example["archive_path"], example["label_path"])

    example["pre_image"] = pre_image
    example["post_image"] = post_image
    example["label"] = label
    example["binary_mask"] = remap_labels(label)
    return example


def preprocess_split(split_ds: Dataset, full_preprocess: bool) -> Dataset:
    if full_preprocess:
        return split_ds

    preview_count = min(PREVIEW_SAMPLES_PER_SPLIT, len(split_ds))
    return split_ds.select(range(preview_count))


dataset = build_dataset_dict()

train_ds = preprocess_split(dataset["train"], FULL_PREPROCESS_TRAIN)
val_ds = preprocess_split(dataset["val"], FULL_PREPROCESS_VAL)
test_ds = preprocess_split(dataset["test"], FULL_PREPROCESS_TEST)

print(f"train: {len(train_ds)} samples (will be loaded on-demand from zip)")
print(f"val: {len(val_ds)} samples (will be loaded on-demand from zip)")
print(f"test: {len(test_ds)} samples (will be loaded on-demand from zip)")
print(
    f"full_preprocess(train/val/test)=({FULL_PREPROCESS_TRAIN}/{FULL_PREPROCESS_VAL}/{FULL_PREPROCESS_TEST})"
)


# Paired augmentation utilities
class PairedAugmentation:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, eo: Image.Image, sar: Image.Image, mask: Image.Image):
        # Spatial: horizontal flip
        if random.random() < self.p:
            eo = TF.hflip(eo)
            sar = TF.hflip(sar)
            mask = TF.hflip(mask)

        # Spatial: vertical flip
        if random.random() < self.p:
            eo = TF.vflip(eo)
            sar = TF.vflip(sar)
            mask = TF.vflip(mask)

        # Spatial: 90/180/270 rotation
        if random.random() < self.p:
            angle = random.choice([90, 180, 270])
            eo = TF.rotate(eo, angle)
            sar = TF.rotate(sar, angle)
            mask = TF.rotate(mask, angle)

        # Random crop and resize (zoom effect)
        if random.random() < self.p:
            i, j, h, w = T.RandomCrop.get_params(eo, output_size=(200, 200))
            eo = TF.resized_crop(eo, i, j, h, w, (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
            sar = TF.resized_crop(sar, i, j, h, w, (256, 256), interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.resized_crop(mask, i, j, h, w, (256, 256), interpolation=TF.InterpolationMode.NEAREST)

        # Photometric: small brightness/contrast jitter only on eo
        if random.random() < self.p:
            eo = TF.adjust_brightness(eo, random.uniform(0.8, 1.2))
            eo = TF.adjust_contrast(eo, random.uniform(0.8, 1.2))

        # SAR-like multiplicative noise applied only to `sar` (post image)
        if random.random() < 0.3:
            sar_arr = np.array(sar).astype(np.float32)
            noise = np.random.randn(*sar_arr.shape).astype(np.float32) * 0.05 * 255.0
            sar_arr = sar_arr + noise
            sar_arr = np.clip(sar_arr, 0, 255).astype(np.uint8)
            sar = Image.fromarray(sar_arr)

        return eo, sar, mask


# Step 5: Build a PyTorch Dataset
class ChangeDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset: Dataset, transform=None, augmentation: PairedAugmentation = None):
        self.data = hf_dataset
        self.transform = transform or T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load from zip on-demand (lazy loading)
        pre_image = read_tiff_from_zip(sample["archive_path"], sample["pre_image_path"])
        post_image = read_tiff_from_zip(sample["archive_path"], sample["post_image_path"])
        label = read_tiff_from_zip(sample["archive_path"], sample["label_path"])

        pre = np.asarray(pre_image).astype(np.uint8)
        post = np.asarray(post_image).astype(np.uint8)
        binary_mask = remap_labels(label)
        mask = np.asarray(binary_mask, dtype=np.float32)

        # Convert to PIL images for augmentation and transforms
        pre_pil = Image.fromarray(pre)
        if post.ndim == 2:
            post = np.stack([post, post, post], axis=-1)
        post_pil = Image.fromarray(post)
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

        # Apply paired augmentation if provided
        if self.augmentation is not None:
            pre_pil, post_pil, mask_pil = self.augmentation(pre_pil, post_pil, mask_pil)

        # Apply transforms to each modality independently, then normalize each one separately.
        pre_tensor = normalize_modality_tensor(self.transform(pre_pil))
        post_tensor = normalize_modality_tensor(self.transform(post_pil))

        # Ensure mask is resized to same size as transforms output (256x256)
        mask_arr = np.array(mask_pil, dtype=np.uint8)
        if mask_arr.max() > 1:
            mask_arr = (mask_arr > 127).astype(np.uint8)
        mask_arr = np.array(Image.fromarray(mask_arr).resize((256, 256), Image.NEAREST), dtype=np.float32)
        mask_arr = mask_arr / 1.0

        return pre_tensor, post_tensor, torch.from_numpy(mask_arr).unsqueeze(0)


# Create PyTorch datasets and dataloaders with split-wise augmentation control
train_aug = PairedAugmentation(p=0.5) if AUGMENT_TRAIN else None
val_aug = PairedAugmentation(p=0.5) if AUGMENT_VAL else None
test_aug = PairedAugmentation(p=0.5) if AUGMENT_TEST else None

train_pytorch_ds = ChangeDetectionDataset(train_ds, augmentation=train_aug)
val_pytorch_ds = ChangeDetectionDataset(val_ds, augmentation=val_aug)
test_pytorch_ds = ChangeDetectionDataset(test_ds, augmentation=test_aug)

train_loader = torch.utils.data.DataLoader(train_pytorch_ds, batch_size=8, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_pytorch_ds, batch_size=8, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_pytorch_ds, batch_size=8, shuffle=False)

print("\n=== Step 5: PyTorch Dataset Built ===")
print(f"PyTorch dataset size: {len(train_pytorch_ds)}")
print(f"augmentation(train/val/test)=({AUGMENT_TRAIN}/{AUGMENT_VAL}/{AUGMENT_TEST})")
print("EO/SAR normalization: independent per-sample standardization before shared encoder")

# Loss functions (Focal + Dice combined)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        # alpha weights positives, (1-alpha) weights negatives
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal = alpha_t * (1 - pt) ** self.gamma * bce
        return focal.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        intersection = (pred * target).sum(1)
        dice = (2 * intersection + self.smooth) / (pred.sum(1) + target.sum(1) + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(alpha=0.80, gamma=2.0)
        self.dice = DiceLoss(smooth=1.0)

    def forward(self, pred, target):
        return 0.5 * self.focal(pred, target) + 0.5 * self.dice(pred, target)


criterion = CombinedLoss()


def build_batch_from_dataset(dataset_obj, batch_size=4):
    pre_images = []
    post_images = []
    ys = []
    n = min(batch_size, len(dataset_obj))
    for i in range(n):
        pre_i, post_i, y_i = dataset_obj[i]
        pre_images.append(pre_i)
        post_images.append(post_i)
        ys.append(y_i)
    return torch.stack(pre_images, dim=0), torch.stack(post_images, dim=0), torch.stack(ys, dim=0)


def build_resnet34_backbone(pretrained: bool = True) -> nn.Module:
    if not pretrained:
        return resnet34(weights=None)

    try:
        return resnet34(weights=ResNet34_Weights.DEFAULT)
    except Exception as exc:
        print(f"Warning: pretrained ResNet34 weights unavailable ({exc}); using random initialization.")
        return resnet34(weights=None)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        backbone = build_resnet34_backbone(pretrained=pretrained)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return [x0, x1, x2, x3, x4]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class SiameseUNet(nn.Module):
    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained=pretrained_encoder)
        # Fuse concat(pre, post) at the bottleneck for richer change context
        self.bottleneck_proj = nn.Sequential(
            nn.Conv2d(512 * 2, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder4 = DecoderBlock(512, 256, 256, dropout=0.3)
        self.decoder3 = DecoderBlock(256, 128, 128, dropout=0.2)
        self.decoder2 = DecoderBlock(128, 64, 64, dropout=0.1)
        self.decoder1 = DecoderBlock(64, 64, 64, dropout=0.1)
        self.refine = ConvBlock(64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, pre, post):
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)

        diff_features = [torch.abs(pre_feat - post_feat) for pre_feat, post_feat in zip(pre_features, post_features)]

        # Concat pre+post at bottleneck so the decoder sees both images, not just their difference
        bottleneck = self.bottleneck_proj(torch.cat([pre_features[4], post_features[4]], dim=1))

        x = self.decoder4(bottleneck, diff_features[3])
        x = self.decoder3(x, diff_features[2])
        x = self.decoder2(x, diff_features[1])
        x = self.decoder1(x, diff_features[0])
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.refine(x)
        return self.head(x)


def binary_f1_score(targets: np.ndarray, preds: np.ndarray) -> float:
    tp = np.logical_and(targets == 1, preds == 1).sum()
    fp = np.logical_and(targets == 0, preds == 1).sum()
    fn = np.logical_and(targets == 1, preds == 0).sum()
    denom = (2 * tp + fp + fn)
    return float((2 * tp) / denom) if denom > 0 else 0.0


def find_best_threshold(model: nn.Module, val_dataset, device: torch.device, max_samples: int | None = None) -> float:
    model.eval()
    all_preds = []
    all_targets = []

    n_samples = len(val_dataset) if max_samples is None else min(max_samples, len(val_dataset))
    print(f"Threshold tuning on device={device} with {n_samples} val sample(s)")
    with torch.no_grad():
        for idx in range(n_samples):
            pre, post, mask = val_dataset[idx]
            pre = pre.unsqueeze(0).to(device)
            post = post.unsqueeze(0).to(device)
            logits = model(pre, post)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(mask.unsqueeze(0).numpy())
            print(f"  processed val sample {idx + 1}/{n_samples}")

    all_preds = np.concatenate(all_preds).ravel()
    all_targets = np.concatenate(all_targets).ravel().astype(np.uint8)

    best_threshold, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.01):
        preds_binary = (all_preds >= t).astype(np.uint8)
        f1 = binary_f1_score(all_targets, preds_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    print(f"Best threshold: {best_threshold:.2f} | F1: {best_f1:.4f}")
    return best_threshold

print("\n=== Step 6: Quick Loss Test ===")

try:
    batch_pre, batch_post, batch_mask = build_batch_from_dataset(train_pytorch_ds, batch_size=4)
    logits = torch.zeros_like(batch_mask)
    loss_val = criterion(logits, batch_mask)
    print(f"Combined loss on zero logits: {loss_val.item():.6f}")
except Exception as e:
    print("Loss test failed:", e)


if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def build_device_loader(dataset_obj, batch_size: int, shuffle: bool):
    return torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def compute_binary_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).int()
    targets = targets.int()

    tp = int(((preds == 1) & (targets == 1)).sum().item())
    tn = int(((preds == 0) & (targets == 0)).sum().item())
    fp = int(((preds == 1) & (targets == 0)).sum().item())
    fn = int(((preds == 0) & (targets == 1)).sum().item())

    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_one_epoch(model: nn.Module, loader, optimizer, criterion, device: torch.device):
    model.train()
    running_loss = 0.0

    for batch_idx, (pre, post, mask) in enumerate(loader, start=1):
        pre = pre.to(device)
        post = post.to(device)
        mask = mask.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(pre, post)
        loss = criterion(logits, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        print(f"  train batch {batch_idx}/{len(loader)} | loss={loss.item():.4f}")

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader, criterion, device: torch.device, threshold: float):
    model.eval()
    total_loss = 0.0
    totals = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for pre, post, mask in loader:
        pre = pre.to(device)
        post = post.to(device)
        mask = mask.to(device)

        logits = model(pre, post)
        loss = criterion(logits, mask)
        metrics = compute_binary_metrics(logits, mask, threshold)

        total_loss += loss.item()
        for key in totals:
            totals[key] += metrics[key]

    tp = totals["tp"]
    tn = totals["tn"]
    fp = totals["fp"]
    fn = totals["fn"]
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * tp) / max(2 * tp + fp + fn, 1)

    return {
        "loss": total_loss / max(len(loader), 1),
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def print_metrics(prefix: str, metrics: dict):
    print(
        f"{prefix} | loss={metrics['loss']:.4f} | IoU={metrics['iou']:.4f} | "
        f"Precision={metrics['precision']:.4f} | Recall={metrics['recall']:.4f} | F1={metrics['f1']:.4f}"
    )
    print(
        f"{prefix} confusion matrix: TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}"
    )


demo_model = SiameseUNet(pretrained_encoder=PRETRAINED_ENCODER).to(device)
optimizer = torch.optim.AdamW(demo_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
)

train_loader = build_device_loader(train_pytorch_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = build_device_loader(val_pytorch_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
test_loader = build_device_loader(test_pytorch_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

print("\n=== Step 7: Training Loop ===")
best_val_f1 = -1.0
best_state = None
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"Epoch {epoch}/{NUM_EPOCHS}")
    train_loss = train_one_epoch(demo_model, train_loader, optimizer, criterion, device)
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"  avg train loss: {train_loss:.4f} | lr={current_lr:.2e}")

    epoch_val_metrics = evaluate_model(demo_model, val_loader, criterion, device, threshold=0.5)
    print_metrics("  val@0.50", epoch_val_metrics)

    scheduler.step(epoch_val_metrics["f1"])

    if epoch_val_metrics["f1"] > best_val_f1:
        best_val_f1 = epoch_val_metrics["f1"]
        best_state = {key: value.detach().cpu().clone() for key, value in demo_model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"  Early stopping: no val F1 improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            break

if best_state is not None:
    demo_model.load_state_dict(best_state)
    torch.save(best_state, "siamese_unet_best.pt")
    print("Saved best checkpoint: siamese_unet_best.pt")

best_t = find_best_threshold(
    demo_model,
    val_pytorch_ds,
    device,
    max_samples=THRESHOLD_TUNING_MAX_SAMPLES,
)
print(f"Recommended inference threshold from val split: {best_t:.2f}")

print("\n=== Step 8: Final Evaluation ===")
val_metrics = evaluate_model(demo_model, val_loader, criterion, device, threshold=best_t)
test_metrics = evaluate_model(demo_model, test_loader, criterion, device, threshold=best_t)
print_metrics("Val", val_metrics)
print_metrics("Test", test_metrics)

