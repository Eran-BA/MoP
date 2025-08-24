#!/usr/bin/env python3
import argparse
import os
import random
import sys
from typing import Dict, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mop.models.components import ViTEncoder  # type: ignore


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_voc_bbox(target: Dict) -> Tuple[int, int, int, int, int, int]:
    ann = target["annotation"]
    W = int(ann["size"]["width"])  # original image size
    H = int(ann["size"]["height"])
    obj = ann.get("object", None)
    if obj is None:
        # Fallback dummy box
        return 0, 0, 1, 1, W, H
    if isinstance(obj, list):
        # choose largest area object
        best = None
        best_area = -1
        for o in obj:
            b = o["bndbox"]
            xmin, ymin = int(b["xmin"]), int(b["ymin"])
            xmax, ymax = int(b["xmax"]), int(b["ymax"])
            area = max(0, xmax - xmin) * max(0, ymax - ymin)
            if area > best_area:
                best_area = area
                best = (xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = best
    else:
        b = obj["bndbox"]
        xmin, ymin = int(b["xmin"]), int(b["ymin"])
        xmax, ymax = int(b["xmax"]), int(b["ymax"])
    return xmin, ymin, xmax, ymax, W, H


class VOCLocDataset(Dataset):
    def __init__(self, root: str, year: str, split: str, img_size: int, download: bool):
        self.ds = datasets.VOCDetection(root=root, year=year, image_set=split, download=download)
        self.img_size = int(img_size)
        self.tfm_img = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, target = self.ds[idx]
        xmin, ymin, xmax, ymax, W, H = _parse_voc_bbox(target)
        # Resize image to square and scale bbox accordingly
        img_resized = TF.resize(img, (self.img_size, self.img_size))
        sx = self.img_size / max(1, W)
        sy = self.img_size / max(1, H)
        x0 = xmin * sx
        y0 = ymin * sy
        x1 = xmax * sx
        y1 = ymax * sy
        # Normalize coords to [0,1]
        box = torch.tensor([x0 / self.img_size, y0 / self.img_size, x1 / self.img_size, y1 / self.img_size], dtype=torch.float32)
        return self.tfm_img(img_resized), box


class ViTLocHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.GELU(approximate="tanh"), nn.Linear(dim, 4, bias=True))

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        # tok: [B,N,D] → pooled [B,D]
        pooled = tok.mean(dim=1)
        pred = self.mlp(self.ln(pooled))
        return torch.sigmoid(pred)  # [B,4] normalized [0,1]


class ViTLocalizer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float, drop_path: float, patch: int, img_size: int):
        super().__init__()
        num_tokens = (img_size // patch) ** 2
        self.enc = ViTEncoder(dim=dim, depth=depth, heads=heads, mlp_ratio=mlp_ratio, drop_path=drop_path, patch=patch, num_tokens=num_tokens)
        self.head = ViTLocHead(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok, _ = self.enc(x)
        return self.head(tok)


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    # boxes normalized [x0,y0,x1,y1], shape [...,4]
    xA = torch.maximum(box1[..., 0], box2[..., 0])
    yA = torch.maximum(box1[..., 1], box2[..., 1])
    xB = torch.minimum(box1[..., 2], box2[..., 2])
    yB = torch.minimum(box1[..., 3], box2[..., 3])
    inter = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
    area1 = torch.clamp(box1[..., 2] - box1[..., 0], min=0) * torch.clamp(box1[..., 3] - box1[..., 1], min=0)
    area2 = torch.clamp(box2[..., 2] - box2[..., 0], min=0) * torch.clamp(box2[..., 3] - box2[..., 1], min=0)
    union = area1 + area2 - inter + 1e-6
    return inter / union


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    ious = []
    l1s = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        iou = bbox_iou(pred, yb)
        l1 = (pred - yb).abs().mean(dim=-1)
        ious.append(iou.mean().item())
        l1s.append(l1.mean().item())
    return float(np.mean(ious)), float(np.mean(l1s))


def main():
    ap = argparse.ArgumentParser(description="VOC07/12 single-object localization with ViT backbone")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--year", type=str, default="2007", choices=["2007", "2012"])
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--mlp_ratio", type=float, default=4.0)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_frac", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--out", type=str, default="results/voc_localization")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    # Splits: VOC2007 has train/val/test; VOC2012 lacks test
    if args.year == "2007":
        train_split, val_split, test_split = "train", "val", "test"
    else:
        train_split, val_split, test_split = "train", "val", "val"

    train_ds = VOCLocDataset(args.data_root, args.year, train_split, args.img_size, download=args.download)
    val_ds = VOCLocDataset(args.data_root, args.year, val_split, args.img_size, download=False)
    test_ds = VOCLocDataset(args.data_root, args.year, test_split, args.img_size, download=False)

    if args.tiny:
        train_ds = torch.utils.data.Subset(train_ds, range(min(2000, len(train_ds))))
        val_ds = torch.utils.data.Subset(val_ds, range(min(500, len(val_ds))))
        test_ds = torch.utils.data.Subset(test_ds, range(min(500, len(test_ds))))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=False)

    model = ViTLocalizer(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        drop_path=args.drop_path,
        patch=args.patch,
        img_size=args.img_size,
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    warmup_steps = int(args.warmup_frac * total_steps)
    sched1 = optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=max(1, warmup_steps))
    sched2 = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps))
    sch = optim.lr_scheduler.SequentialLR(opt, [sched1, sched2], milestones=[max(1, warmup_steps)])

    loss_fn = nn.SmoothL1Loss(beta=1.0)

    history = {"epoch": [], "val_iou": [], "val_l1": []}

    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            sch.step()
            step += 1
        if (epoch % max(1, args.eval_every)) == 0:
            viou, vl1 = evaluate(model, val_loader, device)
            print(f"epoch {epoch:02d} | val IoU={viou:.3f} | val L1={vl1:.3f}")
            history["epoch"].append(epoch)
            history["val_iou"].append(viou)
            history["val_l1"].append(vl1)

    # Final test
    tiou, tl1 = evaluate(model, test_loader, device)
    print(f"\n📏 Test set: IoU={tiou:.4f} | L1={tl1:.4f}")

    # Save CSVs
    csv_path = os.path.join(args.out, f"voc{args.year}_loc_summary.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,val_iou,val_l1\n")
        for i in range(len(history["epoch"])):
            f.write(f"{history['epoch'][i]},{history['val_iou'][i]:.6f},{history['val_l1'][i]:.6f}\n")
    test_csv = os.path.join(args.out, f"voc{args.year}_loc_test.csv")
    with open(test_csv, "w") as f:
        f.write("iou,l1\n")
        f.write(f"{tiou:.6f},{tl1:.6f}\n")
    print(f"Saved: {csv_path}\nSaved: {test_csv}")

    # Plot
    try:
        plt.figure(figsize=(6,4))
        plt.plot(history["epoch"], history["val_iou"], label="Val IoU")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(args.out, f"voc{args.year}_val_iou.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()


