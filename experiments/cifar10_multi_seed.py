import argparse, random, os
import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def try_import_models():
    try:
        from mop import ViT_Baseline, ViT_MoP
    except Exception:
        from mop.models import ViT_Baseline, ViT_MoP
    return ViT_Baseline, ViT_MoP

def get_loaders(batch=256):
    train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    val   = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
    return DataLoader(train, batch_size=batch, shuffle=True, num_workers=2), \
           DataLoader(val,   batch_size=batch, shuffle=False, num_workers=2)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct=0; total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item()
        total += y.numel()
    return correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--out", type=str, default="results/cifar10")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ViT_Baseline, ViT_MoP = try_import_models()
    train_loader, val_loader = get_loaders(args.batch)

    accs_base=[]; accs_mop=[]
    for s in args.seeds:
        set_seed(s)
        base = ViT_Baseline(dim=256, depth=6, heads=4, n_classes=10).to(device)
        mop  = ViT_MoP(dim=256, depth=6, heads=4, n_classes=10, n_views=5, n_kernels=3).to(device)

        opt_b = optim.AdamW(base.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        opt_m = optim.AdamW(mop.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        steps=0
        while steps < args.steps:
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                # baseline
                base.train(); opt_b.zero_grad()
                loss_b = nn.functional.cross_entropy(base(xb), yb); loss_b.backward(); opt_b.step()
                # mop
                mop.train();  opt_m.zero_grad()
                loss_m = nn.functional.cross_entropy(mop(xb),  yb); loss_m.backward(); opt_m.step()

                steps += 1
                if steps >= args.steps: break

        a_b = evaluate(base, val_loader, device)
        a_m = evaluate(mop,  val_loader, device)
        accs_base.append(a_b); accs_mop.append(a_m)
        print(f"seed {s}: baseline={a_b:.4f}  mop={a_m:.4f}")

    # save simple CSV
    with open(os.path.join(args.out, "cifar10_acc.csv"), "w") as f:
        f.write("seed,baseline,mop\n")
        for s,(b,m) in enumerate(zip(accs_base, accs_mop)):
            f.write(f"{args.seeds[s]},{b:.4f},{m:.4f}\n")
    print("done.")

if __name__ == "__main__":
    main()
