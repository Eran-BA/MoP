import argparse, random, os
import numpy as np
import torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def try_import_models():
    try:
        from mop import ViT_Baseline, ViT_MoP
    except Exception:
        from mop.models import ViT_Baseline, ViT_MoP
    return ViT_Baseline, ViT_MoP

def get_loaders(batch=256, tiny=False, workers=2):
    tfm_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    tfm_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train = datasets.CIFAR10("./data", train=True,  download=True, transform=tfm_train)
    test  = datasets.CIFAR10("./data", train=False, download=True, transform=tfm_test)
    if tiny:
        # fast smoke: ~5k train / 1k test
        train = torch.utils.data.Subset(train, range(5000))
        test  = torch.utils.data.Subset(test,  range(1000))
    train_loader = DataLoader(train, batch_size=batch, shuffle=True,  num_workers=workers, pin_memory=False)
    test_loader  = DataLoader(test,  batch_size=batch, shuffle=False, num_workers=workers, pin_memory=False)
    return train_loader, test_loader

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
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=5e-2)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--tiny", action="store_true", help="use small subset for a quick smoke run")
    ap.add_argument("--out", type=str, default="results/cifar10")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    torch.set_float32_matmul_precision("high")  # helps on MPS
    print(f"Device: {device}")

    ViT_Baseline, ViT_MoP = try_import_models()
    train_loader, val_loader = get_loaders(args.batch, tiny=args.tiny, workers=2)

    def make_opt(m):
        opt = optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.steps, 1))
        return opt, sch

    accs_base=[]; accs_mop=[]
    for s in args.seeds:
        set_seed(s)
        # small-ish configs for quick runs
        base = ViT_Baseline(dim=256, depth=6, heads=4, n_classes=10).to(device)
        mop  = ViT_MoP(dim=256, depth=6, heads=4, n_classes=10, n_views=5, n_kernels=3).to(device)

        opt_b, sch_b = make_opt(base)
        opt_m, sch_m = make_opt(mop)

        steps=0
        base.train(); mop.train()
        it = iter(train_loader)

        while steps < args.steps:
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(train_loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)

            # baseline
            opt_b.zero_grad(set_to_none=True)
            loss_b = nn.functional.cross_entropy(base(xb), yb)
            loss_b.backward(); opt_b.step(); sch_b.step()

            # mop
            opt_m.zero_grad(set_to_none=True)
            loss_m = nn.functional.cross_entropy(mop(xb), yb)
            loss_m.backward(); opt_m.step(); sch_m.step()

            steps += 1
            if steps % max(args.eval_every,1) == 0 or steps == 1:
                a_b = evaluate(base, val_loader, device)
                a_m = evaluate(mop,  val_loader, device)
                print(f"step {steps:4d} | loss_b={loss_b.item():.3f} loss_m={loss_m.item():.3f} | "
                      f"acc_b={a_b:.3f} acc_m={a_m:.3f}")

        # final eval per seed
        a_b = evaluate(base, val_loader, device)
        a_m = evaluate(mop,  val_loader, device)
        accs_base.append(a_b); accs_mop.append(a_m)
        print(f"seed {s}: baseline={a_b:.4f}  mop={a_m:.4f}")

    # save CSV
    with open(os.path.join(args.out, "cifar10_acc.csv"), "w") as f:
        f.write("seed,baseline,mop\n")
        for s,(b,m) in enumerate(zip(accs_base, accs_mop)):
            f.write(f"{args.seeds[s]},{b:.4f},{m:.4f}\n")
    print("done.")

if __name__ == "__main__":
    main()
