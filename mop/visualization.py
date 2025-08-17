import os
import math
import matplotlib.pyplot as plt
import numpy as np
import torch

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

def visualize_gates(images, gates, views=None, kernels=None, save_path="outputs/attention_maps.png"):
    """
    images:  (B,3,H,W)  torch.Tensor
    gates:   (B,1,h,w)  torch.Tensor
    views:   (B,V,h,w)  torch.Tensor or None
    kernels: (B,K,h,w)  torch.Tensor or None
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    imgs  = _to_numpy(images)
    G     = _to_numpy(gates)
    V     = _to_numpy(views) if views is not None else None
    K     = _to_numpy(kernels) if kernels is not None else None

    B = imgs.shape[0]
    cols = 1 + (1 if V is not None else 0) + (1 if K is not None else 0)

    for b in range(B):
        fig = plt.figure(figsize=(12, 3.0))
        ax = fig.add_subplot(1, cols, 1)
        # show image
        img = np.clip(imgs[b].transpose(1,2,0), 0, 1)
        ax.imshow(img)
        ax.set_title("image")
        ax.axis("off")

        c = 2
        if V is not None:
            ax = fig.add_subplot(1, cols, c)
            ax.imshow(V[b].sum(0), interpolation="nearest")
            ax.set_title("views (sum)")
            ax.axis("off")
            c += 1
        if K is not None:
            ax = fig.add_subplot(1, cols, c)
            ax.imshow(K[b].sum(0), interpolation="nearest")
            ax.set_title("kernels (sum)")
            ax.axis("off")

        # overlay gates as a separate saved figure
        fig2 = plt.figure(figsize=(4,3))
        ax2 = fig2.add_subplot(1,1,1)
        ax2.imshow(G[b,0], interpolation="nearest")
        ax2.set_title("gates")
        ax2.axis("off")
        fig2.tight_layout()
        fig2.savefig(save_path.replace(".png", f".gates.{b}.png"), bbox_inches="tight")
        plt.close(fig2)

        fig.tight_layout()
        fig.savefig(save_path.replace(".png", f".sample.{b}.png"), bbox_inches="tight")
        plt.close(fig)

    # mosaic summary (first min(B,8))
    num = min(B, 8)
    cols = 4
    rows = int(math.ceil(num / cols))
    fig = plt.figure(figsize=(cols*3, rows*3))
    for i in range(num):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(G[i,0], interpolation="nearest")
        ax.set_title(f"gate {i}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
