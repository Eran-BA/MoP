"""
Baseline Vision Transformer

Author: Eran Ben Artzy
"""

import torch
import torch.nn as nn

from .components import ViTEncoder


class ViT_Baseline(nn.Module):
    """
    Standard Vision Transformer baseline.

    Args:
        dim: Embedding dimension
        depth: Number of transformer layers
        heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        n_classes: Number of output classes
        drop_path: DropPath rate
        patch: Patch size
        img_size: Input image size
    """

    def __init__(
        self,
        dim=256,
        depth=6,
        heads=4,
        mlp_ratio=4.0,
        n_classes=10,
        drop_path=0.1,
        patch=4,
        img_size=32,
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} not divisible by heads {heads}"

        num_tokens = (img_size // patch) ** 2

        self.enc = ViTEncoder(
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            patch=patch,
            num_tokens=num_tokens,
        )

        self.cls = nn.Linear(dim, n_classes, bias=False)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            logits: Classification logits (B, n_classes)
        """
        tok, _ = self.enc(x)  # (B,N,D)
        pooled = tok.mean(dim=1)  # Global average pooling
        return self.cls(pooled)
