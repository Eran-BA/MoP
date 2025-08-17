"""
ViT-MoP: Vision Transformer with Mixture of Products

Author: Eran Ben Artzy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import ViTEncoder, ViewsLinear, Kernels3, FuseExcInh


class ViT_MoP(nn.Module):
    """
    Vision Transformer with Mixture of Products.
    
    Adds spatial boolean logic through excitatory/inhibitory gating.
    
    Args:
        dim: Embedding dimension
        depth: Number of transformer layers
        heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        n_classes: Number of output classes
        n_views: Number of spatial views
        n_kernels: Number of learnable kernels
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
        n_views=5,
        n_kernels=3,
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
        
        self.views = ViewsLinear(dim, n_views=n_views)
        self.kerns = Kernels3(in_ch=n_views, n_kernels=n_kernels)
        self.fuse = FuseExcInh(in_ch=n_views + n_kernels)
        self.cls = nn.Linear(dim, n_classes, bias=False)
        
        self.n_views = n_views
        self.n_kernels = n_kernels
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            logits: Classification logits (B, n_classes)
        """
        tok, grid = self.enc(x)  # (B,N,D), (Gh,Gw)
        B, N, D = tok.shape
        Gh, Gw = grid
        
        # Multi-view projection
        V = self.views(tok, grid)  # (B,V,Gh,Gw)
        
        # Learnable kernels
        K = self.kerns(V)  # (B,K,Gh,Gw)
        
        # Combine views and kernels
        maps = torch.cat([V, K], dim=1)  # (B,V+K,Gh,Gw)
        
        # Excitatory/inhibitory gating
        G_pos, G_neg, a_pos, a_neg = self.fuse(maps)
        gate = (1 + a_pos * G_pos - a_neg * G_neg)  # (B,1,Gh,Gw)
        
        # Apply gate to tokens
        gate_flat = gate.reshape(B, 1, N)  # (B,1,N)
        tok = tok.transpose(1, 2) * gate_flat  # (B,D,N)
        tok = tok.transpose(1, 2)  # (B,N,D)
        
        # Global average pooling + classification
        pooled = tok.mean(dim=1)  # (B,D)
        return self.cls(pooled)
    
    def get_gate_maps(self, x):
        """
        Extract spatial gate maps for visualization.
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            gate: Spatial gate maps (B, 1, H, W)
            views: Multi-view projections (B, V, H, W)
            kernels: Kernel responses (B, K, H, W)
        """
        with torch.no_grad():
            tok, grid = self.enc(x)
            V = self.views(tok, grid)
            K = self.kerns(V)
            maps = torch.cat([V, K], dim=1)
            G_pos, G_neg, a_pos, a_neg = self.fuse(maps)
            gate = (1 + a_pos * G_pos - a_neg * G_neg)
            
        return gate, V, K