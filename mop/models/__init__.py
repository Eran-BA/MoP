"""
Model implementations for MoP (Mixture of Products)

This module contains the core model implementations including:
- ViT_MoP: Vision Transformer with Mixture of Products (first application)
- ViT_Baseline: Standard Vision Transformer baseline
- Core MoP components: ViewsLinear, Kernels3, FuseExcInh, etc.

The MoP mechanism is architecture-agnostic and can be adapted to:
- GPT models for sequential token interactions
- Audio Transformers for temporal-spectral patterns  
- Any Transformer architecture for general feature gating
"""

from .vit_mop import ViT_MoP
from .vit_baseline import ViT_Baseline
from .components import (
    ViewsLinear, 
    Kernels3, 
    FuseExcInh,
    ViTEncoder,
    PatchEmbed,
    MSA,
    MLP,
    Block,
    DropPath
)

__all__ = [
    # Main models (ViT application of MoP)
    "ViT_MoP", 
    "ViT_Baseline",
    
    # Core MoP components (architecture-agnostic)
    "ViewsLinear", 
    "Kernels3", 
    "FuseExcInh",
    
    # Transformer components (reusable)
    "ViTEncoder",
    "PatchEmbed",
    "MSA", 
    "MLP",
    "Block",
    "DropPath",
]