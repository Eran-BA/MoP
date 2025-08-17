"""
Model implementations for MoP (Mixture of Products)

This module contains the core model implementations including:
- ViT_MoP: Vision Transformer with Mixture of Products (first application)
- ViT_Baseline: Standard Vision Transformer baseline
- GPT_MoP: Language model with Mixture of Products for sequential data
- Whisper_MoP: Audio transformer with Mixture of Products for temporal-spectral patterns
- Core MoP components: ViewsLinear, Kernels3, FuseExcInh, etc.

The MoP mechanism is architecture-agnostic and can be adapted to:
- GPT models for sequential token interactions
- Audio Transformers for temporal-spectral patterns  
- Any Transformer architecture for general feature gating
"""

# Vision Transformer implementations
from .vit_mop import ViT_MoP
from .vit_baseline import ViT_Baseline

# GPT/Language Model implementations
from .gpt_mop import GPT_MoP, create_gpt_mop, create_gpt_baseline, create_gpt_quartet
from .gpt_comparison import GPTComparisonFramework, ComparisonConfig, create_comparison_framework

# Whisper/Audio Transformer implementations
from .whisper_mop import WhisperMoP, create_whisper_mop, create_whisper_baseline, WhisperConfig
from .whisper_comparison import WhisperComparisonFramework, WhisperComparisonConfig, create_whisper_comparison_framework

# Core MoP components (architecture-agnostic)
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

# GPT-specific MoP components
from .gpt_mop import (
    ViewsLinear1D,
    Kernels1D, 
    FuseExcInh1D,
    MoPBlock
)

# Whisper-specific MoP components
from .whisper_mop import (
    ViewsConv2D,
    Kernels2D,
    FuseExcInh2D,
    MoP2D,
    EncoderBlock,
    DecoderBlock,
    WhisperMoP
)

__all__ = [
    # Main models (ViT application of MoP)
    "ViT_MoP", 
    "ViT_Baseline",
    
    # Main models (GPT application of MoP)
    "GPT_MoP",
    "create_gpt_mop",
    "create_gpt_baseline", 
    "create_gpt_quartet",
    
    # Main models (Whisper application of MoP)
    "WhisperMoP",
    "create_whisper_mop",
    "create_whisper_baseline",
    "WhisperConfig",
    
    # Comparison frameworks
    "GPTComparisonFramework",
    "ComparisonConfig",
    "create_comparison_framework",
    "WhisperComparisonFramework",
    "WhisperComparisonConfig",
    "create_whisper_comparison_framework",
    
    # Core MoP components (architecture-agnostic)
    "ViewsLinear", 
    "Kernels3", 
    "FuseExcInh",
    
    # GPT-specific MoP components
    "ViewsLinear1D",
    "Kernels1D",
    "FuseExcInh1D", 
    "MoPBlock",
    
    # Whisper-specific MoP components
    "ViewsConv2D",
    "Kernels2D",
    "FuseExcInh2D",
    "MoP2D",
    "EncoderBlock",
    "DecoderBlock",
    
    # Transformer components (reusable)
    "ViTEncoder",
    "PatchEmbed",
    "MSA", 
    "MLP",
    "Block",
    "DropPath",
]