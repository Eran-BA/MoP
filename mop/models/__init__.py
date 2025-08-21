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

# Unified attention variants
from .attention_variants import (BaselineMSA, CrossViewMixerMSA, EdgewiseMSA,
                                 MultiHopMSA, UnifiedMSA)
# Core MoP components (architecture-agnostic)
from .components import (MLP, MSA, Block, DropPath, FuseExcInh, Kernels3,
                         PatchEmbed, ViewsLinear, ViTEncoder)
from .gpt_comparison import (ComparisonConfig, GPTComparisonFramework,
                             create_comparison_framework)
# GPT-specific MoP components
# GPT/Language Model implementations
from .gpt_mop import (FuseExcInh1D, GPT_MoP, Kernels1D, MoPBlock,
                      ViewsLinear1D, create_gpt_baseline, create_gpt_mop,
                      create_gpt_quartet)
from .vit_baseline import ViT_Baseline
# Vision Transformer implementations
from .vit_mop import ViT_MoP
from .whisper_comparison import (WhisperComparisonConfig,
                                 WhisperComparisonFramework,
                                 create_whisper_comparison_framework)
# Whisper-specific MoP components
# Whisper/Audio Transformer implementations
from .whisper_mop import (DecoderBlock, EncoderBlock, FuseExcInh2D, Kernels2D,
                          MoP2D, ViewsConv2D, WhisperConfig, WhisperMoP,
                          create_whisper_baseline, create_whisper_mop)

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
    # Unified attention variants
    "BaselineMSA",
    "CrossViewMixerMSA",
    "MultiHopMSA",
    "EdgewiseMSA",
    "UnifiedMSA",
]
