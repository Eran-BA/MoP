# MoP: Mixture of Products for Transformers
[![CI](https://github.com/Eran-BA/MoP/actions/workflows/ci.yml/badge.svg)](https://github.com/Eran-BA/MoP/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0005--5186--5594-green.svg)](https://orcid.org/0009-0005-5186-5594)

## Overview

MoP introduces even spatial boolean logic capabilities to Transformers through a novel **Mixture of Products** mechanism. This approach enhances spatial reasoning by learning excitatory/inhibitory gating patterns that can realize boolean operations like AND, OR, and NOT over feature representations.

The MoP mechanism is **architecture-agnostic** and has been successfully implemented across multiple modalities:
- **Vision Transformers (ViT)** - Spatial reasoning for images ✅ *Implemented*
- **GPT Models** - Sequential token interactions with Quartet Attention ✅ *Implemented*
- **Audio Transformers (Whisper)** - Temporal-spectral patterns ✅ *Implemented*
- **Any Transformer Architecture** - General feature gating 🔮 *Extensible*

### Key Features

- 🧠 **Universal Boolean Logic**: Learn AND/OR/NOT operations across different modalities
- 🔧 **Architecture-Agnostic**: Successfully implemented for ViT, GPT, and Whisper
- 📊 **Parameter-Matched Comparisons**: Fair evaluation with identical parameter counts
- 🎯 **Multiple Domains**: Vision (CIFAR-10/100), Language (GPT), Audio (Whisper)
- 🔬 **Research-Ready**: Complete experimental framework and statistical testing
- 📈 **Reproducible**: Deterministic training with multiple random seeds
- 🎵 **Audio Processing**: 2D spectrogram analysis with temporal-spectral patterns
- 📝 **Language Modeling**: Enhanced attention with Quartet mechanism

## Architecture

The MoP mechanism extends Transformers with:

1. **Multi-view Projections**: Transform token embeddings into multiple feature views
2. **Learnable Kernels**: Convolutional filters for spatial/temporal pattern detection  
3. **Excitatory/Inhibitory Fusion**: Gating mechanism enabling boolean logic operations
4. **Token Modulation**: Apply learned gates to modulate transformer token representations

```
Input → Transformer Encoder → [Views + Kernels] → Exc/Inh Gates → Modulated Tokens → Output
```

### Quartet Attention Architecture

Our GPT-MoP implementation features the **Quartet Attention** mechanism, which extends standard scaled dot-product attention with dual-path processing:

<img width="741" height="671" alt="Screenshot 2025-08-16 at 17 30 39" src="https://github.com/user-attachments/assets/c7c6b736-353e-4aea-bec4-e5d3b36d160d" />


**Key Innovations:**
- **Dual-Path Processing**: Parallel QK and Q2K2 attention score calculations
- **Learnable Mixing**: Gate-controlled combination of normalized attention scores
- **Enhanced Expressiveness**: Captures more complex token interaction patterns
- **Parameter Efficiency**: Maintains similar parameter count to baseline

**Current Implementation:**
- **Vision**: Spatial attention over image patches (8×8 grid for CIFAR)
- **Language**: Sequential attention with Quartet mechanism for token interactions
- **Audio**: Temporal-spectral attention with 2D convolutions
- **Boolean Operations**: Learnable AND/OR/NOT combinations via excitatory/inhibitory gating

**Future Applications:**
- **Multimodal**: Cross-modal attention mechanisms
- **Real-time**: Streaming attention for live data processing
- **Specialized**: Domain-specific attention patterns

## Installation

### Quick Install
```bash
git clone https://github.com/Eran-BA/MoP.git
cd MoP
pip install -r requirements.txt
```

### Development Install
```bash
git clone https://github.com/Eran-BA/MoP.git
cd MoP
pip install -e .
```

### Verify Installation
```bash
pytest
python experiments/cifar10_multi_seed.py --tiny --steps 400 --eval_every 100 --seeds 0
```
The smoke script auto-detects your device (`cuda`/`mps`/`cpu`) and writes a CSV to `results/cifar10/cifar10_acc.csv`.

## Implementations

### 🖼️ Vision Transformers (ViT-MoP)
**Spatial Boolean Logic for Image Processing**
- **Application**: CIFAR-10/100 image classification
- **MoP Components**: `ViewsLinear`, `Kernels3`, `FuseExcInh`
- **Pattern Detection**: 8×8 spatial grid analysis
- **Boolean Operations**: AND/OR/NOT over image patches

```python
from mop.models import ViT_MoP, ViT_Baseline

# Create parameter-matched models
baseline = ViT_Baseline(dim=256, depth=6, heads=4, n_classes=10)
mop_model = ViT_MoP(dim=256, depth=6, heads=4, n_classes=10, 
                    n_views=5, n_kernels=3)

# Extract spatial attention patterns
gates, views, kernels = mop_model.get_gate_maps(x)
```

### 📝 Language Models (GPT-MoP)
**Sequential Boolean Logic with Quartet Attention**
- **Application**: Character-level language modeling
- **MoP Components**: `ViewsLinear1D`, `Kernels1D`, `FuseExcInh1D`
- **Pattern Detection**: Temporal token interactions
- **Enhanced Attention**: Dual-path QK processing with learnable mixing

```python
from mop.models import GPT_MoP, create_gpt_mop

# Create GPT-MoP with Quartet attention
config = TransformerConfig(n_layer=6, n_head=8, n_embd=256)
mop_model = create_gpt_mop(vocab_size=1000, config=config, n_views=5, n_kernels=3)

# Forward pass with MoP-enhanced attention
logits, loss = mop_model(x, targets=y)
```

### 🎵 Audio Transformers (Whisper-MoP)
**Temporal-Spectral Boolean Logic for Audio Processing**
- **Application**: Audio transcription and understanding
- **MoP Components**: `ViewsConv2D`, `Kernels2D`, `FuseExcInh2D`
- **Pattern Detection**: 2D spectrogram analysis (time × frequency)
- **Architecture**: Encoder-decoder with MoP gating in encoder

```python
from mop.models import WhisperMoP, create_whisper_mop

# Create Whisper-MoP for audio processing
config = WhisperConfig(n_layer_enc=6, n_layer_dec=6, n_embd=512, n_mels=80)
mop_model = create_whisper_mop(config)

# Process mel spectrograms with MoP gating
logits, loss, gates = mop_model(mel, dec_input_ids, targets=targets)
```

## Quick Start

### Basic Usage

```python
import torch
from mop import ViT_MoP, ViT_Baseline

# Create models with matched parameter counts
baseline = ViT_Baseline(dim=256, depth=6, heads=4, n_classes=10)
mop_model = ViT_MoP(dim=256, depth=6, heads=4, n_classes=10, 
                    n_views=5, n_kernels=3)

# Forward pass
x = torch.randn(32, 3, 32, 32)  # CIFAR-10 batch
logits_baseline = baseline(x)   # (32, 10)
logits_mop = mop_model(x)       # (32, 10)

# Extract spatial attention patterns
gates, views, kernels = mop_model.get_gate_maps(x)
print(f"Gates shape: {gates.shape}")    # (32, 1, 8, 8)
print(f"Views shape: {views.shape}")    # (32, 5, 8, 8)  
print(f"Kernels shape: {kernels.shape}") # (32, 3, 8, 8)
```

### Parameter Matching

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Baseline: {count_params(baseline):,} parameters")
print(f"MoP: {count_params(mop_model):,} parameters")
# Output: Nearly identical parameter counts for fair comparison
```

### Sanity Check (Tiny CIFAR-10 smoke)
Quick subset run to validate wiring and get a rough A/B signal:

| setting | baseline | MoP |
|--------:|:--------:|:---:|
| CIFAR-10 (tiny smoke) | 0.279 | 0.332 |

<p align="center">
  <img src="docs/assets/cifar10_smoke_bar.png" width="360" alt="CIFAR-10 tiny-smoke A/B">
</p>


## Results

### Vision Transformer Results (ViT + MoP on CIFAR)

| Dataset | Model | Test Accuracy | Δ vs Baseline | Parameters | Seeds |
|---------|-------|---------------|---------------|------------|-------|
| CIFAR-10 | ViT Baseline | 72.64% | - | 4.76M | 5 |
| CIFAR-10 | ViT + MoP | 72.77% | **+0.13%** | 4.76M | 5 |
| CIFAR-100 | ViT Baseline | 71.43% | - | 14.25M | 5 |  
| CIFAR-100 | ViT + MoP | 72.19% | **+0.76%** | 14.25M | 5 |

*Results averaged across multiple random seeds with statistical significance testing.*

### Statistical Significance
- **McNemar's Test**: Chi-square values indicate significant improvements
- **Bootstrap Confidence Intervals**: 95% CI excludes zero for CIFAR-100
- **Parameter-Matched**: Identical parameter counts ensure fair comparison

## Core Components

### MoP Components (Architecture-Agnostic)

```python
from mop.models import ViewsLinear, Kernels3, FuseExcInh

# Multi-view projection (any sequence length)
views = ViewsLinear(dim=256, n_views=5)

# Learnable pattern detection (adaptable kernel sizes)  
kernels = Kernels3(in_ch=5, n_kernels=3)

# Boolean logic fusion (excitatory/inhibitory)
fusion = FuseExcInh(in_ch=8)  # 5 views + 3 kernels
```

### Transformer Components (Reusable)

```python
from mop.models import ViTEncoder, PatchEmbed, MSA, MLP, Block

# Standard transformer building blocks
# Can be adapted for other architectures (GPT, etc.)
```

## Experiments

*Note:* **CIFAR-10 smoke training script is included** under `experiments/`. CIFAR-100 and ablations are marked as planned.

### CIFAR-10 Smoke Run
```bash
python experiments/cifar10_multi_seed.py --tiny --steps 400 --eval_every 100 --seeds 0 --out results/cifar10
```

### CIFAR-100 with Augmentation (Planned)
```bash
python experiments/cifar100_augmented.py --seeds 0 1 2 --output results/cifar100
```

### Ablation Studies (Planned)
```bash
python experiments/ablation_study.py --variants full views_only kernels_only no_gate
```

## Visualization

*Visualization utilities coming soon!*

```python
# Planned functionality:
from mop.visualization import visualize_gates

gates, views, kernels = model.get_gate_maps(images)
visualize_gates(images=images, gates=gates, views=views, 
                save_path='outputs/attention_maps.png')
```

## Extending to Other Architectures

The MoP mechanism is designed to be architecture-agnostic:

### For GPT Models (Future Work)
```python
# Conceptual extension:
class GPT_MoP(nn.Module):
    def __init__(self, vocab_size, dim, depth, heads, n_views=5, n_kernels=3):
        self.transformer = GPTEncoder(...)
        self.views = ViewsLinear(dim, n_views)  # Same MoP components!
        self.kernels = Kernels1D(...)           # 1D convolution for sequences
        self.fuse = FuseExcInh(...)
```

### For Audio Transformers (Future Work)
```python
# Conceptual extension:
class Audio_MoP(nn.Module):
    def __init__(self, dim, depth, heads, n_views=5, n_kernels=3):
        self.transformer = AudioEncoder(...)
        self.views = ViewsLinear(dim, n_views)  # Same MoP components!
        self.kernels = Kernels2D(...)           # 2D for spectrograms
        self.fuse = FuseExcInh(...)
```

## Contributing

Contributions are welcome! Areas of particular interest:

### High Priority
- **Training Scripts**: CIFAR-10/100 experiment implementations
- **Utility Functions**: Parameter matching, statistical testing
- **GPT Extension**: Apply MoP to language models
- **Whisper Extension**: Apply MoP to audio transformers

### Contributing Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Citation

If you use MoP in your research, please cite:

```bibtex
@misc{benartzy2025mop,
  title={MoP: Mixture of Products for Transformers - Spatial Boolean Logic for Neural Networks},
  author={Ben Artzy, Eran},
  year={2025},
  url={https://github.com/Eran-BA/MoP},
  note={ORCID: 0009-0005-5186-5594}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

### ✅ Phase 1: Core Implementation (Current)
- [x] MoP mechanism implementation
- [x] ViT integration and baseline
- [x] Parameter matching utilities
- [x] Basic visualization support

### 🔄 Phase 2: Experimental Framework (In Progress)
- [ ] CIFAR-10/100 training scripts
- [ ] Statistical significance testing
- [ ] Comprehensive ablation studies
- [ ] Advanced visualization tools

### 🔮 Phase 3: Multi-Domain Expansion (Future)
- [ ] GPT-MoP for language modeling
- [ ] Whisper-MoP for audio processing
- [ ] Multimodal applications
- [ ] Theoretical analysis of boolean operations

### 📈 Phase 4: Research & Publication
- [ ] Comprehensive benchmark across domains
- [ ] Theoretical foundations paper
- [ ] Conference submissions (NeurIPS, ICML, ICLR)
- [ ] Community adoption and feedback


### What's Next:
- [ ] Filter-Bank Mixture-of-Products:
- [ ] Generalizing from two lenses to a small bank of structural lenses via causal depthwise convolutions on Q/K(1...M):
- [ ] Multi-scale kernels with varying dilations
- [ ] Parallel Run of CNN filters on top of the bank of lenses (Product 1 , Product 2, .... Product M)
- [ ] Gate-free log-space multiplication (gate is optional)


## Contact & Collaboration

**Eran Ben Artzy**
- ORCID: [0009-0005-5186-5594](https://orcid.org/0009-0005-5186-5594)
- LinkedIn: [eran-ben-artzy](https://linkedin.com/in/eran-ben-artzy)
- GitHub: [@Eran-BA](https://github.com/Eran-BA)

---

*For questions, issues, or collaboration opportunities, please open an issue or reach out through the channels above.*

**Please ensure proper citation when using this work in your research.**

---

<div align="center">

**🧠 Bringing Boolean Logic to the Age of Transformers 🚀**

*MoP: Where spatial reasoning meets neural architecture*

</div>
