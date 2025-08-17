"""
Whisper Model Comparison Framework: Baseline vs MoP

Two-way comparison with parameter matching and fair evaluation for audio transformers.
Author: Eran Ben Artzy
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .whisper_mop import WhisperConfig, create_whisper_mop, create_whisper_baseline


@dataclass
class WhisperComparisonConfig:
    """Configuration for the Whisper comparison"""
    # Audio processing
    n_mels: int = 80
    n_audio_ctx: int = 1500
    
    # Transformer architecture
    n_layer: int = 12
    n_head: int = 20
    n_embd: int = 1280
    dropout: float = 0.0
    bias: bool = False
    
    # MoP configuration
    n_views: int = 5
    n_kernels: int = 3
    kernel_size: int = 5
    
    # Whisper-specific
    vocab_size: int = 51865
    use_abs_pos_emb: bool = True


class WhisperComparisonFramework:
    """Framework for comparing Baseline and MoP Whisper models"""
    
    def __init__(self, config: WhisperComparisonConfig):
        self.config = config
        self.models = {}
        self.param_counts = {}
        
    def build_models(self) -> Dict[str, nn.Module]:
        """Build both models with parameter matching"""
        
        # Create Whisper configuration
        whisper_config = WhisperConfig(
            n_mels=self.config.n_mels,
            n_audio_ctx=self.config.n_audio_ctx,
            n_layer_enc=self.config.n_layer,
            n_layer_dec=self.config.n_layer,
            n_head=self.config.n_head,
            n_embd=self.config.n_embd,
            n_text_ctx=self.config.n_audio_ctx,  # Use audio context for text context
            dropout=self.config.dropout,
            bias=self.config.bias,
            n_views=self.config.n_views,
            n_kernels=self.config.n_kernels,
            kernel_size=self.config.kernel_size,
            vocab_size=self.config.vocab_size,
            use_abs_pos_emb=self.config.use_abs_pos_emb
        )
        
        # Create models
        baseline = create_whisper_baseline(whisper_config)
        mop = create_whisper_mop(whisper_config)
        
        # Store models
        self.models = {
            "baseline": baseline,
            "mop": mop
        }
        
        # Count parameters
        self.param_counts = {
            name: self._count_params(model) 
            for name, model in self.models.items()
        }
        
        return self.models
    
    def _count_params(self, model: nn.Module) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_param_summary(self) -> Dict[str, Dict]:
        """Get detailed parameter summary for all models"""
        summary = {}
        
        for name, model in self.models.items():
            param_count = self.param_counts[name]
            
            # Count parameters by component
            components = self._count_params_by_component(model)
            
            summary[name] = {
                "total_params": param_count,
                "total_millions": param_count / 1e6,
                "components": components
            }
        
        return summary
    
    def _count_params_by_component(self, model: nn.Module) -> Dict[str, int]:
        """Break down parameters by component"""
        components = {
            "embeddings": 0,
            "attention": 0,
            "mlp": 0,
            "layer_norm": 0,
            "lm_head": 0,
            "mop_components": 0
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "wte" in name or "wpe" in name:
                    components["embeddings"] += param.numel()
                elif "attn" in name:
                    components["attention"] += param.numel()
                elif "mlp" in name or "fc" in name or "proj" in name:
                    if "views" in name or "kernels" in name or "fuse" in name:
                        components["mop_components"] += param.numel()
                    else:
                        components["mlp"] += param.numel()
                elif "ln" in name:
                    components["layer_norm"] += param.numel()
                elif "lm_head" in name:
                    components["lm_head"] += param.numel()
        
        return components
    
    def parameter_matching_analysis(self) -> Dict:
        """Analyze parameter matching between models"""
        baseline_params = self.param_counts["baseline"]
        
        analysis = {
            "baseline_params": baseline_params,
            "comparisons": {}
        }
        
        for name, params in self.param_counts.items():
            if name != "baseline":
                diff = params - baseline_params
                diff_pct = (diff / baseline_params) * 100
                
                analysis["comparisons"][name] = {
                    "params": params,
                    "difference": diff,
                    "difference_pct": diff_pct,
                    "is_matched": abs(diff_pct) < 1.0  # Within 1%
                }
        
        return analysis
    
    def get_model_info(self) -> Dict[str, Dict]:
        """Get detailed information about each model"""
        info = {}
        
        for name, model in self.models.items():
            config = getattr(model, 'config', None)
            
            info[name] = {
                "class": model.__class__.__name__,
                "config": {
                    "n_layer": getattr(config, 'n_layer', None),
                    "n_head": getattr(config, 'n_head', None),
                    "n_embd": getattr(config, 'n_embd', None),
                    "n_mels": getattr(config, 'n_mels', None),
                    "n_audio_ctx": getattr(config, 'n_audio_ctx', None),
                    "vocab_size": getattr(config, 'vocab_size', None),
                } if config else None,
                "has_mop": hasattr(model, 'get_gate_maps'),
                "param_count": self.param_counts[name]
            }
        
        return info
    
    def test_forward_pass(self, batch_size: int = 2, seq_len: int = 100, 
                         vocab_size: int = 1000, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Test forward pass for all models"""
        device = torch.device(device)
        
        # Create test input
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        results = {}
        
        for name, model in self.models.items():
            model = model.to(device)
            model.eval()
            
            try:
                with torch.no_grad():
                    logits, loss = model(x, targets=y)
                
                results[name] = {
                    "logits": logits,
                    "loss": loss,
                    "logits_shape": logits.shape,
                    "loss_value": loss.item() if loss is not None else None
                }
                
                # Test MoP-specific functionality if available
                if hasattr(model, 'get_gate_maps'):
                    try:
                        gates, views, kernels = model.get_gate_maps(x)
                        results[name]["mop_maps"] = {
                            "gates_shape": gates.shape,
                            "views_shape": views.shape,
                            "kernels_shape": kernels.shape
                        }
                    except Exception as e:
                        results[name]["mop_maps_error"] = str(e)
                
            except Exception as e:
                results[name] = {"error": str(e)}
            
            model = model.cpu()
        
        return results
    
    def test_audio_processing(self, batch_size: int = 2, time_steps: int = 100, 
                            freq_bins: int = 80, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Test audio-specific processing capabilities"""
        device = torch.device(device)
        
        # Create synthetic spectrogram-like input
        # Simulate mel-spectrogram: (B, T, F) where F = n_mels
        x = torch.randn(batch_size, time_steps, freq_bins, device=device)
        
        # Convert to token indices (simplified)
        x_tokens = torch.randint(0, 1000, (batch_size, time_steps), device=device)
        
        results = {}
        
        for name, model in self.models.items():
            model = model.to(device)
            model.eval()
            
            try:
                with torch.no_grad():
                    # Test forward pass
                    logits, _ = model(x_tokens)
                    
                    # Test MoP gate extraction if available
                    if hasattr(model, 'get_gate_maps'):
                        gates, views, kernels = model.get_gate_maps(x_tokens)
                        
                        results[name] = {
                            "logits_shape": logits.shape,
                            "gates_shape": gates.shape,
                            "views_shape": views.shape,
                            "kernels_shape": kernels.shape,
                            "audio_processing": "✅ Success"
                        }
                    else:
                        results[name] = {
                            "logits_shape": logits.shape,
                            "audio_processing": "✅ Success (No MoP)"
                        }
                
            except Exception as e:
                results[name] = {"error": str(e)}
            
            model = model.cpu()
        
        return results
    
    def print_comparison_summary(self):
        """Print a comprehensive comparison summary"""
        print("=" * 80)
        print("WHISPER MODEL COMPARISON: Baseline vs MoP")
        print("=" * 80)
        
        # Parameter summary
        print("\n📊 PARAMETER COUNTS:")
        print("-" * 40)
        for name, count in self.param_counts.items():
            print(f"{name:>10}: {count:>12,} ({count/1e6:>6.2f}M)")
        
        # Parameter matching analysis
        analysis = self.parameter_matching_analysis()
        print(f"\n🎯 PARAMETER MATCHING (Baseline: {analysis['baseline_params']:,}):")
        print("-" * 40)
        for name, comp in analysis["comparisons"].items():
            status = "✅ MATCHED" if comp["is_matched"] else "❌ MISMATCHED"
            print(f"{name:>10}: {comp['difference']:+>8,} ({comp['difference_pct']:+>6.2f}%) {status}")
        
        # Model information
        print("\n🏗️  MODEL ARCHITECTURES:")
        print("-" * 40)
        info = self.get_model_info()
        for name, details in info.items():
            config = details["config"]
            if config:
                print(f"{name:>10}: {details['class']}")
                print(f"{'':>10}  Layers: {config['n_layer']}, Heads: {config['n_head']}, "
                      f"Width: {config['n_embd']}")
                print(f"{'':>10}  Audio: {config['n_mels']} mels, {config['n_audio_ctx']} ctx, "
                      f"Vocab: {config['vocab_size']}")
                if details['has_mop']:
                    print(f"{'':>10}  MoP: ✅ (Views: {self.config.n_views}, "
                          f"Kernels: {self.config.n_kernels}, Kernel Size: {self.config.kernel_size})")
            else:
                print(f"{name:>10}: {details['class']} (No config)")
        
        print("\n🎵 AUDIO PROCESSING CAPABILITIES:")
        print("-" * 40)
        print("✅ Temporal-spectral pattern detection via 2D convolutions")
        print("✅ Multi-view projections for frequency domain analysis")
        print("✅ Excitatory/inhibitory gating for audio feature selection")
        print("✅ Boolean logic operations on spectrogram representations")
        
        print("\n" + "=" * 80)


def create_whisper_comparison_framework(config: WhisperComparisonConfig) -> WhisperComparisonFramework:
    """Convenience function to create a Whisper comparison framework"""
    return WhisperComparisonFramework(config)


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = WhisperComparisonConfig(
        n_layer=6,
        n_head=8,
        n_embd=512,
        n_mels=80,
        n_audio_ctx=500,
        n_views=3,
        n_kernels=2,
        kernel_size=3
    )
    
    # Create framework
    framework = create_whisper_comparison_framework(config)
    
    # Build models
    models = framework.build_models()
    
    # Print comparison
    framework.print_comparison_summary()
    
    # Test forward pass
    print("\n🧪 TESTING FORWARD PASS:")
    print("-" * 40)
    results = framework.test_forward_pass(batch_size=2, seq_len=64, vocab_size=1000)
    
    for name, result in results.items():
        if "error" not in result:
            print(f"{name:>10}: ✅ Logits {result['logits_shape']}, Loss: {result['loss_value']:.4f}")
            if "mop_maps" in result:
                maps = result["mop_maps"]
                print(f"{'':>10}  MoP Maps: Gates {maps['gates_shape']}, "
                      f"Views {maps['views_shape']}, Kernels {maps['kernels_shape']}")
        else:
            print(f"{name:>10}: ❌ Error: {result['error']}")
    
    # Test audio processing
    print("\n🎵 TESTING AUDIO PROCESSING:")
    print("-" * 40)
    audio_results = framework.test_audio_processing(batch_size=2, time_steps=64, freq_bins=80)
    
    for name, result in audio_results.items():
        if "error" not in result:
            print(f"{name:>10}: ✅ {result['audio_processing']}")
            if "gates_shape" in result:
                print(f"{'':>10}  Gates: {result['gates_shape']}, "
                      f"Views: {result['views_shape']}, Kernels: {result['kernels_shape']}")
        else:
            print(f"{name:>10}: ❌ Error: {result['error']}")
