"""
GPT Model Comparison Framework: Baseline vs Quartet vs MoP

Three-way comparison with parameter matching and fair evaluation.
Author: Eran Ben Artzy
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .gpt_mop import create_gpt_baseline, create_gpt_mop, create_gpt_quartet
from .quartet_attn_patch import TransformerConfig


@dataclass
class ComparisonConfig:
    """Configuration for the three-way comparison"""

    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 640
    dropout: float = 0.1
    block_size: int = 256
    bias: bool = False
    n_views: int = 5
    n_kernels: int = 3
    quartet_gate_init: float = -5.0
    quartet_scale: float = 1.0


class GPTComparisonFramework:
    """Framework for comparing Baseline, Quartet, and MoP GPT models"""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.models = {}
        self.param_counts = {}

    def build_models(self, vocab_size: int) -> Dict[str, nn.Module]:
        """Build all three models with parameter matching"""

        # Base configuration
        base_config = TransformerConfig(
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_embd=self.config.n_embd,
            dropout=self.config.dropout,
            block_size=self.config.block_size,
            bias=self.config.bias,
            use_quartet=False,
        )

        quartet_config = TransformerConfig(
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_embd=self.config.n_embd,
            dropout=self.config.dropout,
            block_size=self.config.block_size,
            bias=self.config.bias,
            use_quartet=True,
            quartet_gate_init=self.config.quartet_gate_init,
            quartet_scale=self.config.quartet_scale,
        )

        # Create models
        baseline = create_gpt_baseline(vocab_size, base_config)
        quartet = create_gpt_quartet(vocab_size, quartet_config)
        mop = create_gpt_mop(
            vocab_size,
            base_config,
            n_views=self.config.n_views,
            n_kernels=self.config.n_kernels,
        )

        # Store models
        self.models = {"baseline": baseline, "quartet": quartet, "mop": mop}

        # Count parameters
        self.param_counts = {
            name: self._count_params(model) for name, model in self.models.items()
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
                "components": components,
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
            "mop_components": 0,
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

        analysis = {"baseline_params": baseline_params, "comparisons": {}}

        for name, params in self.param_counts.items():
            if name != "baseline":
                diff = params - baseline_params
                diff_pct = (diff / baseline_params) * 100

                analysis["comparisons"][name] = {
                    "params": params,
                    "difference": diff,
                    "difference_pct": diff_pct,
                    "is_matched": abs(diff_pct) < 1.0,  # Within 1%
                }

        return analysis

    def get_model_info(self) -> Dict[str, Dict]:
        """Get detailed information about each model"""
        info = {}

        for name, model in self.models.items():
            config = getattr(model, "config", None)

            info[name] = {
                "class": model.__class__.__name__,
                "config": (
                    {
                        "n_layer": getattr(config, "n_layer", None),
                        "n_head": getattr(config, "n_head", None),
                        "n_embd": getattr(config, "n_embd", None),
                        "block_size": getattr(config, "block_size", None),
                        "use_quartet": getattr(config, "use_quartet", None),
                    }
                    if config
                    else None
                ),
                "has_mop": hasattr(model, "get_gate_maps"),
                "param_count": self.param_counts[name],
            }

        return info

    def test_forward_pass(
        self,
        batch_size: int = 2,
        seq_len: int = 64,
        vocab_size: int = 1000,
        device: str = "cpu",
    ) -> Dict[str, torch.Tensor]:
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
                    "loss_value": loss.item() if loss is not None else None,
                }

                # Test MoP-specific functionality if available
                if hasattr(model, "get_gate_maps"):
                    try:
                        gates, views, kernels = model.get_gate_maps(x)
                        results[name]["mop_maps"] = {
                            "gates_shape": gates.shape,
                            "views_shape": views.shape,
                            "kernels_shape": kernels.shape,
                        }
                    except Exception as e:
                        results[name]["mop_maps_error"] = str(e)

            except Exception as e:
                results[name] = {"error": str(e)}

            model = model.cpu()

        return results

    def print_comparison_summary(self):
        """Print a comprehensive comparison summary"""
        print("=" * 80)
        print("GPT MODEL COMPARISON: Baseline vs Quartet vs MoP")
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
            print(
                f"{name:>10}: {comp['difference']:+>8,} ({comp['difference_pct']:+>6.2f}%) {status}"
            )

        # Model information
        print("\n🏗️  MODEL ARCHITECTURES:")
        print("-" * 40)
        info = self.get_model_info()
        for name, details in info.items():
            config = details["config"]
            if config:
                print(f"{name:>10}: {details['class']}")
                print(
                    f"{'':>10}  Layers: {config['n_layer']}, Heads: {config['n_head']}, "
                    f"Width: {config['n_embd']}, Block: {config['block_size']}"
                )
                if config["use_quartet"]:
                    print(f"{'':>10}  Quartet: ✅")
                if details["has_mop"]:
                    print(
                        f"{'':>10}  MoP: ✅ (Views: {self.config.n_views}, Kernels: {self.config.n_kernels})"
                    )
            else:
                print(f"{name:>10}: {details['class']} (No config)")

        print("\n" + "=" * 80)


def create_comparison_framework(config: ComparisonConfig) -> GPTComparisonFramework:
    """Convenience function to create a comparison framework"""
    return GPTComparisonFramework(config)


# Example usage and testing
if __name__ == "__main__":
    # Create configuration
    config = ComparisonConfig(
        n_layer=6, n_head=8, n_embd=512, block_size=256, n_views=5, n_kernels=3
    )

    # Create framework
    framework = create_comparison_framework(config)

    # Build models
    models = framework.build_models(vocab_size=1000)

    # Print comparison
    framework.print_comparison_summary()

    # Test forward pass
    print("\n🧪 TESTING FORWARD PASS:")
    print("-" * 40)
    results = framework.test_forward_pass(batch_size=2, seq_len=64, vocab_size=1000)

    for name, result in results.items():
        if "error" not in result:
            print(
                f"{name:>10}: ✅ Logits {result['logits_shape']}, Loss: {result['loss_value']:.4f}"
            )
            if "mop_maps" in result:
                maps = result["mop_maps"]
                print(
                    f"{'':>10}  MoP Maps: Gates {maps['gates_shape']}, "
                    f"Views {maps['views_shape']}, Kernels {maps['kernels_shape']}"
                )
        else:
            print(f"{name:>10}: ❌ Error: {result['error']}")
