"""
Training utilities

Author: Eran Ben Artzy
"""

import math
import random
from typing import Union

import numpy as np
import torch


def cosine_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    """
    Cosine learning rate schedule with warmup.

    Args:
        step: Current step
        total_steps: Total training steps
        base_lr: Base learning rate (typically 1.0 for scheduler)
        warmup_steps: Number of warmup steps

    Returns:
        Learning rate multiplier
    """
    if step < warmup_steps:
        return base_lr * (step / max(1, warmup_steps))

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def set_seed(seed: int):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Additional CUDA settings for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_params(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print model information including parameter count.

    Args:
        model: PyTorch model
        model_name: Name for display
    """
    total_params = count_params(model)

    print(f"\n📊 {model_name} Information:")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Count parameters by module type
    param_breakdown = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_params = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
            if module_params > 0:
                module_type = type(module).__name__
                param_breakdown[module_type] = (
                    param_breakdown.get(module_type, 0) + module_params
                )

    print("  Parameter breakdown:")
    for module_type, params in sorted(
        param_breakdown.items(), key=lambda x: x[1], reverse=True
    ):
        percentage = 100 * params / total_params
        print(f"    {module_type}: {params:,} ({percentage:.1f}%)")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filepath)
    print(f"💾 Checkpoint saved: {filepath}")


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: Union[str, torch.device] = "cpu",
) -> dict:
    """
    Load model checkpoint.

    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint file
        device: Device to map tensors to

    Returns:
        Dictionary with epoch and loss info
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"📁 Checkpoint loaded: {filepath}")
    print(f"  Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")

    return {"epoch": checkpoint["epoch"], "loss": checkpoint["loss"]}


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.val:.4f} (avg: {self.avg:.4f})"


class EarlyStopping:
    """
    Early stopping utility.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation score (higher is better)
            model: Model to potentially save weights from

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"🔄 Restored best weights (score: {self.best_score:.4f})")
                return True
        else:
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()

        return False
