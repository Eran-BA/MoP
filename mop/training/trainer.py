"""
Training utilities for MoP models

Author: Eran Ben Artzy
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from .utils import cosine_lr


class Trainer:
    """
    Trainer class for MoP models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "auto",
        use_amp: bool = True,
        compile_model: bool = True
    ):
        self.model = model
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_amp = use_amp and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Compile model for speed (PyTorch 2.0+)
        if compile_model:
            try:
                self.model = torch.compile(self.model)
                print(f"✅ Model compiled successfully")
            except Exception as e:
                print(f"⚠️ Model compilation failed: {e}")
    
    def train_epoch(
        self,
        train_loader,
        optimizer,
        criterion,
        epoch: int,
        total_epochs: int,
        lr_scheduler=None,
        log_interval: int = 100
    ) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(data)
                loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch}/{total_epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                      f'Loss: {loss.item():.6f}\tLR: {current_lr:.6f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'lr': optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def evaluate(self, test_loader, criterion) -> Dict[str, float]:
        """Evaluate model on test set."""
        
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        for data, target in test_loader:
            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(data)
                test_loss += criterion(output, target).item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': test_loss,
            'accuracy': accuracy
        }


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 3e-3,
    weight_decay: float = 0.05,
    warmup_epochs: int = 5,
    device: str = "auto",
    label: str = "model",
    eval_every: int = 1
) -> tuple:
    """
    Complete training function for MoP models.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader  
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay
        warmup_epochs: Warmup epochs for lr scheduler
        device: Device to use
        label: Label for logging
        eval_every: Evaluate every N epochs
        
    Returns:
        (trained_model, history)
    """
    
    # Setup trainer
    trainer = Trainer(model, device=device)
    
    # Setup optimizer and criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Setup learning rate scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    def lr_lambda(step):
        return cosine_lr(step, total_steps, 1.0, warmup_steps)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    history = []
    best_acc = 0.0
    start_time = time.time()
    
    print(f"🚀 Training {label}")
    print(f"Device: {trainer.device}")
    print(f"Epochs: {epochs}, LR: {lr}, Weight Decay: {weight_decay}")
    print("=" * 60)
    
    for epoch in range(1, epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(
            train_loader, optimizer, criterion, epoch, epochs, scheduler
        )
        
        # Evaluate
        if epoch % eval_every == 0 or epoch == epochs:
            val_metrics = trainer.evaluate(val_loader, criterion)
            
            elapsed = (time.time() - start_time) / 60.0
            
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"Time: {elapsed:.1f}min, LR: {train_metrics['lr']:.6f}")
            print("-" * 60)
            
            # Save metrics
            history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'lr': train_metrics['lr'],
                'time_min': elapsed
            })
            
            # Track best model
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                print(f"✅ New best validation accuracy: {best_acc:.2f}%")
    
    total_time = (time.time() - start_time) / 60.0
    print(f"\n🎉 Training completed!")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    return model, history