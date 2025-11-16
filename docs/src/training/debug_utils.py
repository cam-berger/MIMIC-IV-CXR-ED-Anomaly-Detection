"""
Debugging Utilities for Training

Tools to diagnose and debug common training issues:
- Gradient flow analysis
- Activation statistics
- Learning rate finder
- Batch visualization
- Memory profiling
- Model summary
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GradientFlowAnalyzer:
    """
    Analyze gradient flow through the network

    Detects:
    - Vanishing gradients (grad norm < 1e-8)
    - Exploding gradients (grad norm > 100)
    - Dead layers (no gradients)
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: PyTorch model to analyze
        """
        self.model = model
        self.gradient_stats = []

    def hook_gradients(self):
        """Register hooks to track gradients"""
        self.hooks = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, name=name: self._save_gradient_stats(grad, name)
                )
                self.hooks.append(hook)

    def _save_gradient_stats(self, grad: torch.Tensor, name: str):
        """Save gradient statistics"""
        if grad is not None:
            grad_norm = grad.norm().item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()

            self.gradient_stats.append({
                'name': name,
                'norm': grad_norm,
                'mean': grad_mean,
                'std': grad_std
            })

    def analyze_batch(self, loss: torch.Tensor) -> Dict[str, any]:
        """
        Analyze gradients for a single batch

        Args:
            loss: Loss tensor to backpropagate

        Returns:
            Dictionary with gradient statistics
        """
        # Clear previous stats
        self.gradient_stats = []

        # Backward pass
        loss.backward()

        # Analyze gradients
        vanishing_layers = []
        exploding_layers = []
        dead_layers = []

        for stat in self.gradient_stats:
            if stat['norm'] < 1e-8:
                vanishing_layers.append(stat['name'])
            elif stat['norm'] > 100:
                exploding_layers.append(stat['name'])

        # Find layers with no gradients
        params_with_grads = {stat['name'] for stat in self.gradient_stats}
        all_params = {name for name, param in self.model.named_parameters() if param.requires_grad}
        dead_layers = list(all_params - params_with_grads)

        return {
            'total_layers': len(all_params),
            'vanishing_layers': vanishing_layers,
            'exploding_layers': exploding_layers,
            'dead_layers': dead_layers,
            'gradient_stats': self.gradient_stats
        }

    def plot_gradient_flow(self, save_path: Optional[str] = None):
        """
        Plot gradient norms for each layer

        Args:
            save_path: Path to save figure (optional)
        """
        if len(self.gradient_stats) == 0:
            logger.warning("No gradient stats to plot. Run analyze_batch() first.")
            return

        # Extract data
        layer_names = [stat['name'] for stat in self.gradient_stats]
        grad_norms = [stat['norm'] for stat in self.gradient_stats]

        # Create figure
        fig, ax = plt.subplots(figsize=(15, 6))

        # Plot gradient norms
        x = np.arange(len(layer_names))
        ax.bar(x, grad_norms)

        # Add threshold lines
        ax.axhline(y=1e-8, color='r', linestyle='--', label='Vanishing threshold (1e-8)')
        ax.axhline(y=100, color='r', linestyle='--', label='Exploding threshold (100)')

        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, rotation=90, ha='right')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Gradient Flow Through Network')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved gradient flow plot to: {save_path}")

        return fig

    def remove_hooks(self):
        """Remove gradient hooks"""
        for hook in self.hooks:
            hook.remove()


class ActivationStatsTracker:
    """
    Track activation statistics to detect dead neurons

    Monitors mean and std of activations for each layer.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: PyTorch model to analyze
        """
        self.model = model
        self.activation_stats = {}
        self.hooks = []

    def register_hooks(self):
        """Register forward hooks to track activations"""
        def hook_fn(module, input, output, name):
            if isinstance(output, torch.Tensor):
                self.activation_stats[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'shape': tuple(output.shape)
                }

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(
                    lambda m, i, o, name=name: hook_fn(m, i, o, name)
                )
                self.hooks.append(hook)

    def analyze_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """
        Analyze activations for a single batch

        Args:
            batch: Input batch

        Returns:
            Dictionary with activation statistics
        """
        # Clear previous stats
        self.activation_stats = {}

        # Forward pass
        with torch.no_grad():
            _ = self.model(batch)

        # Find dead neurons (near-zero activations)
        dead_layers = []
        for name, stats in self.activation_stats.items():
            if abs(stats['mean']) < 1e-6 and stats['std'] < 1e-6:
                dead_layers.append(name)

        return {
            'total_layers': len(self.activation_stats),
            'dead_layers': dead_layers,
            'activation_stats': self.activation_stats
        }

    def remove_hooks(self):
        """Remove activation hooks"""
        for hook in self.hooks:
            hook.remove()


class LearningRateFinder:
    """
    Find optimal learning rate using Leslie Smith's LR range test

    Trains for one epoch with exponentially increasing LR and plots loss vs LR.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 criterion: nn.Module):
        """
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.lrs = []
        self.losses = []

    def find(self,
            train_loader: torch.utils.data.DataLoader,
            start_lr: float = 1e-7,
            end_lr: float = 10,
            num_iter: int = 100,
            smooth_f: float = 0.05) -> Tuple[float, plt.Figure]:
        """
        Run LR range test

        Args:
            train_loader: Training data loader
            start_lr: Starting learning rate
            end_lr: Ending learning rate
            num_iter: Number of iterations
            smooth_f: Smoothing factor for loss (exponential moving average)

        Returns:
            Tuple of (suggested_lr, figure)
        """
        # Save initial state
        model_state = self.model.state_dict()
        optimizer_state = self.optimizer.state_dict()

        # Initialize
        lr_mult = (end_lr / start_lr) ** (1 / num_iter)
        lr = start_lr
        self.optimizer.param_groups[0]['lr'] = lr

        avg_loss = 0.0
        best_loss = float('inf')
        batch_num = 0

        # Training loop
        self.model.train()
        iterator = iter(train_loader)

        for i in range(num_iter):
            try:
                batch = next(iterator)
            except StopIteration:
                # Restart iterator if needed
                iterator = iter(train_loader)
                batch = next(iterator)

            batch_num += 1

            # Forward pass
            outputs = self.model(batch)
            labels = self._extract_labels(batch['labels'])
            loss = self.criterion(outputs, labels)

            # Compute smoothed loss
            if i == 0:
                avg_loss = loss.item()
            else:
                avg_loss = smooth_f * loss.item() + (1 - smooth_f) * avg_loss

            # Record
            self.lrs.append(lr)
            self.losses.append(avg_loss)

            # Stop if loss is exploding
            if avg_loss > 4 * best_loss or torch.isnan(loss):
                break

            if avg_loss < best_loss:
                best_loss = avg_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update learning rate
            lr *= lr_mult
            self.optimizer.param_groups[0]['lr'] = lr

        # Restore initial state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)

        # Find suggested LR (steepest gradient)
        suggested_lr = self._find_steepest_gradient()

        # Plot
        fig = self.plot(suggested_lr=suggested_lr)

        return suggested_lr, fig

    def _find_steepest_gradient(self) -> float:
        """Find LR with steepest loss gradient"""
        # Compute gradient
        gradients = np.gradient(np.array(self.losses))

        # Find steepest (most negative) gradient
        min_grad_idx = np.argmin(gradients)

        return self.lrs[min_grad_idx]

    def plot(self, suggested_lr: Optional[float] = None, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot loss vs learning rate

        Args:
            suggested_lr: Suggested learning rate to highlight
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.lrs, self.losses, linewidth=2)
        ax.set_xscale('log')
        ax.set_xlabel('Learning Rate', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Learning Rate Finder', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Mark suggested LR
        if suggested_lr:
            ax.axvline(x=suggested_lr, color='r', linestyle='--', linewidth=2,
                      label=f'Suggested LR: {suggested_lr:.2e}')
            ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved LR finder plot to: {save_path}")

        return fig

    def _extract_labels(self, labels_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract labels from dictionary"""
        # Get batch size and number of classes
        batch_size = len(next(iter(labels_dict.values())))
        num_classes = len(labels_dict)

        # Stack labels
        labels_list = []
        for class_name in sorted(labels_dict.keys()):
            labels_list.append(labels_dict[class_name])

        return torch.stack(labels_list, dim=1).float()


class BatchInspector:
    """
    Visualize batches of data to verify data loading

    Shows:
    - Images
    - Text snippets
    - Clinical features
    - Labels
    """

    def __init__(self, class_names: List[str]):
        """
        Args:
            class_names: List of abnormality class names
        """
        self.class_names = class_names

    def inspect_batch(self,
                     batch: Dict[str, torch.Tensor],
                     num_samples: int = 4,
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize a batch of data

        Args:
            batch: Batch dictionary from DataLoader
            num_samples: Number of samples to visualize
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure
        """
        num_samples = min(num_samples, batch['image'].shape[0])

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Image
            image = batch['image'][i].cpu().numpy().transpose(1, 2, 0)
            image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]

            axes[i, 0].imshow(image)
            axes[i, 0].set_title(f'Sample {i}: Image', fontweight='bold')
            axes[i, 0].axis('off')

            # Clinical features (as bar chart)
            clinical_features = batch['clinical_features'][i].cpu().numpy()

            axes[i, 1].barh(range(len(clinical_features)), clinical_features)
            axes[i, 1].set_xlabel('Value')
            axes[i, 1].set_ylabel('Feature Index')
            axes[i, 1].set_title(f'Sample {i}: Clinical Features', fontweight='bold')
            axes[i, 1].grid(axis='x', alpha=0.3)

            # Labels (as text)
            labels_dict = {k: v[i].item() for k, v in batch['labels'].items()}
            positive_classes = [k for k, v in labels_dict.items() if v == 1]

            labels_text = "Positive labels:\n" + "\n".join(f"• {c}" for c in positive_classes)
            if len(positive_classes) == 0:
                labels_text = "No positive labels\n(Normal)"

            axes[i, 2].text(0.1, 0.5, labels_text, fontsize=10,
                          verticalalignment='center')
            axes[i, 2].set_title(f'Sample {i}: Labels', fontweight='bold')
            axes[i, 2].axis('off')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved batch inspection to: {save_path}")

        return fig


class MemoryProfiler:
    """
    Profile GPU memory usage during training

    Tracks:
    - Peak memory allocated
    - Current memory allocated
    - Memory reserved
    """

    def __init__(self, device: torch.device):
        """
        Args:
            device: CUDA device to profile
        """
        self.device = device
        self.memory_snapshots = []

    def snapshot(self, label: str = ""):
        """
        Take a memory snapshot

        Args:
            label: Label for this snapshot
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Skipping memory profiling.")
            return

        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)  # GB
        peak = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)  # GB

        self.memory_snapshots.append({
            'label': label,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'peak_gb': peak
        })

        logger.info(f"Memory snapshot '{label}': "
                   f"Allocated={allocated:.2f}GB, "
                   f"Reserved={reserved:.2f}GB, "
                   f"Peak={peak:.2f}GB")

    def plot(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot memory usage over time

        Args:
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure
        """
        if len(self.memory_snapshots) == 0:
            logger.warning("No memory snapshots to plot")
            return None

        labels = [s['label'] for s in self.memory_snapshots]
        allocated = [s['allocated_gb'] for s in self.memory_snapshots]
        reserved = [s['reserved_gb'] for s in self.memory_snapshots]
        peak = [s['peak_gb'] for s in self.memory_snapshots]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(labels))
        width = 0.25

        ax.bar(x - width, allocated, width, label='Allocated', alpha=0.8)
        ax.bar(x, reserved, width, label='Reserved', alpha=0.8)
        ax.bar(x + width, peak, width, label='Peak', alpha=0.8)

        ax.set_xlabel('Snapshot')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('GPU Memory Usage', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved memory profile to: {save_path}")

        return fig


def print_model_summary(model: nn.Module, input_sample: Dict[str, torch.Tensor]):
    """
    Print detailed model summary

    Args:
        model: PyTorch model
        input_sample: Sample input for tracing
    """
    logger.info("\n" + "=" * 80)
    logger.info("MODEL SUMMARY")
    logger.info("=" * 80)

    # Count parameters
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params

    logger.info(f"\nTotal parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # Estimate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)

    logger.info(f"Model size: {size_mb:.2f} MB")

    # Per-module statistics
    logger.info("\nPer-module parameters:")
    logger.info("-" * 80)
    logger.info(f"{'Module Name':<50} {'Parameters':>15} {'Trainable':>10}")
    logger.info("-" * 80)

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

            if num_params > 0:
                logger.info(f"{name:<50} {num_params:>15,} {num_trainable:>10,}")

    logger.info("=" * 80 + "\n")
