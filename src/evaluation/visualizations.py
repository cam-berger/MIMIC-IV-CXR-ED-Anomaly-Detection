"""
Visualization tools for model evaluation

Classes for creating publication-quality plots for model analysis:
- Confusion matrices
- ROC curves
- Precision-Recall curves
- Calibration plots
- Attention visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
import torch


class ConfusionMatrixPlotter:
    """
    Generate confusion matrices for multi-label classification

    Supports both per-class and combined confusion matrices with
    publication-quality formatting.
    """

    def __init__(self, class_names: List[str], figsize: Tuple[int, int] = (10, 8)):
        """
        Args:
            class_names: List of abnormality class names
            figsize: Figure size for matplotlib (width, height)
        """
        self.class_names = class_names
        self.figsize = figsize

    def plot_single_class(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_idx: int,
                         threshold: float = 0.5,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix for a single class

        Args:
            y_true: True labels (binary)
            y_pred: Predicted probabilities
            class_idx: Index of class to plot
            threshold: Classification threshold
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        # Binarize predictions
        y_pred_binary = (y_pred >= threshold).astype(int)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 5))

        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=ax, cbar_kws={'label': 'Count'})

        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix: {self.class_names[class_idx]}',
                    fontsize=14, fontweight='bold')

        # Add metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics_text = f'Acc: {accuracy:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}'
        ax.text(0.5, -0.15, metrics_text, ha='center', transform=ax.transAxes, fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_all_classes(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        threshold: float = 0.5,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrices for all classes in a grid

        Args:
            y_true: True labels [n_samples, n_classes]
            y_pred: Predicted probabilities [n_samples, n_classes]
            threshold: Classification threshold
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        n_classes = y_true.shape[1]
        n_cols = 4
        n_rows = (n_classes + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten()

        for i in range(n_classes):
            ax = axes[i]

            # Binarize predictions
            y_pred_binary = (y_pred[:, i] >= threshold).astype(int)

            # Compute confusion matrix
            cm = confusion_matrix(y_true[:, i], y_pred_binary)

            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Neg', 'Pos'],
                       yticklabels=['Neg', 'Pos'],
                       ax=ax, cbar=False)

            ax.set_title(self.class_names[i], fontsize=10, fontweight='bold')

            if i % n_cols == 0:
                ax.set_ylabel('True', fontsize=9)
            if i >= n_classes - n_cols:
                ax.set_xlabel('Predicted', fontsize=9)

        # Hide unused subplots
        for i in range(n_classes, len(axes)):
            axes[i].axis('off')

        fig.suptitle('Confusion Matrices: All Classes', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class ROCCurvePlotter:
    """
    Generate ROC curves with confidence intervals

    Supports individual and combined ROC curves for all classes.
    """

    def __init__(self, class_names: List[str], figsize: Tuple[int, int] = (10, 8)):
        """
        Args:
            class_names: List of abnormality class names
            figsize: Figure size for matplotlib
        """
        self.class_names = class_names
        self.figsize = figsize

    def plot_single_class(self,
                         y_true: np.ndarray,
                         y_pred_probs: np.ndarray,
                         class_idx: int,
                         add_ci: bool = False,
                         n_bootstrap: int = 1000,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curve for a single class

        Args:
            y_true: True labels (binary)
            y_pred_probs: Predicted probabilities
            class_idx: Index of class to plot
            add_ci: Whether to add bootstrap confidence intervals
            n_bootstrap: Number of bootstrap iterations
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
        roc_auc = auc(fpr, tpr)

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')

        # Add bootstrap confidence intervals if requested
        if add_ci:
            tprs = []
            base_fpr = np.linspace(0, 1, 101)

            np.random.seed(42)
            for _ in range(n_bootstrap):
                # Bootstrap sample
                indices = np.random.choice(len(y_true), len(y_true), replace=True)

                if len(np.unique(y_true[indices])) < 2:
                    continue

                fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_pred_probs[indices])
                tpr_interp = np.interp(base_fpr, fpr_boot, tpr_boot)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)

            tprs = np.array(tprs)
            mean_tpr = tprs.mean(axis=0)
            std_tpr = tprs.std(axis=0)

            tpr_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
            tpr_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)

            ax.fill_between(base_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.2,
                           label='95% CI')

        # Plot diagonal (random baseline)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random (AUC = 0.500)')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve: {self.class_names[class_idx]}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_all_classes(self,
                        y_true: np.ndarray,
                        y_pred_probs: np.ndarray,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot ROC curves for all classes on the same plot

        Args:
            y_true: True labels [n_samples, n_classes]
            y_pred_probs: Predicted probabilities [n_samples, n_classes]
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Color palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.class_names)))

        aucs = []
        for i, (name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            ax.plot(fpr, tpr, color=color, lw=1.5, alpha=0.8,
                   label=f'{name} (AUC={roc_auc:.3f})')

        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC=0.500)')

        # Calculate mean AUC
        mean_auc = np.mean(aucs)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves: All Classes (Mean AUC={mean_auc:.3f})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class PRCurvePlotter:
    """
    Generate Precision-Recall curves

    Supports individual and combined PR curves for all classes.
    """

    def __init__(self, class_names: List[str], figsize: Tuple[int, int] = (10, 8)):
        """
        Args:
            class_names: List of abnormality class names
            figsize: Figure size for matplotlib
        """
        self.class_names = class_names
        self.figsize = figsize

    def plot_single_class(self,
                         y_true: np.ndarray,
                         y_pred_probs: np.ndarray,
                         class_idx: int,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curve for a single class

        Args:
            y_true: True labels (binary)
            y_pred_probs: Predicted probabilities
            class_idx: Index of class to plot
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        # Compute PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
        avg_precision = average_precision_score(y_true, y_pred_probs)

        # Baseline (prevalence)
        prevalence = y_true.mean()

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot PR curve
        ax.plot(recall, precision, color='darkorange', lw=2,
               label=f'PR curve (AP = {avg_precision:.3f})')

        # Plot baseline (prevalence)
        ax.axhline(y=prevalence, color='navy', linestyle='--', lw=2,
                  label=f'Baseline (prevalence = {prevalence:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curve: {self.class_names[class_idx]}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_all_classes(self,
                        y_true: np.ndarray,
                        y_pred_probs: np.ndarray,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot Precision-Recall curves for all classes on the same plot

        Args:
            y_true: True labels [n_samples, n_classes]
            y_pred_probs: Predicted probabilities [n_samples, n_classes]
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Color palette
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.class_names)))

        aps = []
        for i, (name, color) in enumerate(zip(self.class_names, colors)):
            precision, recall, _ = precision_recall_curve(y_true[:, i], y_pred_probs[:, i])
            avg_precision = average_precision_score(y_true[:, i], y_pred_probs[:, i])
            aps.append(avg_precision)

            ax.plot(recall, precision, color=color, lw=1.5, alpha=0.8,
                   label=f'{name} (AP={avg_precision:.3f})')

        # Calculate mean AP
        mean_ap = np.mean(aps)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Precision-Recall Curves: All Classes (Mean AP={mean_ap:.3f})',
                    fontsize=14, fontweight='bold')
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class CalibrationPlotter:
    """
    Generate calibration (reliability) diagrams

    Shows how well predicted probabilities match actual frequencies.
    """

    def __init__(self, class_names: List[str], n_bins: int = 10):
        """
        Args:
            class_names: List of abnormality class names
            n_bins: Number of bins for calibration plot
        """
        self.class_names = class_names
        self.n_bins = n_bins

    def plot_single_class(self,
                         y_true: np.ndarray,
                         y_pred_probs: np.ndarray,
                         class_idx: int,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot calibration diagram for a single class

        Args:
            y_true: True labels (binary)
            y_pred_probs: Predicted probabilities
            class_idx: Index of class to plot
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        # Bin predictions
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(y_pred_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        # Calculate mean predicted probability and actual frequency per bin
        bin_probs = []
        bin_freqs = []
        bin_counts = []

        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_probs.append(y_pred_probs[mask].mean())
                bin_freqs.append(y_true[mask].mean())
                bin_counts.append(mask.sum())
            else:
                bin_probs.append((bins[i] + bins[i+1]) / 2)
                bin_freqs.append(np.nan)
                bin_counts.append(0)

        bin_probs = np.array(bin_probs)
        bin_freqs = np.array(bin_freqs)
        bin_counts = np.array(bin_counts)

        # Calculate ECE (Expected Calibration Error)
        valid_mask = ~np.isnan(bin_freqs)
        ece = np.sum(np.abs(bin_freqs[valid_mask] - bin_probs[valid_mask]) *
                     bin_counts[valid_mask]) / bin_counts[valid_mask].sum()

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot calibration curve
        ax.plot(bin_probs[valid_mask], bin_freqs[valid_mask],
               marker='o', markersize=8, color='darkorange', lw=2,
               label=f'Calibration curve (ECE = {ece:.3f})')

        # Plot perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')

        # Add histogram of predictions
        ax2 = ax.twinx()
        ax2.hist(y_pred_probs, bins=bins, alpha=0.3, color='blue', label='Prediction distribution')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.legend(loc='upper left', fontsize=10)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Actual Frequency', fontsize=12)
        ax.set_title(f'Calibration Plot: {self.class_names[class_idx]}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class AttentionVisualizer:
    """
    Visualize cross-modal attention on chest X-rays

    Overlays attention heatmaps on X-ray images to show which regions
    the model focused on.
    """

    def __init__(self):
        """Initialize attention visualizer"""
        pass

    def visualize_attention(self,
                           image: np.ndarray,
                           attention_weights: np.ndarray,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize attention weights overlaid on image

        Args:
            image: Input image [H, W] or [H, W, 3]
            attention_weights: Attention weights [H_attn, W_attn]
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        from scipy.ndimage import zoom

        # Ensure image is 2D (grayscale)
        if len(image.shape) == 3:
            image = image.mean(axis=2)

        # Resize attention to match image size
        zoom_factor = (image.shape[0] / attention_weights.shape[0],
                      image.shape[1] / attention_weights.shape[1])
        attention_resized = zoom(attention_weights, zoom_factor, order=1)

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original X-ray', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        # Attention heatmap
        im = axes[1].imshow(attention_resized, cmap='jet', alpha=0.8)
        axes[1].set_title('Attention Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        # Overlay
        axes[2].imshow(image, cmap='gray')
        axes[2].imshow(attention_resized, cmap='jet', alpha=0.5)
        axes[2].set_title('Attention Overlay', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class FeatureImportancePlotter:
    """
    Visualize feature importance from SHAP or other methods

    Creates bar charts showing which clinical features are most important
    for each abnormality prediction.
    """

    def __init__(self, feature_names: List[str]):
        """
        Args:
            feature_names: List of clinical feature names
        """
        self.feature_names = feature_names

    def plot_importance(self,
                       importance_scores: np.ndarray,
                       class_name: str,
                       top_k: int = 15,
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance as horizontal bar chart

        Args:
            importance_scores: Importance scores for each feature
            class_name: Name of abnormality class
            top_k: Number of top features to show
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        # Get top k features
        top_indices = np.argsort(np.abs(importance_scores))[-top_k:][::-1]
        top_features = [self.feature_names[i] for i in top_indices]
        top_scores = importance_scores[top_indices]

        # Color based on positive/negative
        colors = ['green' if score > 0 else 'red' for score in top_scores]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_scores, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Feature Importance: {class_name} (Top {top_k})',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Positive impact'),
                          Patch(facecolor='red', alpha=0.7, label='Negative impact')]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


class TrainingCurvePlotter:
    """
    Plot training curves from TensorBoard logs or saved metrics

    Visualizes loss and metric curves over training epochs.
    """

    def __init__(self):
        """Initialize training curve plotter"""
        pass

    def plot_loss_curves(self,
                        train_losses: List[float],
                        val_losses: List[float],
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training and validation loss curves

        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        epochs = range(1, len(train_losses) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
        ax.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_metric_curves(self,
                          train_metrics: Dict[str, List[float]],
                          val_metrics: Dict[str, List[float]],
                          metric_name: str = 'AUROC',
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training and validation metric curves

        Args:
            train_metrics: Dict of training metrics per epoch
            val_metrics: Dict of validation metrics per epoch
            metric_name: Name of metric to plot
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure object
        """
        epochs = range(1, len(train_metrics[metric_name]) + 1)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(epochs, train_metrics[metric_name], 'b-', linewidth=2,
               label=f'Training {metric_name}')
        ax.plot(epochs, val_metrics[metric_name], 'r-', linewidth=2,
               label=f'Validation {metric_name}')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'Training and Validation {metric_name}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
