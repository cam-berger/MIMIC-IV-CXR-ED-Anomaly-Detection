"""
Metrics Computation for Multi-Label Classification

Computes comprehensive metrics for Enhanced MDF-Net evaluation:
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- F1-Score, Precision, Recall
- Sensitivity, Specificity
- NPV (Negative Predictive Value)
- Calibration metrics (ECE, Brier Score)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from scipy import stats
import pandas as pd


class MetricsComputer:
    """
    Comprehensive metrics computation for multi-label abnormality detection
    """

    def __init__(self, class_names: List[str]):
        """
        Args:
            class_names: List of abnormality class names (14 for CheXpert)
        """
        self.class_names = class_names
        self.num_classes = len(class_names)

    def compute_all_metrics(self,
                           y_true: np.ndarray,
                           y_pred_probs: np.ndarray,
                           threshold: float = 0.5) -> pd.DataFrame:
        """
        Compute all metrics for each abnormality class

        Args:
            y_true: [N, num_classes] - Binary ground truth
            y_pred_probs: [N, num_classes] - Predicted probabilities
            threshold: Classification threshold (default: 0.5)

        Returns:
            DataFrame with metrics for each class
        """
        results = []

        for idx, class_name in enumerate(self.class_names):
            y_true_class = y_true[:, idx]
            y_pred_class = y_pred_probs[:, idx]

            # Skip if class has no positive or negative samples
            if len(np.unique(y_true_class)) < 2:
                print(f"Warning: {class_name} has only one class, skipping metrics")
                continue

            metrics = self._compute_class_metrics(
                y_true_class, y_pred_class, threshold, class_name
            )
            results.append(metrics)

        df = pd.DataFrame(results)
        df = df.set_index('class_name')

        # Add mean metrics across all classes
        mean_metrics = df.mean().to_dict()
        mean_metrics['class_name'] = 'MEAN'
        df = pd.concat([df, pd.DataFrame([mean_metrics]).set_index('class_name')])

        return df

    def _compute_class_metrics(self,
                               y_true: np.ndarray,
                               y_pred_probs: np.ndarray,
                               threshold: float,
                               class_name: str) -> Dict:
        """
        Compute metrics for a single class

        Args:
            y_true: [N] - Binary ground truth
            y_pred_probs: [N] - Predicted probabilities
            threshold: Classification threshold
            class_name: Name of abnormality class

        Returns:
            Dictionary of metrics
        """
        # Binarize predictions
        y_pred_binary = (y_pred_probs >= threshold).astype(int)

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

        # Core metrics
        auroc = roc_auc_score(y_true, y_pred_probs)
        auprc = average_precision_score(y_true, y_pred_probs)

        # Classification metrics at threshold
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)

        # Clinical metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        ppv = precision  # Same as precision

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Calibration metrics
        ece = self._compute_ece(y_true, y_pred_probs)
        brier = brier_score_loss(y_true, y_pred_probs)

        # Optimal threshold (maximizing F1)
        optimal_threshold, optimal_f1 = self._find_optimal_threshold(y_true, y_pred_probs)

        # Prevalence
        prevalence = y_true.sum() / len(y_true)

        return {
            'class_name': class_name,
            'auroc': auroc,
            'auprc': auprc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'npv': npv,
            'ppv': ppv,
            'accuracy': accuracy,
            'ece': ece,
            'brier_score': brier,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': optimal_f1,
            'prevalence': prevalence,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'n_positive': int(y_true.sum()),
            'n_negative': int((1 - y_true).sum())
        }

    def _compute_ece(self, y_true: np.ndarray, y_pred_probs: np.ndarray,
                    n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)

        ECE measures calibration: are predicted probabilities accurate?
        Lower is better (ECE < 0.05 = well-calibrated)

        Args:
            y_true: [N] - Binary ground truth
            y_pred_probs: [N] - Predicted probabilities
            n_bins: Number of bins for calibration

        Returns:
            ECE value
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_pred_probs > bin_lower) & (y_pred_probs <= bin_upper)

            if in_bin.sum() == 0:
                continue

            # Compute accuracy and confidence in this bin
            bin_accuracy = y_true[in_bin].mean()
            bin_confidence = y_pred_probs[in_bin].mean()
            bin_weight = in_bin.sum() / len(y_true)

            # ECE contribution from this bin
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return ece

    def _find_optimal_threshold(self, y_true: np.ndarray, y_pred_probs: np.ndarray) -> Tuple[float, float]:
        """
        Find threshold that maximizes F1-score

        Args:
            y_true: [N] - Binary ground truth
            y_pred_probs: [N] - Predicted probabilities

        Returns:
            (optimal_threshold, best_f1)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)

        # Compute F1 for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Find best F1
        best_idx = np.argmax(f1_scores)
        optimal_f1 = f1_scores[best_idx]

        # Get corresponding threshold
        if best_idx < len(thresholds):
            optimal_threshold = thresholds[best_idx]
        else:
            optimal_threshold = 1.0

        return optimal_threshold, optimal_f1

    def compute_bootstrap_ci(self,
                            y_true: np.ndarray,
                            y_pred_probs: np.ndarray,
                            metric: str = 'auroc',
                            n_bootstrap: int = 1000,
                            confidence_level: float = 0.95) -> pd.DataFrame:
        """
        Compute bootstrap confidence intervals for metrics

        Args:
            y_true: [N, num_classes] - Binary ground truth
            y_pred_probs: [N, num_classes] - Predicted probabilities
            metric: Which metric to compute CI for ('auroc', 'auprc', 'f1')
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (default: 0.95)

        Returns:
            DataFrame with CIs for each class
        """
        np.random.seed(42)
        n_samples = y_true.shape[0]
        alpha = 1 - confidence_level

        results = []

        for idx, class_name in enumerate(self.class_names):
            y_true_class = y_true[:, idx]
            y_pred_class = y_pred_probs[:, idx]

            # Skip if only one class
            if len(np.unique(y_true_class)) < 2:
                continue

            # Bootstrap
            metric_values = []
            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                y_true_boot = y_true_class[indices]
                y_pred_boot = y_pred_class[indices]

                # Skip if bootstrap sample has only one class
                if len(np.unique(y_true_boot)) < 2:
                    continue

                # Compute metric
                if metric == 'auroc':
                    value = roc_auc_score(y_true_boot, y_pred_boot)
                elif metric == 'auprc':
                    value = average_precision_score(y_true_boot, y_pred_boot)
                elif metric == 'f1':
                    y_pred_binary = (y_pred_boot >= 0.5).astype(int)
                    value = f1_score(y_true_boot, y_pred_binary, zero_division=0)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                metric_values.append(value)

            # Compute CI
            lower = np.percentile(metric_values, alpha / 2 * 100)
            upper = np.percentile(metric_values, (1 - alpha / 2) * 100)
            mean_val = np.mean(metric_values)

            results.append({
                'class_name': class_name,
                f'{metric}_mean': mean_val,
                f'{metric}_ci_lower': lower,
                f'{metric}_ci_upper': upper,
                f'{metric}_ci_width': upper - lower
            })

        df = pd.DataFrame(results)
        return df

    def delong_test(self,
                   y_true: np.ndarray,
                   y_pred_probs1: np.ndarray,
                   y_pred_probs2: np.ndarray,
                   model1_name: str = 'Model 1',
                   model2_name: str = 'Model 2') -> pd.DataFrame:
        """
        DeLong test for comparing two models' AUROC

        Tests if difference in AUROC is statistically significant

        Args:
            y_true: [N, num_classes] - Binary ground truth
            y_pred_probs1: [N, num_classes] - Model 1 predictions
            y_pred_probs2: [N, num_classes] - Model 2 predictions
            model1_name: Name of model 1
            model2_name: Name of model 2

        Returns:
            DataFrame with test results per class
        """
        from scipy.stats import norm

        results = []

        for idx, class_name in enumerate(self.class_names):
            y_true_class = y_true[:, idx]
            y_pred1_class = y_pred_probs1[:, idx]
            y_pred2_class = y_pred_probs2[:, idx]

            # Skip if only one class
            if len(np.unique(y_true_class)) < 2:
                continue

            # Compute AUROC for both models
            auroc1 = roc_auc_score(y_true_class, y_pred1_class)
            auroc2 = roc_auc_score(y_true_class, y_pred2_class)

            # DeLong test (simplified version using Mann-Whitney U statistic)
            # Full implementation requires covariance estimation
            # For now, use a simpler z-test approximation

            n_pos = y_true_class.sum()
            n_neg = (1 - y_true_class).sum()

            # Compute variance (Hanley-McNeil formula)
            q1_1 = auroc1 / (2 - auroc1)
            q2_1 = 2 * auroc1**2 / (1 + auroc1)
            se1 = np.sqrt(
                (auroc1 * (1 - auroc1) + (n_pos - 1) * (q1_1 - auroc1**2) +
                 (n_neg - 1) * (q2_1 - auroc1**2)) / (n_pos * n_neg)
            )

            q1_2 = auroc2 / (2 - auroc2)
            q2_2 = 2 * auroc2**2 / (1 + auroc2)
            se2 = np.sqrt(
                (auroc2 * (1 - auroc2) + (n_pos - 1) * (q1_2 - auroc2**2) +
                 (n_neg - 1) * (q2_2 - auroc2**2)) / (n_pos * n_neg)
            )

            # Z-statistic
            z = (auroc1 - auroc2) / np.sqrt(se1**2 + se2**2)
            p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed test

            # Determine significance
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = ''

            results.append({
                'class_name': class_name,
                f'auroc_{model1_name}': auroc1,
                f'auroc_{model2_name}': auroc2,
                'difference': auroc1 - auroc2,
                'z_statistic': z,
                'p_value': p_value,
                'significant': sig
            })

        df = pd.DataFrame(results)
        return df


if __name__ == '__main__':
    # Test metrics computation
    from sklearn.datasets import make_multilabel_classification

    # Generate dummy multi-label data
    X, y_true = make_multilabel_classification(
        n_samples=1000,
        n_features=50,
        n_classes=14,
        n_labels=3,
        random_state=42
    )

    # Simulate predictions (random probabilities)
    y_pred_probs = np.random.rand(1000, 14)

    # Class names
    class_names = [
        'No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation',
        'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    # Compute metrics
    computer = MetricsComputer(class_names)
    metrics_df = computer.compute_all_metrics(y_true, y_pred_probs)

    print("Metrics Summary:")
    print(metrics_df[['auroc', 'auprc', 'f1_score', 'sensitivity', 'specificity']].round(3))

    # Bootstrap CI
    print("\nBootstrap Confidence Intervals (AUROC):")
    ci_df = computer.compute_bootstrap_ci(y_true, y_pred_probs, metric='auroc', n_bootstrap=100)
    print(ci_df[['class_name', 'auroc_mean', 'auroc_ci_lower', 'auroc_ci_upper']].round(3))

    # DeLong test
    print("\nDeLong Test (comparing two models):")
    y_pred_probs2 = np.random.rand(1000, 14)
    delong_df = computer.delong_test(y_true, y_pred_probs, y_pred_probs2,
                                     'Model A', 'Model B')
    print(delong_df[['class_name', 'auroc_Model A', 'auroc_Model B', 'difference', 'p_value', 'significant']])
