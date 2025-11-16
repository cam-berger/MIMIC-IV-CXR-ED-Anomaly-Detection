"""
Evaluation Package for Enhanced MDF-Net

Modules:
- metrics: Compute AUROC, AUPRC, F1, etc.
- visualizations: Generate plots (ROC, PR, confusion matrices)
- correlation: Analyze clinical features ↔ abnormalities
- attention: Visualize cross-modal attention
- shap_analysis: Feature importance with SHAP
- error_analysis: Identify failure modes
"""

from .metrics import MetricsComputer
from .visualizations import ConfusionMatrixPlotter, ROCCurvePlotter, PRCurvePlotter
from .correlation import CorrelationAnalyzer

__all__ = [
    'MetricsComputer',
    'ConfusionMatrixPlotter',
    'ROCCurvePlotter',
    'PRCurvePlotter',
    'CorrelationAnalyzer'
]
