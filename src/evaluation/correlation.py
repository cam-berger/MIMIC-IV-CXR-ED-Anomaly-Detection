"""
Correlation Analysis: Clinical Features ↔ CXR Abnormalities

Analyzes relationships between clinical measurements and abnormality predictions:
- Pearson correlation (linear relationships)
- Spearman correlation (monotonic relationships)
- Point-biserial correlation (binary features)
- Visualization with heatmaps and bar charts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr, pointbiserialr
import matplotlib.pyplot as plt
import seaborn as sns


class CorrelationAnalyzer:
    """
    Analyze correlations between clinical features and CXR abnormalities
    """

    def __init__(self, clinical_feature_names: List[str], abnormality_names: List[str]):
        """
        Args:
            clinical_feature_names: Names of clinical features
            abnormality_names: Names of abnormality classes
        """
        self.clinical_feature_names = clinical_feature_names
        self.abnormality_names = abnormality_names

    def compute_correlations(self,
                            clinical_features: np.ndarray,
                            abnormality_probs: np.ndarray,
                            method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlations between clinical features and abnormality predictions

        Args:
            clinical_features: [N, num_clinical_features] - Clinical feature values
            abnormality_probs: [N, num_abnormalities] - Predicted abnormality probabilities
            method: 'pearson', 'spearman', or 'auto'

        Returns:
            DataFrame with correlations [num_clinical_features x num_abnormalities]
        """
        n_features = clinical_features.shape[1]
        n_abnormalities = abnormality_probs.shape[1]

        # Initialize correlation and p-value matrices
        corr_matrix = np.zeros((n_features, n_abnormalities))
        pval_matrix = np.zeros((n_features, n_abnormalities))

        for i in range(n_features):
            for j in range(n_abnormalities):
                feature_vals = clinical_features[:, i]
                abnorm_vals = abnormality_probs[:, j]

                # Remove NaN values
                mask = ~(np.isnan(feature_vals) | np.isnan(abnorm_vals))
                feature_vals = feature_vals[mask]
                abnorm_vals = abnorm_vals[mask]

                if len(feature_vals) == 0:
                    continue

                # Choose correlation method
                if method == 'pearson' or (method == 'auto' and self._is_continuous(feature_vals)):
                    corr, pval = pearsonr(feature_vals, abnorm_vals)
                elif method == 'spearman':
                    corr, pval = spearmanr(feature_vals, abnorm_vals)
                elif method == 'auto' and self._is_binary(feature_vals):
                    corr, pval = pointbiserialr(feature_vals, abnorm_vals)
                else:
                    # Default to Spearman for robustness
                    corr, pval = spearmanr(feature_vals, abnorm_vals)

                corr_matrix[i, j] = corr
                pval_matrix[i, j] = pval

        # Create DataFrames
        corr_df = pd.DataFrame(
            corr_matrix,
            index=self.clinical_feature_names,
            columns=self.abnormality_names
        )

        pval_df = pd.DataFrame(
            pval_matrix,
            index=self.clinical_feature_names,
            columns=self.abnormality_names
        )

        return corr_df, pval_df

    def _is_continuous(self, values: np.ndarray) -> bool:
        """Check if values are continuous (many unique values)"""
        return len(np.unique(values)) > 10

    def _is_binary(self, values: np.ndarray) -> bool:
        """Check if values are binary"""
        unique_vals = np.unique(values)
        return len(unique_vals) == 2 and set(unique_vals).issubset({0, 1})

    def get_top_correlations(self,
                            corr_df: pd.DataFrame,
                            pval_df: pd.DataFrame,
                            top_k: int = 20,
                            alpha: float = 0.05) -> pd.DataFrame:
        """
        Get top correlations (by absolute value) that are statistically significant

        Args:
            corr_df: Correlation matrix DataFrame
            pval_df: P-value matrix DataFrame
            top_k: Number of top correlations to return
            alpha: Significance level

        Returns:
            DataFrame with top correlations
        """
        results = []

        for feature in corr_df.index:
            for abnormality in corr_df.columns:
                corr = corr_df.loc[feature, abnormality]
                pval = pval_df.loc[feature, abnormality]

                # Only include significant correlations
                if pval < alpha:
                    results.append({
                        'clinical_feature': feature,
                        'abnormality': abnormality,
                        'correlation': corr,
                        'abs_correlation': abs(corr),
                        'p_value': pval,
                        'direction': 'positive' if corr > 0 else 'negative'
                    })

        df = pd.DataFrame(results)

        if len(df) == 0:
            return df

        # Sort by absolute correlation
        df = df.sort_values('abs_correlation', ascending=False)

        return df.head(top_k)

    def plot_correlation_heatmap(self,
                                corr_df: pd.DataFrame,
                                pval_df: pd.DataFrame,
                                figsize: Tuple[int, int] = (16, 10),
                                alpha: float = 0.05,
                                save_path: Optional[str] = None):
        """
        Plot correlation heatmap with significance markers

        Args:
            corr_df: Correlation matrix DataFrame
            pval_df: P-value matrix DataFrame
            figsize: Figure size
            alpha: Significance level for marking
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create mask for non-significant correlations
        mask = pval_df >= alpha

        # Plot heatmap
        sns.heatmap(
            corr_df,
            mask=None,  # Show all values
            annot=False,  # Don't annotate all cells
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Correlation Coefficient'},
            ax=ax,
            linewidths=0.5
        )

        # Add asterisks for significant correlations
        for i in range(len(corr_df.index)):
            for j in range(len(corr_df.columns)):
                if pval_df.iloc[i, j] < 0.001:
                    sig_marker = '***'
                elif pval_df.iloc[i, j] < 0.01:
                    sig_marker = '**'
                elif pval_df.iloc[i, j] < 0.05:
                    sig_marker = '*'
                else:
                    continue

                ax.text(j + 0.5, i + 0.5, sig_marker,
                       ha='center', va='center',
                       color='black', fontsize=8, fontweight='bold')

        ax.set_title('Correlation: Clinical Features ↔ CXR Abnormalities\n'
                    '(* p<0.05, ** p<0.01, *** p<0.001)',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Abnormalities', fontsize=12)
        ax.set_ylabel('Clinical Features', fontsize=12)

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_top_correlations_bar(self,
                                  top_corr_df: pd.DataFrame,
                                  figsize: Tuple[int, int] = (12, 8),
                                  save_path: Optional[str] = None):
        """
        Plot bar chart of top correlations

        Args:
            top_corr_df: DataFrame from get_top_correlations()
            figsize: Figure size
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create labels
        labels = [
            f"{row['clinical_feature']} → {row['abnormality']}"
            for _, row in top_corr_df.iterrows()
        ]

        # Color by direction
        colors = [
            'firebrick' if row['direction'] == 'positive' else 'steelblue'
            for _, row in top_corr_df.iterrows()
        ]

        # Plot horizontal bars
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, top_corr_df['correlation'], color=colors, alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel('Correlation Coefficient', fontsize=12)
        ax.set_title('Top 20 Correlations: Clinical Features ↔ Abnormalities',
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='firebrick', alpha=0.7, label='Positive correlation'),
            Patch(facecolor='steelblue', alpha=0.7, label='Negative correlation')
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def plot_feature_importance_per_abnormality(self,
                                               corr_df: pd.DataFrame,
                                               abnormality: str,
                                               top_k: int = 10,
                                               figsize: Tuple[int, int] = (10, 6),
                                               save_path: Optional[str] = None):
        """
        Plot top clinical features correlated with a specific abnormality

        Args:
            corr_df: Correlation matrix DataFrame
            abnormality: Name of abnormality
            top_k: Number of top features to plot
            figsize: Figure size
            save_path: Path to save figure
        """
        if abnormality not in corr_df.columns:
            raise ValueError(f"Abnormality '{abnormality}' not found in correlation matrix")

        # Get correlations for this abnormality
        feature_corrs = corr_df[abnormality].sort_values(key=abs, ascending=False).head(top_k)

        fig, ax = plt.subplots(figsize=figsize)

        # Colors
        colors = ['firebrick' if val > 0 else 'steelblue' for val in feature_corrs.values]

        # Plot
        feature_corrs.plot(kind='barh', ax=ax, color=colors, alpha=0.7)

        ax.set_xlabel('Correlation Coefficient', fontsize=12)
        ax.set_ylabel('Clinical Features', fontsize=12)
        ax.set_title(f'Top {top_k} Clinical Features Correlated with {abnormality}',
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

    def generate_correlation_report(self,
                                   clinical_features: np.ndarray,
                                   abnormality_probs: np.ndarray,
                                   output_dir: str):
        """
        Generate complete correlation analysis report

        Args:
            clinical_features: [N, num_clinical_features]
            abnormality_probs: [N, num_abnormalities]
            output_dir: Directory to save outputs
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("Computing correlations...")

        # Compute correlations (both methods)
        corr_pearson, pval_pearson = self.compute_correlations(
            clinical_features, abnormality_probs, method='pearson'
        )
        corr_spearman, pval_spearman = self.compute_correlations(
            clinical_features, abnormality_probs, method='spearman'
        )

        # Save matrices as CSV
        corr_pearson.to_csv(os.path.join(output_dir, 'correlation_pearson.csv'))
        pval_pearson.to_csv(os.path.join(output_dir, 'pvalue_pearson.csv'))
        corr_spearman.to_csv(os.path.join(output_dir, 'correlation_spearman.csv'))
        pval_spearman.to_csv(os.path.join(output_dir, 'pvalue_spearman.csv'))

        print("Generating visualizations...")

        # Plot heatmaps
        self.plot_correlation_heatmap(
            corr_pearson, pval_pearson,
            save_path=os.path.join(output_dir, 'heatmap_pearson.png')
        )

        self.plot_correlation_heatmap(
            corr_spearman, pval_spearman,
            save_path=os.path.join(output_dir, 'heatmap_spearman.png')
        )

        # Get and plot top correlations
        top_corr = self.get_top_correlations(corr_pearson, pval_pearson, top_k=20)
        top_corr.to_csv(os.path.join(output_dir, 'top_correlations.csv'), index=False)

        self.plot_top_correlations_bar(
            top_corr,
            save_path=os.path.join(output_dir, 'top_correlations_bar.png')
        )

        # Per-abnormality feature importance
        abnorm_dir = os.path.join(output_dir, 'per_abnormality')
        os.makedirs(abnorm_dir, exist_ok=True)

        for abnormality in self.abnormality_names:
            self.plot_feature_importance_per_abnormality(
                corr_pearson,
                abnormality,
                save_path=os.path.join(abnorm_dir, f'{abnormality}_features.png')
            )

        print(f"Correlation analysis complete! Results saved to {output_dir}")

        # Print summary
        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"\nTop 10 Correlations (Pearson):")
        print(top_corr.head(10)[['clinical_feature', 'abnormality', 'correlation', 'p_value']].to_string(index=False))


if __name__ == '__main__':
    # Test correlation analysis
    np.random.seed(42)

    # Simulate data
    n_samples = 1000
    n_clinical_features = 10
    n_abnormalities = 14

    clinical_feature_names = [
        'age', 'gender', 'heart_rate', 'respiratory_rate', 'temperature',
        'o2_saturation', 'systolic_bp', 'diastolic_bp', 'acuity', 'pain_score'
    ]

    abnormality_names = [
        'No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation',
        'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

    # Generate clinical features
    clinical_features = np.random.randn(n_samples, n_clinical_features)

    # Generate abnormality probabilities with some correlation to clinical features
    abnormality_probs = np.random.rand(n_samples, n_abnormalities)

    # Add some correlations
    # Temperature correlates with Pneumonia
    abnormality_probs[:, 11] += 0.3 * clinical_features[:, 4]  # Pneumonia ~ Temperature
    # Age correlates with Cardiomegaly
    abnormality_probs[:, 2] += 0.2 * clinical_features[:, 0]   # Cardiomegaly ~ Age
    # Clip to [0, 1]
    abnormality_probs = np.clip(abnormality_probs, 0, 1)

    # Create analyzer
    analyzer = CorrelationAnalyzer(clinical_feature_names, abnormality_names)

    # Compute correlations
    corr_df, pval_df = analyzer.compute_correlations(clinical_features, abnormality_probs)

    print("Correlation Matrix (Pearson):")
    print(corr_df.round(3))

    # Get top correlations
    top_corr = analyzer.get_top_correlations(corr_df, pval_df, top_k=10)
    print("\nTop 10 Correlations:")
    print(top_corr[['clinical_feature', 'abnormality', 'correlation', 'p_value']])

    # Generate visualizations
    print("\nGenerating test visualizations...")
    analyzer.plot_correlation_heatmap(corr_df, pval_df)
    analyzer.plot_top_correlations_bar(top_corr)
    analyzer.plot_feature_importance_per_abnormality(corr_df, 'Pneumonia')
