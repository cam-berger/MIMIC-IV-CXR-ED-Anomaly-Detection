"""
Comprehensive Dataset Analysis

Analyzes MIMIC-IV-CXR-ED Phase 3 output data to provide insights:
1. Class distribution and imbalance
2. Co-occurrence patterns
3. Clinical feature statistics
4. Text length distribution
5. Image statistics
6. Recommended class weights
7. Data quality assessment

Generates:
- Statistical report (Markdown)
- Visualization plots
- Recommended hyperparameters
"""

import os
import sys
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """Comprehensive dataset analyzer"""

    def __init__(self, data_root: str, class_names: List[str]):
        """
        Args:
            data_root: Directory containing train/val/test files (.pt or chunks)
            class_names: List of abnormality class names
        """
        self.data_root = Path(data_root)
        self.class_names = class_names

        # Load data (supports both combined and chunked formats)
        self.train_data = self._load_split_auto('train')
        self.val_data = self._load_split_auto('val')
        self.test_data = self._load_split_auto('test')

        logger.info(f"Loaded {len(self.train_data)} train, {len(self.val_data)} val, "
                   f"{len(self.test_data)} test samples")

    def _load_split_auto(self, split_name: str) -> List[Dict]:
        """
        Auto-detect and load data split (combined or chunked format)

        Args:
            split_name: 'train', 'val', or 'test'

        Returns:
            List of data samples
        """
        # Try combined format first
        combined_path = self.data_root / f'{split_name}_final.pt'
        if combined_path.exists():
            logger.info(f"Loading {split_name} (combined format)...")
            return torch.load(combined_path, map_location='cpu', weights_only=False)

        # Try chunked format
        chunk_pattern = f'{split_name}_chunk_*.pt'
        chunk_files = sorted(self.data_root.glob(chunk_pattern))

        if chunk_files:
            logger.info(f"Loading {split_name} from {len(chunk_files)} chunks...")
            all_data = []
            for chunk_file in chunk_files:
                chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
                all_data.extend(chunk_data)
            logger.info(f"  Loaded {len(all_data)} samples from chunks")
            return all_data

        # Not found
        logger.warning(f"No data found for {split_name} split")
        logger.warning(f"  Looked for: {split_name}_final.pt or {chunk_pattern}")
        return []

    def _load_split(self, filename: str) -> List[Dict]:
        """Load a data split (legacy method for backward compatibility)"""
        path = self.data_root / filename
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return []
        return torch.load(path, map_location='cpu', weights_only=False)

    def analyze_class_distribution(self) -> Dict:
        """Analyze class distribution across splits"""
        logger.info("Analyzing class distribution...")

        results = {}

        for split_name, split_data in [('train', self.train_data),
                                       ('val', self.val_data),
                                       ('test', self.test_data)]:
            if len(split_data) == 0:
                continue

            # Count positive samples per class
            class_counts = {name: 0 for name in self.class_names}

            for sample in split_data:
                for class_name, label in sample['labels'].items():
                    if label == 1:
                        class_counts[class_name] += 1

            # Calculate prevalence
            total_samples = len(split_data)
            class_prevalence = {name: count / total_samples
                              for name, count in class_counts.items()}

            # Calculate imbalance ratio (relative to most common class)
            max_count = max(class_counts.values())
            imbalance_ratio = {name: max_count / max(count, 1)
                             for name, count in class_counts.items()}

            results[split_name] = {
                'counts': class_counts,
                'prevalence': class_prevalence,
                'imbalance_ratio': imbalance_ratio,
                'total_samples': total_samples
            }

        return results

    def compute_class_weights(self, dist_results: Dict) -> np.ndarray:
        """
        Compute recommended positive class weights for loss function

        Args:
            dist_results: Output from analyze_class_distribution

        Returns:
            Array of recommended weights for each class
        """
        logger.info("Computing recommended class weights...")

        train_counts = dist_results['train']['counts']
        total_samples = dist_results['train']['total_samples']
        n_classes = len(self.class_names)

        # Compute weights: w = total / (n_classes * count)
        weights = []
        for class_name in self.class_names:
            count = train_counts[class_name]
            if count > 0:
                weight = total_samples / (n_classes * count)
            else:
                weight = 1.0
            weights.append(weight)

        return np.array(weights)

    def analyze_co_occurrence(self) -> pd.DataFrame:
        """Analyze co-occurrence of abnormalities"""
        logger.info("Analyzing class co-occurrence...")

        # Create co-occurrence matrix
        n_classes = len(self.class_names)
        co_occurrence = np.zeros((n_classes, n_classes))

        for sample in self.train_data:
            # Get positive classes
            positive_classes = [i for i, name in enumerate(self.class_names)
                              if sample['labels'][name] == 1]

            # Increment co-occurrence counts
            for i in positive_classes:
                for j in positive_classes:
                    co_occurrence[i, j] += 1

        # Convert to DataFrame
        co_occurrence_df = pd.DataFrame(
            co_occurrence,
            index=self.class_names,
            columns=self.class_names
        )

        return co_occurrence_df

    def analyze_clinical_features(self) -> pd.DataFrame:
        """Analyze clinical feature statistics"""
        logger.info("Analyzing clinical features...")

        # Collect all clinical features
        all_features = []
        for sample in self.train_data:
            all_features.append(sample['clinical_features'].numpy())

        all_features = np.array(all_features)  # [n_samples, 45]

        # Compute statistics
        stats = {
            'mean': all_features.mean(axis=0),
            'std': all_features.std(axis=0),
            'min': all_features.min(axis=0),
            'max': all_features.max(axis=0),
            'median': np.median(all_features, axis=0)
        }

        # Convert to DataFrame
        stats_df = pd.DataFrame(stats)
        stats_df.index = [f'feature_{i}' for i in range(len(stats_df))]

        return stats_df

    def analyze_text_length(self) -> Dict:
        """Analyze text sequence lengths"""
        logger.info("Analyzing text lengths...")

        lengths = []
        for sample in self.train_data:
            lengths.append(len(sample['text_input_ids']))

        return {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'median': np.median(lengths),
            'p95': np.percentile(lengths, 95),
            'p99': np.percentile(lengths, 99),
            'lengths': lengths
        }

    def analyze_image_statistics(self) -> Dict:
        """Analyze image statistics"""
        logger.info("Analyzing image statistics...")

        # Sample 1000 images for efficiency
        sample_size = min(1000, len(self.train_data))
        sample_indices = np.random.choice(len(self.train_data), sample_size, replace=False)

        brightness_vals = []
        contrast_vals = []

        for idx in sample_indices:
            image = self.train_data[idx]['image'].numpy()  # [3, 518, 518]

            # Compute brightness (mean pixel value)
            brightness = image.mean()
            brightness_vals.append(brightness)

            # Compute contrast (std of pixel values)
            contrast = image.std()
            contrast_vals.append(contrast)

        return {
            'brightness': {
                'mean': np.mean(brightness_vals),
                'std': np.std(brightness_vals),
                'min': np.min(brightness_vals),
                'max': np.max(brightness_vals)
            },
            'contrast': {
                'mean': np.mean(contrast_vals),
                'std': np.std(contrast_vals),
                'min': np.min(contrast_vals),
                'max': np.max(contrast_vals)
            }
        }

    def generate_report(self, output_dir: str):
        """Generate comprehensive analysis report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating report in: {output_dir}")

        # Run all analyses
        dist_results = self.analyze_class_distribution()
        class_weights = self.compute_class_weights(dist_results)
        co_occurrence = self.analyze_co_occurrence()
        clinical_stats = self.analyze_clinical_features()
        text_stats = self.analyze_text_length()
        image_stats = self.analyze_image_statistics()

        # Generate visualizations
        self._plot_class_distribution(dist_results, output_dir / 'class_distribution.png')
        self._plot_co_occurrence(co_occurrence, output_dir / 'co_occurrence.png')
        self._plot_text_length(text_stats, output_dir / 'text_length_distribution.png')
        self._plot_class_weights(class_weights, output_dir / 'class_weights.png')

        # Generate markdown report
        report_path = output_dir / 'dataset_analysis_report.md'
        with open(report_path, 'w') as f:
            f.write(self._generate_markdown_report(
                dist_results, class_weights, co_occurrence,
                clinical_stats, text_stats, image_stats
            ))

        logger.info(f"Report saved to: {report_path}")

        # Save statistics to CSV
        self._save_statistics_csv(dist_results, class_weights, output_dir)

        return report_path

    def _plot_class_distribution(self, dist_results: Dict, save_path: Path):
        """Plot class distribution"""
        train_counts = dist_results['train']['counts']

        fig, ax = plt.subplots(figsize=(12, 6))

        classes = list(train_counts.keys())
        counts = list(train_counts.values())

        # Sort by count
        sorted_idx = np.argsort(counts)[::-1]
        classes = [classes[i] for i in sorted_idx]
        counts = [counts[i] for i in sorted_idx]

        bars = ax.bar(range(len(classes)), counts, alpha=0.7)

        # Color bars by imbalance
        max_count = max(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ratio = count / max_count
            if ratio < 0.1:
                bar.set_color('red')  # Very rare
            elif ratio < 0.3:
                bar.set_color('orange')  # Rare
            else:
                bar.set_color('green')  # Common

        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_xlabel('Abnormality Class', fontsize=12)
        ax.set_ylabel('Number of Positive Samples', fontsize=12)
        ax.set_title('Class Distribution (Training Set)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Common (>30%)'),
            Patch(facecolor='orange', label='Rare (10-30%)'),
            Patch(facecolor='red', label='Very rare (<10%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_co_occurrence(self, co_occurrence: pd.DataFrame, save_path: Path):
        """Plot co-occurrence heatmap"""
        fig, ax = plt.subplots(figsize=(14, 12))

        # Normalize by diagonal (total count for each class)
        normalized = co_occurrence.div(np.diag(co_occurrence), axis=0)

        sns.heatmap(normalized, annot=False, cmap='YlOrRd', ax=ax,
                   cbar_kws={'label': 'Co-occurrence Probability'})

        ax.set_title('Abnormality Co-occurrence Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Abnormality', fontsize=12)
        ax.set_ylabel('Abnormality', fontsize=12)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_text_length(self, text_stats: Dict, save_path: Path):
        """Plot text length distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(text_stats['lengths'], bins=50, alpha=0.7, edgecolor='black')

        ax.axvline(text_stats['mean'], color='r', linestyle='--', linewidth=2,
                  label=f"Mean: {text_stats['mean']:.0f}")
        ax.axvline(text_stats['median'], color='g', linestyle='--', linewidth=2,
                  label=f"Median: {text_stats['median']:.0f}")
        ax.axvline(text_stats['p95'], color='orange', linestyle='--', linewidth=2,
                  label=f"95th %ile: {text_stats['p95']:.0f}")
        ax.axvline(8192, color='purple', linestyle='--', linewidth=2,
                  label="Max context (8192)")

        ax.set_xlabel('Text Length (tokens)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Text Sequence Length Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_class_weights(self, weights: np.ndarray, save_path: Path):
        """Plot recommended class weights"""
        fig, ax = plt.subplots(figsize=(12, 6))

        # Sort by weight
        sorted_idx = np.argsort(weights)[::-1]
        classes = [self.class_names[i] for i in sorted_idx]
        sorted_weights = weights[sorted_idx]

        bars = ax.bar(range(len(classes)), sorted_weights, alpha=0.7)

        # Color by weight magnitude
        for bar, weight in zip(bars, sorted_weights):
            if weight > 10:
                bar.set_color('red')
            elif weight > 3:
                bar.set_color('orange')
            else:
                bar.set_color('green')

        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.set_xlabel('Abnormality Class', fontsize=12)
        ax.set_ylabel('Recommended Weight', fontsize=12)
        ax.set_title('Recommended Class Weights for Loss Function', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_markdown_report(self, dist_results, class_weights,
                                  co_occurrence, clinical_stats, text_stats,
                                  image_stats) -> str:
        """Generate markdown report"""
        report = f"""# Dataset Analysis Report

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Summary Statistics

| Split | Samples |
|-------|---------|
| Train | {dist_results['train']['total_samples']:,} |
| Val   | {dist_results['val']['total_samples']:,} |
| Test  | {dist_results['test']['total_samples']:,} |
| **Total** | **{sum(r['total_samples'] for r in dist_results.values()):,}** |

---

## Class Distribution (Training Set)

| Class | Count | Prevalence | Imbalance Ratio | Recommended Weight |
|-------|-------|------------|-----------------|-------------------|
"""
        train_counts = dist_results['train']['counts']
        train_prevalence = dist_results['train']['prevalence']
        train_imbalance = dist_results['train']['imbalance_ratio']

        for i, class_name in enumerate(self.class_names):
            count = train_counts[class_name]
            prev = train_prevalence[class_name]
            imb = train_imbalance[class_name]
            weight = class_weights[i]

            report += f"| {class_name} | {count:,} | {prev:.3f} | {imb:.2f}x | {weight:.2f} |\n"

        report += f"""
**Key Observations:**
- Most common class: {max(train_counts, key=train_counts.get)} ({max(train_counts.values()):,} samples)
- Rarest class: {min(train_counts, key=train_counts.get)} ({min(train_counts.values()):,} samples)
- Maximum imbalance ratio: {max(train_imbalance.values()):.1f}x

---

## Top Co-occurring Abnormalities

"""
        # Find top co-occurring pairs
        co_occur_pairs = []
        for i, class1 in enumerate(self.class_names):
            for j, class2 in enumerate(self.class_names):
                if i < j:  # Upper triangle only
                    count = co_occurrence.iloc[i, j]
                    if count > 0:
                        co_occur_pairs.append((class1, class2, count))

        co_occur_pairs.sort(key=lambda x: x[2], reverse=True)

        report += "| Class 1 | Class 2 | Co-occurrences |\n"
        report += "|---------|---------|----------------|\n"

        for class1, class2, count in co_occur_pairs[:10]:
            report += f"| {class1} | {class2} | {int(count):,} |\n"

        report += f"""
---

## Text Statistics

| Metric | Value |
|--------|-------|
| Mean length | {text_stats['mean']:.0f} tokens |
| Median length | {text_stats['median']:.0f} tokens |
| Min length | {text_stats['min']:.0f} tokens |
| Max length | {text_stats['max']:.0f} tokens |
| 95th percentile | {text_stats['p95']:.0f} tokens |
| 99th percentile | {text_stats['p99']:.0f} tokens |

**Context window**: 8192 tokens (Clinical ModernBERT)
- **Samples exceeding limit**: {sum(1 for l in text_stats['lengths'] if l > 8192)} ({100 * sum(1 for l in text_stats['lengths'] if l > 8192) / len(text_stats['lengths']):.1f}%)

---

## Image Statistics (sampled 1000 images)

| Metric | Brightness | Contrast |
|--------|------------|----------|
| Mean | {image_stats['brightness']['mean']:.4f} | {image_stats['contrast']['mean']:.4f} |
| Std | {image_stats['brightness']['std']:.4f} | {image_stats['contrast']['std']:.4f} |
| Min | {image_stats['brightness']['min']:.4f} | {image_stats['contrast']['min']:.4f} |
| Max | {image_stats['brightness']['max']:.4f} | {image_stats['contrast']['max']:.4f} |

---

## Recommendations

### 1. Class Weights for Loss Function

Update your config with these recommended weights:

```python
RECOMMENDED_POS_WEIGHTS = torch.tensor([
"""
        for i, class_name in enumerate(self.class_names):
            report += f"    {class_weights[i]:.2f},  # {class_name}\n"

        report += """])
```

### 2. Data Augmentation

Based on image statistics:
- ✅ **Horizontal flip**: Recommended (anatomically valid for chest X-rays)
- ✅ **Rotation**: Recommended (±10 degrees)
- ✅ **Color jitter**: Recommended (brightness/contrast variations present)

### 3. Sampling Strategy

"""
        rare_classes = [name for name, imb in train_imbalance.items() if imb > 5]

        if len(rare_classes) > 0:
            report += f"""
**⚠️ Severe class imbalance detected!**

Rare classes ({len(rare_classes)}): {', '.join(rare_classes)}

Consider:
- Using **weighted sampling** during training
- Enabling `use_weighted_sampler: true` in config
- Using **focal loss** to focus on hard examples (already enabled)
"""
        else:
            report += "✅ Class distribution is reasonably balanced. Standard training should work well.\n"

        report += f"""
### 4. Text Processing

"""
        if text_stats['p99'] > 8192:
            report += f"""
**⚠️ Some texts exceed 8192 token limit!**

- {sum(1 for l in text_stats['lengths'] if l > 8192)} samples will be truncated
- Consider summarization or chunking for very long texts
"""
        else:
            report += "✅ All texts fit within 8192 token context window.\n"

        report += f"""
---

## Next Steps

1. **Review class weights**: Update `src/model/losses.py` with recommended weights
2. **Configure sampling**: Consider enabling weighted sampling for imbalanced classes
3. **Run baseline training**: Train single-modality baselines first
4. **Train full model**: Use Stage 1 → Stage 2 training strategy
5. **Evaluate results**: Compare against baselines and expected performance

---

## Files Generated

- `class_distribution.png` - Bar chart of class counts
- `co_occurrence.png` - Heatmap of class co-occurrences
- `text_length_distribution.png` - Histogram of text lengths
- `class_weights.png` - Bar chart of recommended weights
- `dataset_statistics.csv` - All statistics in CSV format

"""
        return report

    def _save_statistics_csv(self, dist_results, class_weights, output_dir):
        """Save statistics to CSV"""
        # Class distribution
        dist_df = pd.DataFrame({
            'class': self.class_names,
            'train_count': [dist_results['train']['counts'][c] for c in self.class_names],
            'train_prevalence': [dist_results['train']['prevalence'][c] for c in self.class_names],
            'train_imbalance': [dist_results['train']['imbalance_ratio'][c] for c in self.class_names],
            'recommended_weight': class_weights
        })

        dist_df.to_csv(output_dir / 'class_statistics.csv', index=False)
        logger.info(f"Saved class statistics to: {output_dir / 'class_statistics.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze MIMIC-IV-CXR-ED dataset')

    parser.add_argument('--data-root', type=str,
                       default='/media/dev/MIMIC_DATA/phase1_with_path_fixes_raw',
                       help='Directory containing train/val/test files (.pt or chunks)')

    parser.add_argument('--output-dir', type=str,
                       default='reports/dataset_analysis',
                       help='Output directory for analysis results')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Class names (from CheXpert)
    class_names = [
        "No Finding",
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Enlarged Cardiomediastinum",
        "Fracture",
        "Lung Lesion",
        "Lung Opacity",
        "Pleural Effusion",
        "Pleural Other",
        "Pneumonia",
        "Pneumothorax",
        "Support Devices"
    ]

    # Create analyzer
    logger.info("\n" + "=" * 60)
    logger.info("MIMIC-IV-CXR-ED Dataset Analysis")
    logger.info("=" * 60 + "\n")

    analyzer = DatasetAnalyzer(args.data_root, class_names)

    # Generate report
    report_path = analyzer.generate_report(args.output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Analysis Complete!")
    logger.info("=" * 60)
    logger.info(f"\n📊 Report: {report_path}")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Review the report and visualizations")
    logger.info("  2. Update class weights in src/model/losses.py if needed")
    logger.info("  3. Configure weighted sampling in configs/*.yaml if needed")
    logger.info("  4. Start training!")


if __name__ == '__main__':
    main()
