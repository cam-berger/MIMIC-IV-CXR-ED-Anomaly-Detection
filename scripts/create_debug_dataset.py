"""
Create Debug Dataset for Fast Local Testing

Samples a small subset of data for rapid iteration and testing:
- 50 samples from train
- 25 samples from val
- 25 samples from test

Ensures diverse sampling across all abnormality classes.
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stratified_sample(data: List[Dict], n_samples: int, class_names: List[str]) -> List[Dict]:
    """
    Sample data while ensuring representation from all classes

    Args:
        data: List of data dictionaries
        n_samples: Number of samples to select
        class_names: List of class names to balance

    Returns:
        Sampled subset of data
    """
    # Group samples by their positive classes
    class_to_samples = defaultdict(list)

    for idx, sample in enumerate(data):
        for class_name in class_names:
            if sample['labels'].get(class_name, 0) == 1:
                class_to_samples[class_name].append(idx)

    # Ensure at least one sample from each class
    selected_indices = set()

    # First, sample at least one positive example per class
    for class_name in class_names:
        if len(class_to_samples[class_name]) > 0:
            idx = np.random.choice(class_to_samples[class_name])
            selected_indices.add(idx)

    # Fill remaining slots with random samples
    all_indices = set(range(len(data)))
    remaining_indices = list(all_indices - selected_indices)

    if len(selected_indices) < n_samples:
        n_additional = min(n_samples - len(selected_indices), len(remaining_indices))
        additional_indices = np.random.choice(remaining_indices, n_additional, replace=False)
        selected_indices.update(additional_indices)

    # Convert to sorted list
    selected_indices = sorted(list(selected_indices))[:n_samples]

    return [data[i] for i in selected_indices]


def create_debug_dataset(
    source_dir: str,
    output_dir: str,
    train_samples: int = 50,
    val_samples: int = 25,
    test_samples: int = 25
):
    """
    Create debug dataset from full dataset

    Args:
        source_dir: Directory containing train/val/test_final.pt files
        output_dir: Directory to save debug dataset
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load full datasets
    logger.info("Loading full datasets...")

    train_path = source_dir / "train_final.pt"
    val_path = source_dir / "val_final.pt"
    test_path = source_dir / "test_final.pt"

    # Check if files exist
    if not train_path.exists():
        logger.error(f"Train file not found: {train_path}")
        raise FileNotFoundError(f"Train file not found: {train_path}")

    if not val_path.exists():
        logger.error(f"Val file not found: {val_path}")
        raise FileNotFoundError(f"Val file not found: {val_path}")

    if not test_path.exists():
        logger.error(f"Test file not found: {test_path}")
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_data = torch.load(train_path, map_location='cpu')
    val_data = torch.load(val_path, map_location='cpu')
    test_data = torch.load(test_path, map_location='cpu')

    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")

    # Get class names from first sample
    if len(train_data) > 0:
        class_names = list(train_data[0]['labels'].keys())
        logger.info(f"Found {len(class_names)} classes: {class_names}")
    else:
        logger.error("Training data is empty!")
        raise ValueError("Training data is empty!")

    # Sample subsets
    logger.info(f"Sampling {train_samples} train samples...")
    debug_train = stratified_sample(train_data, train_samples, class_names)

    logger.info(f"Sampling {val_samples} val samples...")
    debug_val = stratified_sample(val_data, val_samples, class_names)

    logger.info(f"Sampling {test_samples} test samples...")
    debug_test = stratified_sample(test_data, test_samples, class_names)

    # Print statistics
    logger.info("\nDebug Dataset Statistics:")
    logger.info("=" * 60)

    for split_name, split_data in [("Train", debug_train), ("Val", debug_val), ("Test", debug_test)]:
        logger.info(f"\n{split_name} split: {len(split_data)} samples")

        # Count positive samples per class
        class_counts = {name: 0 for name in class_names}
        for sample in split_data:
            for class_name, label in sample['labels'].items():
                if label == 1:
                    class_counts[class_name] += 1

        logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            pct = 100 * count / len(split_data)
            logger.info(f"  {class_name:30s}: {count:3d} ({pct:5.1f}%)")

    # Save debug datasets
    logger.info(f"\nSaving debug datasets to: {output_dir}")

    train_output = output_dir / "train_final.pt"
    val_output = output_dir / "val_final.pt"
    test_output = output_dir / "test_final.pt"

    torch.save(debug_train, train_output)
    torch.save(debug_val, val_output)
    torch.save(debug_test, test_output)

    logger.info(f"Saved train debug dataset: {train_output}")
    logger.info(f"Saved val debug dataset: {val_output}")
    logger.info(f"Saved test debug dataset: {test_output}")

    # Calculate total size
    train_size_mb = train_output.stat().st_size / (1024 * 1024)
    val_size_mb = val_output.stat().st_size / (1024 * 1024)
    test_size_mb = test_output.stat().st_size / (1024 * 1024)
    total_size_mb = train_size_mb + val_size_mb + test_size_mb

    logger.info(f"\nTotal debug dataset size: {total_size_mb:.1f} MB")
    logger.info("=" * 60)
    logger.info("Debug dataset created successfully!")
    logger.info("\nYou can now test training with:")
    logger.info(f"  python src/training/train_lightning.py \\")
    logger.info(f"    --config configs/base.yaml \\")
    logger.info(f"    --data-root {output_dir} \\")
    logger.info(f"    --max-epochs 3")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Create debug dataset for fast local testing')

    parser.add_argument('--source-dir', type=str,
                       default='/media/dev/MIMIC_DATA/phase1_with_path_fixes_raw',
                       help='Directory containing full train/val/test_final.pt files')

    parser.add_argument('--output-dir', type=str,
                       default='data/debug',
                       help='Directory to save debug dataset')

    parser.add_argument('--train-samples', type=int, default=50,
                       help='Number of training samples')

    parser.add_argument('--val-samples', type=int, default=25,
                       help='Number of validation samples')

    parser.add_argument('--test-samples', type=int, default=25,
                       help='Number of test samples')

    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create debug dataset
    create_debug_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples
    )


if __name__ == '__main__':
    main()
