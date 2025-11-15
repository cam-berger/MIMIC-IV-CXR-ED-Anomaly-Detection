"""
Quick Data Loading Verification

Tests that:
1. Data files exist and are readable
2. Data format is correct
3. All required fields are present
4. DataLoader works correctly
5. Basic sanity checks pass

Run this BEFORE creating debug dataset or starting training.
"""

import os
import sys
import torch
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.dataloader import MIMICDataset, collate_fn
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_file_exists(data_root: str, filename: str) -> bool:
    """Check if file exists"""
    path = Path(data_root) / filename
    if not path.exists():
        logger.error(f"❌ File not found: {path}")
        return False
    logger.info(f"✓ Found: {path}")
    return True


def verify_data_format(data_root: str, filename: str) -> bool:
    """Verify data format is correct"""
    logger.info(f"\nVerifying format of {filename}...")

    try:
        path = Path(data_root) / filename
        data = torch.load(path, map_location='cpu')

        if not isinstance(data, list):
            logger.error(f"❌ Data should be a list, got {type(data)}")
            return False

        if len(data) == 0:
            logger.error(f"❌ Data is empty!")
            return False

        logger.info(f"✓ Loaded {len(data)} samples")

        # Check first sample
        sample = data[0]
        required_keys = [
            'subject_id', 'study_id', 'dicom_id',
            'image', 'text_input_ids', 'text_attention_mask',
            'clinical_features', 'labels'
        ]

        for key in required_keys:
            if key not in sample:
                logger.error(f"❌ Missing required key: {key}")
                return False

        logger.info(f"✓ All required keys present")

        # Check data types and shapes
        if not isinstance(sample['image'], torch.Tensor):
            logger.error(f"❌ image should be torch.Tensor, got {type(sample['image'])}")
            return False

        if sample['image'].shape != (3, 518, 518):
            logger.error(f"❌ image shape should be (3, 518, 518), got {sample['image'].shape}")
            return False

        logger.info(f"✓ Image shape correct: {sample['image'].shape}")

        if not isinstance(sample['text_input_ids'], torch.Tensor):
            logger.error(f"❌ text_input_ids should be torch.Tensor")
            return False

        logger.info(f"✓ Text input_ids shape: {sample['text_input_ids'].shape}")

        if not isinstance(sample['clinical_features'], torch.Tensor):
            logger.error(f"❌ clinical_features should be torch.Tensor")
            return False

        if len(sample['clinical_features']) != 45:
            logger.error(f"❌ clinical_features should have 45 elements, got {len(sample['clinical_features'])}")
            return False

        logger.info(f"✓ Clinical features shape: {sample['clinical_features'].shape}")

        if not isinstance(sample['labels'], dict):
            logger.error(f"❌ labels should be dict, got {type(sample['labels'])}")
            return False

        if len(sample['labels']) != 14:
            logger.error(f"❌ Should have 14 labels, got {len(sample['labels'])}")
            return False

        logger.info(f"✓ Labels format correct ({len(sample['labels'])} classes)")

        # Check label values
        for class_name, label in sample['labels'].items():
            if label not in [0, 1]:
                logger.error(f"❌ Label for {class_name} should be 0 or 1, got {label}")
                return False

        logger.info(f"✓ All label values are binary (0 or 1)")

        return True

    except Exception as e:
        logger.error(f"❌ Error verifying data: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading(data_root: str) -> bool:
    """Test MIMICDataset class"""
    logger.info("\nTesting MIMICDataset class...")

    try:
        train_path = str(Path(data_root) / 'train_final.pt')
        dataset = MIMICDataset(train_path, augmentation=None, max_samples=10)

        logger.info(f"✓ Created dataset with {len(dataset)} samples")

        # Get a sample
        sample = dataset[0]

        logger.info(f"✓ Retrieved sample 0")
        logger.info(f"  - Image shape: {sample['image'].shape}")
        logger.info(f"  - Text input_ids shape: {sample['text_input_ids'].shape}")
        logger.info(f"  - Clinical features shape: {sample['clinical_features'].shape}")
        logger.info(f"  - Labels: {len(sample['labels'])} classes")

        return True

    except Exception as e:
        logger.error(f"❌ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader(data_root: str) -> bool:
    """Test DataLoader with collate_fn"""
    logger.info("\nTesting DataLoader...")

    try:
        train_path = str(Path(data_root) / 'train_final.pt')
        dataset = MIMICDataset(train_path, augmentation=None, max_samples=10)

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Use 0 workers for testing
            collate_fn=collate_fn
        )

        logger.info(f"✓ Created DataLoader")

        # Get a batch
        batch = next(iter(dataloader))

        logger.info(f"✓ Retrieved batch")
        logger.info(f"  - Batch size: {batch['image'].shape[0]}")
        logger.info(f"  - Image batch shape: {batch['image'].shape}")
        logger.info(f"  - Text input_ids batch shape: {batch['text_input_ids'].shape}")
        logger.info(f"  - Clinical features batch shape: {batch['clinical_features'].shape}")
        logger.info(f"  - Labels: {len(batch['labels'])} classes")

        # Verify batching worked correctly
        if batch['image'].shape[0] != 4:
            logger.error(f"❌ Batch size should be 4, got {batch['image'].shape[0]}")
            return False

        if batch['image'].shape != (4, 3, 518, 518):
            logger.error(f"❌ Image batch shape should be (4, 3, 518, 518), got {batch['image'].shape}")
            return False

        if batch['clinical_features'].shape != (4, 45):
            logger.error(f"❌ Clinical features batch shape should be (4, 45), got {batch['clinical_features'].shape}")
            return False

        logger.info(f"✓ Batch shapes are correct")

        return True

    except Exception as e:
        logger.error(f"❌ Error testing dataloader: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Verify data loading works correctly')
    parser.add_argument('--data-root', type=str,
                       default='/media/dev/MIMIC_DATA/phase1_with_path_fixes_raw',
                       help='Directory containing train/val/test_final.pt files')

    args = parser.parse_args()

    logger.info("\n" + "=" * 60)
    logger.info("Data Loading Verification")
    logger.info("=" * 60)
    logger.info(f"Data root: {args.data_root}\n")

    all_passed = True

    # Test 1: Check files exist
    logger.info("Test 1: Checking files exist...")
    for filename in ['train_final.pt', 'val_final.pt', 'test_final.pt']:
        if not verify_file_exists(args.data_root, filename):
            all_passed = False

    if not all_passed:
        logger.error("\n❌ Files not found. Please check data_root path.")
        return

    # Test 2: Verify data format
    logger.info("\n" + "-" * 60)
    for filename in ['train_final.pt', 'val_final.pt', 'test_final.pt']:
        if not verify_data_format(args.data_root, filename):
            all_passed = False

    if not all_passed:
        logger.error("\n❌ Data format verification failed.")
        return

    # Test 3: Test MIMICDataset
    logger.info("\n" + "-" * 60)
    if not test_dataset_loading(args.data_root):
        all_passed = False

    # Test 4: Test DataLoader
    logger.info("\n" + "-" * 60)
    if not test_dataloader(args.data_root):
        all_passed = False

    # Summary
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("=" * 60)
        logger.info("\nYour data is ready to use!")
        logger.info("\nNext steps:")
        logger.info("  1. Create debug dataset:")
        logger.info("     python scripts/create_debug_dataset.py")
        logger.info("  2. Analyze dataset:")
        logger.info("     python scripts/analyze_dataset.py")
        logger.info("  3. Test training pipeline:")
        logger.info("     python scripts/test_full_pipeline_debug.py")
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.error("=" * 60)
        logger.error("\nPlease fix the issues above before proceeding.")

    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
