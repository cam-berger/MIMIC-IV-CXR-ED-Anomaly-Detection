#!/usr/bin/env python3
"""
Test script for Enhanced RAG Adapter

This script tests the adapter with actual preprocessed data to ensure:
1. Enhanced RAG format is correctly detected
2. Conversion to Standard format works
3. Data can be loaded through MIMICDataset
4. All fields have correct shapes and types
"""

import sys
import torch
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.enhanced_rag_adapter import EnhancedRAGAdapter
from src.training.dataloader import MIMICDataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_adapter_direct(chunk_path: str):
    """Test the adapter directly on a chunk file"""
    print("=" * 70)
    print("Test 1: Direct Adapter Testing")
    print("=" * 70)

    # Load chunk
    logger.info(f"Loading chunk: {chunk_path}")
    data = torch.load(chunk_path, map_location='cpu', weights_only=False)
    logger.info(f"Loaded {len(data)} samples")

    if len(data) == 0:
        logger.error("Chunk is empty!")
        return False

    # Initialize adapter
    adapter = EnhancedRAGAdapter()

    # Test conversion on first sample
    logger.info("\nTesting conversion on first sample...")
    raw_sample = data[0]

    # Show raw format
    logger.info(f"\nRaw sample keys: {list(raw_sample.keys())}")

    # Convert
    try:
        converted = adapter.convert_sample(raw_sample)
        logger.info("✓ Conversion successful!")

        # Show converted format
        logger.info(f"\nConverted sample keys: {list(converted.keys())}")

        # Verify shapes
        print("\n" + "-" * 70)
        print("Converted Data Shapes:")
        print("-" * 70)

        assert 'image' in converted, "Missing 'image' key"
        assert converted['image'].shape == torch.Size([3, 518, 518]), f"Wrong image shape: {converted['image'].shape}"
        logger.info(f"✓ image: {converted['image'].shape}")

        assert 'text_input_ids' in converted, "Missing 'text_input_ids' key"
        logger.info(f"✓ text_input_ids: {converted['text_input_ids'].shape}")

        assert 'text_attention_mask' in converted, "Missing 'text_attention_mask' key"
        logger.info(f"✓ text_attention_mask: {converted['text_attention_mask'].shape}")

        assert 'clinical_features' in converted, "Missing 'clinical_features' key"
        assert converted['clinical_features'].shape == torch.Size([45]), f"Wrong clinical_features shape: {converted['clinical_features'].shape}"
        logger.info(f"✓ clinical_features: {converted['clinical_features'].shape}")

        assert 'labels' in converted, "Missing 'labels' key"
        assert isinstance(converted['labels'], dict), "Labels should be dict"
        assert len(converted['labels']) == 14, f"Wrong number of labels: {len(converted['labels'])}"
        logger.info(f"✓ labels: {len(converted['labels'])} classes")

        # Show label names
        logger.info(f"\nLabel names: {list(converted['labels'].keys())}")

        # Show enhanced features
        assert '_enhanced' in converted, "Missing '_enhanced' key"
        logger.info(f"\n✓ Enhanced features preserved:")
        logger.info(f"  - enhanced_note: {len(converted['_enhanced']['enhanced_note'])} chars")
        logger.info(f"  - attention_segments: {list(converted['_enhanced']['attention_segments'].keys())}")
        logger.info(f"  - bbox_coordinates: {len(converted['_enhanced']['bbox_coordinates'])} bboxes")
        logger.info(f"  - severity_scores: {len(converted['_enhanced']['severity_scores'])} scores")

        print("\n" + "=" * 70)
        print("✅ Direct adapter test PASSED")
        print("=" * 70)
        return True

    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_loading(chunk_path: str):
    """Test loading data through MIMICDataset"""
    print("\n" + "=" * 70)
    print("Test 2: MIMICDataset Loading")
    print("=" * 70)

    try:
        # Create dataset
        logger.info(f"Creating MIMICDataset from: {chunk_path}")
        dataset = MIMICDataset(
            data_path=chunk_path,
            max_samples=5  # Just test first 5 samples
        )

        logger.info(f"✓ Dataset created with {len(dataset)} samples")
        logger.info(f"✓ Data format detected: {dataset.data_format}")
        logger.info(f"✓ Adapter enabled: {dataset.adapter is not None}")
        logger.info(f"✓ Class names: {dataset.class_names}")

        # Test getting a sample
        logger.info("\nTesting __getitem__...")
        sample = dataset[0]

        # Verify sample structure
        assert 'image' in sample
        assert 'text_input_ids' in sample
        assert 'text_attention_mask' in sample
        assert 'clinical_features' in sample
        assert 'labels' in sample

        logger.info(f"✓ Sample loaded successfully")
        logger.info(f"  - image: {sample['image'].shape}")
        logger.info(f"  - text_input_ids: {sample['text_input_ids'].shape}")
        logger.info(f"  - clinical_features: {sample['clinical_features'].shape}")
        logger.info(f"  - labels: {len(sample['labels'])} classes")

        # Test class weights computation
        logger.info("\nTesting class weights computation...")
        class_weights = dataset.compute_class_weights()
        logger.info(f"✓ Class weights computed: shape {class_weights.shape}")
        logger.info(f"  Sample weights: {class_weights[:5].tolist()}")

        # Print adapter statistics
        if dataset.adapter:
            logger.info("\nAdapter statistics:")
            dataset.adapter.print_stats()

        print("\n" + "=" * 70)
        print("✅ Dataset loading test PASSED")
        print("=" * 70)
        return True

    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    # Default to user's data location
    default_chunk = "/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/train_test_small_2/train_chunk_000003.pt"

    if len(sys.argv) > 1:
        chunk_path = sys.argv[1]
    else:
        chunk_path = default_chunk

    chunk_file = Path(chunk_path)
    if not chunk_file.exists():
        logger.error(f"Chunk file not found: {chunk_path}")
        logger.info(f"Usage: python {sys.argv[0]} <path_to_chunk.pt>")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("Enhanced RAG Adapter Test Suite")
    print("=" * 70)
    print(f"Testing with: {chunk_path}")
    print("=" * 70)

    # Run tests
    test1_passed = test_adapter_direct(str(chunk_file))
    test2_passed = test_dataset_loading(str(chunk_file))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Test 1 (Direct Adapter):   {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Test 2 (Dataset Loading):  {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("=" * 70)

    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED! The Enhanced RAG adapter is working correctly.")
        print("\nYou can now use this data for training with:")
        print("  python src/training/train.py --config configs/phase3_multimodal.yaml")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED. Please review the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
