#!/usr/bin/env python3
"""
Local testing script for Phase 1 preprocessing with minimal resource requirements
Designed for laptop testing before full AWS deployment
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from phase1_preprocess import (
    DataConfig,
    MIMICDataJoiner,
    ImagePreprocessor,
    TextPreprocessor,
    S3Helper
)
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_minimal_test_config(local_mimic_path: str, num_samples: int = 10) -> DataConfig:
    """
    Create a minimal configuration for local testing

    Args:
        local_mimic_path: Path to local MIMIC data
        num_samples: Number of samples to process (default: 10)
    """
    config = DataConfig()

    # Set local paths
    mimic_path = Path(local_mimic_path).expanduser()
    config.mimic_cxr_path = str(mimic_path / "mimic-cxr-jpg" / "2.1.0")
    config.mimic_iv_path = str(mimic_path / "mimiciv" / "3.1")
    config.mimic_ed_path = str(mimic_path / "mimic-iv-ed" / "2.2")  # Separate ED dataset
    config.reflacx_path = str(mimic_path / "reflacx")

    # Output to local directory
    config.output_path = str(Path("./test_output").absolute())

    # Disable S3
    config.use_s3 = False
    config.s3_bucket = None

    # Reduce resource requirements
    config.image_size = 224  # Smaller than default 518
    config.max_text_length = 512  # Smaller than default 8192
    config.top_k_retrieval = 2  # Fewer retrieval results

    # Store sample size for later use
    config.num_samples = num_samples

    logger.info(f"Created minimal test config with {num_samples} samples")
    logger.info(f"MIMIC-CXR path: {config.mimic_cxr_path}")
    logger.info(f"MIMIC-IV path: {config.mimic_iv_path}")
    logger.info(f"Output path: {config.output_path}")

    return config


def test_data_loading(config: DataConfig):
    """Test 1: Verify we can load metadata without images"""
    logger.info("="*70)
    logger.info("TEST 1: Loading Metadata (No Images)")
    logger.info("="*70)

    try:
        data_joiner = MIMICDataJoiner(config)

        # Load CXR metadata (just the CSV files)
        cxr_metadata = data_joiner.load_mimic_cxr_metadata()
        logger.info(f"✓ Loaded {len(cxr_metadata)} CXR metadata records")

        # Load ED data (just the CSV files)
        ed_data = data_joiner.load_mimic_iv_ed()
        logger.info(f"✓ Loaded {len(ed_data)} ED tables")
        for table_name, table_df in ed_data.items():
            logger.info(f"  - {table_name}: {len(table_df)} records")

        return True, cxr_metadata, ed_data

    except Exception as e:
        logger.error(f"✗ Failed to load metadata: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_pseudo_note_creation(config: DataConfig, ed_data: dict):
    """Test 2: Verify pseudo-note creation from clinical data"""
    logger.info("="*70)
    logger.info("TEST 2: Creating Pseudo-Notes")
    logger.info("="*70)

    try:
        data_joiner = MIMICDataJoiner(config)

        # Create a sample clinical data dict
        if 'edstays' in ed_data and not ed_data['edstays'].empty:
            sample_stay = ed_data['edstays'].iloc[0]

            clinical_data = {
                'age': 65,
                'gender': 'M',
                'chiefcomplaint': 'Chest pain',
                'temperature': 98.6,
                'heartrate': 85,
                'resprate': 16,
                'o2sat': 97,
                'sbp': 130,
                'dbp': 80
            }

            pseudo_note = data_joiner.create_pseudo_notes(clinical_data)
            logger.info(f"✓ Created pseudo-note:")
            logger.info(f"  {pseudo_note}")

            return True
        else:
            logger.warning("No ED stays available for testing")
            return False

    except Exception as e:
        logger.error(f"✗ Failed to create pseudo-note: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_text_preprocessing(config: DataConfig):
    """Test 3: Verify text preprocessing works"""
    logger.info("="*70)
    logger.info("TEST 3: Text Preprocessing")
    logger.info("="*70)

    try:
        text_preprocessor = TextPreprocessor(config)

        sample_text = "Patient is a 65 year old M. Presenting with: Chest pain. Vitals: T: 98.6°F, HR: 85bpm, RR: 16/min, O2: 97%, SBP: 130mmHg, DBP: 80mmHg."

        # Clean text
        cleaned = text_preprocessor.clean_clinical_text(sample_text)
        logger.info(f"✓ Cleaned text: {cleaned[:100]}...")

        # Tokenize (this will download ModernBERT tokenizer if needed)
        encoded = text_preprocessor.preprocess_text(sample_text)
        logger.info(f"✓ Tokenized text shape: {encoded['input_ids'].shape}")

        # Extract entities
        entities = text_preprocessor.extract_medical_entities(sample_text)
        logger.info(f"✓ Extracted entities: {entities}")

        return True

    except Exception as e:
        logger.error(f"✗ Failed text preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_image_preprocessing_without_images(config: DataConfig, cxr_metadata: pd.DataFrame):
    """Test 4: Verify image preprocessing setup (without actually loading images)"""
    logger.info("="*70)
    logger.info("TEST 4: Image Preprocessing Setup (No Actual Images)")
    logger.info("="*70)

    try:
        image_preprocessor = ImagePreprocessor(config)

        # Check transform pipeline
        logger.info(f"✓ Image transform pipeline created")
        logger.info(f"  - Target size: {config.image_size}x{config.image_size}")
        logger.info(f"  - Normalization: mean={config.normalize_mean[:2]}... std={config.normalize_std[:2]}...")

        # Check if we can find image paths (without loading them)
        if not cxr_metadata.empty:
            sample_row = cxr_metadata.iloc[0]
            data_joiner = MIMICDataJoiner(config)
            image_path = data_joiner._get_image_path(sample_row)
            logger.info(f"✓ Sample image path: {image_path}")

            # Check if file exists
            if Path(image_path).exists():
                logger.info(f"✓ Image file exists at: {image_path}")
            else:
                logger.warning(f"⚠ Image file not found (this is OK for testing without images)")

        return True

    except Exception as e:
        logger.error(f"✗ Failed image preprocessing setup: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_minimal_pipeline_with_sample_images(config: DataConfig, num_images: int = 3):
    """
    Test 5: Process a few images if available

    Args:
        config: DataConfig
        num_images: Number of images to test (default: 3)
    """
    logger.info("="*70)
    logger.info(f"TEST 5: Processing {num_images} Sample Images (if available)")
    logger.info("="*70)

    try:
        data_joiner = MIMICDataJoiner(config)
        image_preprocessor = ImagePreprocessor(config)

        # Load metadata
        cxr_metadata = data_joiner.load_mimic_cxr_metadata()

        # Try to find images that exist
        processed_count = 0
        for idx, row in cxr_metadata.head(20).iterrows():  # Check first 20
            image_path = data_joiner._get_image_path(row)

            if Path(image_path).exists():
                # Try to preprocess
                image_tensor = image_preprocessor.preprocess_image(image_path)

                if image_tensor is not None:
                    logger.info(f"✓ Processed image {processed_count + 1}: {Path(image_path).name}")
                    logger.info(f"  - Tensor shape: {image_tensor.shape}")
                    processed_count += 1

                    if processed_count >= num_images:
                        break

        if processed_count > 0:
            logger.info(f"✓ Successfully processed {processed_count} images")
            return True
        else:
            logger.warning("⚠ No images found. You may need to download some sample images.")
            logger.info("  To download images, you can use:")
            logger.info("  - Select a few DICOM IDs from the metadata")
            logger.info("  - Download corresponding .jpg files from PhysioNet")
            return False

    except Exception as e:
        logger.error(f"✗ Failed image processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Local Phase 1 Testing Script')
    parser.add_argument(
        '--mimic-path',
        type=str,
        default='~/Documents/Portfolio/MIMIC_Data/physionet.org/files',
        help='Path to local MIMIC datasets'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='Number of samples to process'
    )
    parser.add_argument(
        '--test-images',
        action='store_true',
        help='Test with actual images (if available)'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=3,
        help='Number of images to test (if --test-images is set)'
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*70)
    print("PHASE 1 LOCAL TESTING - Minimal Resource Mode")
    print("="*70)
    print(f"MIMIC Path: {args.mimic_path}")
    print(f"Sample Size: {args.num_samples}")
    print(f"Test Images: {args.test_images}")
    print("="*70 + "\n")

    # Create test configuration
    config = create_minimal_test_config(args.mimic_path, args.num_samples)

    # Create output directory
    Path(config.output_path).mkdir(parents=True, exist_ok=True)

    # Run tests
    results = {}

    # Test 1: Data loading
    success, cxr_metadata, ed_data = test_data_loading(config)
    results['data_loading'] = success

    if not success:
        logger.error("Cannot proceed without data loading. Please check paths.")
        return

    # Test 2: Pseudo-note creation
    results['pseudo_notes'] = test_pseudo_note_creation(config, ed_data)

    # Test 3: Text preprocessing
    results['text_preprocessing'] = test_text_preprocessing(config)

    # Test 4: Image preprocessing setup
    results['image_setup'] = test_image_preprocessing_without_images(config, cxr_metadata)

    # Test 5: Optional image processing
    if args.test_images:
        results['image_processing'] = test_minimal_pipeline_with_sample_images(config, args.num_images)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:.<40} {status}")
    print("="*70)

    passed_count = sum(results.values())
    total_count = len(results)
    print(f"\nPassed: {passed_count}/{total_count}")

    if passed_count == total_count:
        print("\n🎉 All tests passed! Your Phase 1 setup is working correctly.")
    elif passed_count > 0:
        print(f"\n⚠ Some tests passed ({passed_count}/{total_count}). Review failures above.")
    else:
        print("\n❌ All tests failed. Please check your data paths and installation.")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
