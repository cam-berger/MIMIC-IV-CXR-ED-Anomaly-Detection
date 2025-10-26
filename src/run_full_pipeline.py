#!/usr/bin/env python3
"""
Full preprocessing pipeline: Phase 1 preprocessing + Leakage filtering

This script runs both preprocessing stages in sequence:
1. Phase 1: Data preprocessing (join multimodal data, create pseudo-notes, etc.)
2. Phase 1b: Leakage filtering (remove diagnosis information)

Usage:
    python src/run_full_pipeline.py \
        --gcs-bucket bergermimiciv \
        --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
        --gcs-project-id YOUR_PROJECT_ID \
        --mimic-iv-path physionet.org/files/mimiciv/3.1 \
        --mimic-ed-path physionet.org/files/mimic-iv-ed/2.2 \
        --output-path processed/phase1_final \
        --aggressive-filtering
"""

import sys
import argparse
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_phase1_preprocessing(args):
    """Run Phase 1 preprocessing pipeline"""
    logger.info("=" * 80)
    logger.info("PHASE 1: DATA PREPROCESSING")
    logger.info("=" * 80)

    # Build command
    cmd = [
        "python", "src/phase1_preprocess.py",
        "--gcs-bucket", args.gcs_bucket,
        "--gcs-cxr-bucket", args.gcs_cxr_bucket,
        "--gcs-project-id", args.gcs_project_id,
        "--mimic-iv-path", args.mimic_iv_path,
        "--mimic-ed-path", args.mimic_ed_path,
        "--output-path", f"{args.output_path}_raw",  # Temporary output
        "--image-size", str(args.image_size),
        "--max-text-length", str(args.max_text_length),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
    ]

    if args.reflacx_path:
        cmd.extend(["--reflacx-path", args.reflacx_path])

    logger.info(f"Running command: {' '.join(cmd)}")

    # Run preprocessing
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error("Phase 1 preprocessing failed!")
        sys.exit(1)

    logger.info("Phase 1 preprocessing completed successfully!")
    return f"{args.output_path}_raw"


def run_leakage_filtering(args, input_path):
    """Run leakage filtering on preprocessed data"""
    logger.info("=" * 80)
    logger.info("PHASE 1b: DIAGNOSIS LEAKAGE FILTERING")
    logger.info("=" * 80)

    # Build command
    cmd = [
        "python", "src/apply_leakage_filter.py",
        "--gcs-bucket", args.gcs_bucket,
        "--gcs-cxr-bucket", args.gcs_cxr_bucket,
        "--gcs-project-id", args.gcs_project_id,
        "--input-path", input_path,
        "--output-path", args.output_path,
    ]

    if args.aggressive_filtering:
        cmd.append("--aggressive")

    if args.use_nlp_model:
        cmd.append("--use-nlp-model")

    logger.info(f"Running command: {' '.join(cmd)}")

    # Run filtering
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        logger.error("Leakage filtering failed!")
        sys.exit(1)

    logger.info("Leakage filtering completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Full preprocessing pipeline: Phase 1 + Leakage filtering',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # GCS settings
    parser.add_argument('--gcs-bucket', type=str, required=True,
                       help='Main GCS bucket (e.g., bergermimiciv)')
    parser.add_argument('--gcs-cxr-bucket', type=str, required=True,
                       help='MIMIC-CXR bucket (e.g., mimic-cxr-jpg-2.1.0.physionet.org)')
    parser.add_argument('--gcs-project-id', type=str, required=True,
                       help='GCP project ID for requester pays')

    # Input paths
    parser.add_argument('--mimic-iv-path', type=str,
                       default='physionet.org/files/mimiciv/3.1',
                       help='Path to MIMIC-IV in GCS bucket')
    parser.add_argument('--mimic-ed-path', type=str,
                       default='physionet.org/files/mimic-iv-ed/2.2',
                       help='Path to MIMIC-IV-ED in GCS bucket')
    parser.add_argument('--reflacx-path', type=str,
                       default=None,
                       help='Path to REFLACX in GCS bucket (optional)')

    # Output
    parser.add_argument('--output-path', type=str,
                       default='processed/phase1_final',
                       help='Final output path in GCS bucket')

    # Processing settings
    parser.add_argument('--image-size', type=int, default=518,
                       help='Image size for preprocessing')
    parser.add_argument('--max-text-length', type=int, default=8192,
                       help='Max text length for ModernBERT')

    # Performance optimization
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Number of records to process in each batch for parallel downloading (default: 100)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel workers for image downloading (default: 4)')

    # Filtering options
    parser.add_argument('--aggressive-filtering', action='store_true',
                       help='Use aggressive leakage filtering')
    parser.add_argument('--use-nlp-model', action='store_true',
                       help='Use BioBERT for semantic filtering (slower)')

    # Pipeline control
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip Phase 1 preprocessing (run filtering only)')
    parser.add_argument('--skip-filtering', action='store_true',
                       help='Skip leakage filtering (run preprocessing only)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MIMIC-IV-CXR-ED FULL PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"GCS Bucket: {args.gcs_bucket}")
    logger.info(f"CXR Bucket: {args.gcs_cxr_bucket}")
    logger.info(f"Project ID: {args.gcs_project_id}")
    logger.info(f"Output Path: gs://{args.gcs_bucket}/{args.output_path}")
    logger.info(f"Image Size: {args.image_size}")
    logger.info(f"Max Text Length: {args.max_text_length}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Num Workers: {args.num_workers}")
    logger.info(f"Aggressive Filtering: {args.aggressive_filtering}")
    logger.info("=" * 80)

    # Run pipeline stages
    preprocessing_output = None

    if not args.skip_preprocessing:
        preprocessing_output = run_phase1_preprocessing(args)
    else:
        preprocessing_output = f"{args.output_path}_raw"
        logger.info(f"Skipping preprocessing, using existing data at: {preprocessing_output}")

    if not args.skip_filtering:
        run_leakage_filtering(args, preprocessing_output)
    else:
        logger.info("Skipping leakage filtering")

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Final output: gs://{args.gcs_bucket}/{args.output_path}/")
    logger.info("Files:")
    logger.info("  - train_data.pkl")
    logger.info("  - val_data.pkl")
    logger.info("  - test_data.pkl")
    logger.info("  - filtering_metadata.json")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
