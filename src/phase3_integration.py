"""
Phase 3: Multi-Modal Integration and Final Dataset Preparation
Integrates Phase 2 enhanced outputs and prepares final training-ready datasets

Takes Phase 2 enhanced data and:
- Validates all modalities are present and properly formatted
- Creates comprehensive cross-modal integration
- Generates final training-ready datasets
- Produces detailed statistics and quality metrics
- Prepares data for model training with proper batching

Supports both GCS and local file systems
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import argparse
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

# Import from phase1 for consistency
from phase1_preprocess_streaming import DataConfig, GCSHelper

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ValidationStats:
    """Statistics for dataset validation"""
    total_records: int = 0
    valid_records: int = 0
    missing_images: int = 0
    missing_clinical: int = 0
    missing_text: int = 0
    missing_labels: int = 0
    avg_text_length: float = 0.0
    avg_enhanced_text_length: float = 0.0
    view_position_counts: Dict[str, int] = None

    def __post_init__(self):
        if self.view_position_counts is None:
            self.view_position_counts = {}


class DataQualityValidator:
    """
    Validate data quality and completeness across all modalities
    """

    def __init__(self):
        """Initialize Data Quality Validator"""
        self.required_fields = [
            'subject_id', 'study_id', 'dicom_id',
            'image', 'clinical_features', 'labels'
        ]
        self.phase2_fields = [
            'pseudo_note', 'enhanced_note', 'enhanced_text_tokens'
        ]

    def validate_record(self, record: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a single record for completeness and quality

        Args:
            record: Record to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check required Phase 1 fields
        for field in self.required_fields:
            if field not in record or record[field] is None:
                issues.append(f"Missing required field: {field}")

        # Check Phase 2 fields
        if not record.get('phase2_processed'):
            issues.append("Record not processed by Phase 2")

        for field in self.phase2_fields:
            if field not in record or record[field] is None:
                issues.append(f"Missing Phase 2 field: {field}")

        # Validate image tensor
        if 'image' in record and record['image'] is not None:
            img = record['image']
            if not torch.is_tensor(img):
                issues.append("Image is not a tensor")
            elif img.shape != torch.Size([3, 518, 518]):
                issues.append(f"Invalid image shape: {img.shape}")

        # Validate clinical features
        if 'clinical_features' in record and record['clinical_features'] is not None:
            cf = record['clinical_features']
            if not torch.is_tensor(cf):
                issues.append("Clinical features is not a tensor")

        # Validate enhanced text tokens
        if 'enhanced_text_tokens' in record and record['enhanced_text_tokens'] is not None:
            tokens = record['enhanced_text_tokens']
            if not isinstance(tokens, dict):
                issues.append("Enhanced text tokens is not a dict")
            elif 'input_ids' not in tokens or 'attention_mask' not in tokens:
                issues.append("Enhanced text tokens missing input_ids or attention_mask")

        is_valid = len(issues) == 0
        return is_valid, issues

    def validate_split(self, records: List[Dict], split_name: str) -> ValidationStats:
        """
        Validate an entire data split

        Args:
            records: List of records to validate
            split_name: Name of the split (train/val/test)

        Returns:
            ValidationStats with comprehensive statistics
        """
        stats = ValidationStats()
        stats.total_records = len(records)
        stats.view_position_counts = defaultdict(int)

        text_lengths = []
        enhanced_text_lengths = []

        for record in tqdm(records, desc=f"Validating {split_name}"):
            is_valid, issues = self.validate_record(record)

            if is_valid:
                stats.valid_records += 1
            else:
                # Count specific issues
                if any('image' in issue for issue in issues):
                    stats.missing_images += 1
                if any('clinical' in issue for issue in issues):
                    stats.missing_clinical += 1
                if any('text' in issue for issue in issues):
                    stats.missing_text += 1
                if any('labels' in issue for issue in issues):
                    stats.missing_labels += 1

            # Collect text length statistics
            if 'pseudo_note' in record and record['pseudo_note']:
                text_lengths.append(len(record['pseudo_note']))

            if 'enhanced_note' in record and record['enhanced_note']:
                enhanced_text_lengths.append(len(record['enhanced_note']))

            # Count view positions
            if 'labels' in record and 'view_position' in record['labels']:
                view_pos = record['labels']['view_position']
                stats.view_position_counts[view_pos] += 1

        # Calculate averages
        if text_lengths:
            stats.avg_text_length = np.mean(text_lengths)
        if enhanced_text_lengths:
            stats.avg_enhanced_text_length = np.mean(enhanced_text_lengths)

        return stats


class MultiModalIntegrator:
    """
    Integrate all modalities into final training-ready format
    Combines vision, text, and clinical features with proper alignment
    """

    def __init__(self, config: DataConfig):
        """
        Initialize Multi-Modal Integrator

        Args:
            config: DataConfig with processing settings
        """
        self.config = config

    def integrate_record(self, record: Dict) -> Dict:
        """
        Integrate all modalities for a single record
        Creates final model-ready format

        Args:
            record: Enhanced record from Phase 2

        Returns:
            Integrated record ready for model training
        """
        integrated = {
            # Identifiers
            'subject_id': record['subject_id'],
            'study_id': record['study_id'],
            'dicom_id': record['dicom_id'],

            # Vision modality (BiomedCLIP input)
            'image': record['image'],  # [3, 518, 518]
            'attention_regions': record.get('attention_regions', None),

            # Text modality (Clinical ModernBERT input)
            'text_input_ids': record['enhanced_text_tokens']['input_ids'],
            'text_attention_mask': record['enhanced_text_tokens']['attention_mask'],
            'pseudo_note': record['pseudo_note'],  # Raw text for analysis
            'enhanced_note': record['enhanced_note'],  # Raw enhanced text

            # Clinical features (structured data)
            'clinical_features': record['clinical_features'],

            # Labels and metadata
            'labels': record['labels'],
            'view_position': record['labels'].get('view_position', 'UNKNOWN'),

            # RAG knowledge
            'retrieved_knowledge': record.get('retrieved_knowledge', []),

            # Processing flags
            'phase1_processed': True,
            'phase2_processed': record.get('phase2_processed', False),
            'phase3_integrated': True
        }

        return integrated

    def create_batch_indices(self, num_records: int, batch_size: int = 32) -> List[Tuple[int, int]]:
        """
        Create batch indices for efficient data loading

        Args:
            num_records: Total number of records
            batch_size: Batch size for training

        Returns:
            List of (start_idx, end_idx) tuples
        """
        indices = []
        for i in range(0, num_records, batch_size):
            end_idx = min(i + batch_size, num_records)
            indices.append((i, end_idx))
        return indices


class Phase3Processor:
    """
    Phase 3 Processor: Multi-Modal Integration and Final Dataset Preparation
    Integrates Phase 2 outputs into final training-ready datasets
    """

    def __init__(self, config: DataConfig):
        """
        Initialize Phase 3 Processor

        Args:
            config: DataConfig from phase1/phase2
        """
        self.config = config
        self.gcs_helper = GCSHelper(config)

        # Initialize components
        self.validator = DataQualityValidator()
        self.integrator = MultiModalIntegrator(config)

        logger.info("Phase 3 Processor initialized")
        logger.info(f"  Mode: {'GCS' if config.use_gcs else 'Local'}")
        logger.info(f"  Input path: {config.output_path}")

    def load_enhanced_split(self, split_name: str, use_small_sample: bool = False) -> List[Dict]:
        """
        Load Phase 2 enhanced data for a split

        Args:
            split_name: 'train', 'val', or 'test'
            use_small_sample: Load small sample files

        Returns:
            List of enhanced records from Phase 2
        """
        # Construct input file path
        if use_small_sample:
            filename = f"{split_name}_small_enhanced.pt"
        else:
            filename = f"{split_name}_data_enhanced.pt"

        if self.config.use_gcs:
            input_path = f"{self.config.output_path}/{filename}"
        else:
            input_path = Path(self.config.output_path).expanduser() / filename
            input_path = str(input_path)

        logger.info(f"Loading enhanced {split_name} split from: {input_path}")

        try:
            records = self.gcs_helper.read_torch(input_path)
            logger.info(f"Loaded {len(records)} enhanced records from {split_name} split")
            return records
        except Exception as e:
            logger.error(f"Error loading enhanced {split_name} split: {e}")
            raise

    def process_split(self, split_name: str, use_small_sample: bool = False) -> Tuple[List[Dict], ValidationStats]:
        """
        Process and validate a single split

        Args:
            split_name: 'train', 'val', or 'test'
            use_small_sample: Use small sample files

        Returns:
            Tuple of (integrated_records, validation_stats)
        """
        # Load enhanced records from Phase 2
        enhanced_records = self.load_enhanced_split(split_name, use_small_sample)

        # Validate data quality
        logger.info(f"Validating {split_name} split...")
        validation_stats = self.validator.validate_split(enhanced_records, split_name)

        # Log validation results
        self._log_validation_stats(split_name, validation_stats)

        # Integrate modalities
        logger.info(f"Integrating modalities for {split_name} split...")
        integrated_records = []

        for record in tqdm(enhanced_records, desc=f"Integrating {split_name}"):
            is_valid, issues = self.validator.validate_record(record)

            if is_valid:
                integrated = self.integrator.integrate_record(record)
                integrated_records.append(integrated)
            else:
                logger.debug(f"Skipping invalid record {record.get('dicom_id')}: {issues}")

        logger.info(f"Integrated {len(integrated_records)}/{len(enhanced_records)} records")

        return integrated_records, validation_stats

    def process_all_splits(self, use_small_sample: bool = False) -> Dict[str, Any]:
        """
        Process all splits and create final datasets

        Args:
            use_small_sample: Use small sample files for testing

        Returns:
            Dictionary with processing statistics
        """
        logger.info("=" * 60)
        logger.info("Phase 3: Multi-Modal Integration")
        logger.info("=" * 60)

        splits = ['train', 'val', 'test']
        all_stats = {}

        for split_name in splits:
            logger.info(f"\nProcessing {split_name} split...")

            try:
                # Process and validate split
                integrated_records, validation_stats = self.process_split(
                    split_name, use_small_sample
                )

                # Save integrated records
                if use_small_sample:
                    output_filename = f"{split_name}_small_final.pt"
                else:
                    output_filename = f"{split_name}_final.pt"

                if self.config.use_gcs:
                    output_path = f"{self.config.output_path}/{output_filename}"
                else:
                    output_dir = Path(self.config.output_path).expanduser()
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / output_filename
                    output_path = str(output_path)

                logger.info(f"Saving integrated {split_name} split to: {output_path}")
                self.gcs_helper.write_torch(integrated_records, output_path)

                # Store stats
                all_stats[split_name] = {
                    'total_records': validation_stats.total_records,
                    'valid_records': validation_stats.valid_records,
                    'validation_rate': validation_stats.valid_records / validation_stats.total_records if validation_stats.total_records > 0 else 0,
                    'avg_text_length': validation_stats.avg_text_length,
                    'avg_enhanced_text_length': validation_stats.avg_enhanced_text_length,
                    'view_position_counts': dict(validation_stats.view_position_counts)
                }

                # Log sample record
                if integrated_records:
                    self._log_sample_record(split_name, integrated_records[0])

            except Exception as e:
                logger.error(f"Error processing {split_name} split: {e}")
                continue

        logger.info("=" * 60)
        logger.info("Phase 3 Complete!")
        logger.info("=" * 60)

        return all_stats

    def save_final_metadata(self, all_stats: Dict[str, Any]):
        """
        Save comprehensive metadata for the final integrated datasets

        Args:
            all_stats: Statistics from all splits
        """
        metadata = {
            'phase': 3,
            'timestamp': datetime.now().isoformat(),
            'description': 'Multi-modal integrated datasets ready for model training',
            'splits': all_stats,
            'config': {
                'max_text_length': self.config.max_text_length,
                'image_size': self.config.image_size,
                'top_k_retrieval': self.config.top_k_retrieval
            },
            'modalities': {
                'vision': {
                    'encoder': 'BiomedCLIP-CXR',
                    'image_size': self.config.image_size,
                    'format': 'tensor [3, 518, 518]'
                },
                'text': {
                    'encoder': 'Clinical ModernBERT',
                    'max_length': self.config.max_text_length,
                    'format': 'tokenized (input_ids, attention_mask)'
                },
                'clinical': {
                    'format': 'tensor of normalized features',
                    'features': self.config.clinical_features
                }
            },
            'data_quality': {
                split: {
                    'validation_rate': f"{stats['validation_rate']*100:.2f}%",
                    'valid_records': stats['valid_records'],
                    'total_records': stats['total_records']
                }
                for split, stats in all_stats.items()
            }
        }

        # Construct metadata path
        if self.config.use_gcs:
            metadata_path = f"{self.config.output_path}/phase3_metadata.json"
        else:
            output_dir = Path(self.config.output_path).expanduser()
            metadata_path = output_dir / "phase3_metadata.json"
            metadata_path = str(metadata_path)

        # Save metadata
        metadata_json = json.dumps(metadata, indent=2)
        if self.config.use_gcs and self.gcs_helper.output_bucket:
            blob = self.gcs_helper.output_bucket.blob(metadata_path.replace(f"gs://{self.config.output_gcs_bucket}/", ""))
            blob.upload_from_string(metadata_json)
        else:
            with open(metadata_path, 'w') as f:
                f.write(metadata_json)

        logger.info(f"Saved Phase 3 metadata to: {metadata_path}")

    def generate_dataset_report(self, all_stats: Dict[str, Any]):
        """
        Generate comprehensive dataset report

        Args:
            all_stats: Statistics from all splits
        """
        logger.info("\n" + "=" * 60)
        logger.info("FINAL DATASET REPORT")
        logger.info("=" * 60)

        total_records = sum(stats['total_records'] for stats in all_stats.values())
        total_valid = sum(stats['valid_records'] for stats in all_stats.values())

        logger.info(f"\nOverall Statistics:")
        logger.info(f"  Total records:     {total_records:,}")
        logger.info(f"  Valid records:     {total_valid:,}")
        logger.info(f"  Validation rate:   {total_valid/total_records*100:.2f}%")

        logger.info(f"\nSplit Distribution:")
        for split_name, stats in all_stats.items():
            logger.info(f"  {split_name.capitalize():5s}: {stats['valid_records']:,} records ({stats['validation_rate']*100:.1f}% valid)")

        logger.info(f"\nText Statistics:")
        for split_name, stats in all_stats.items():
            logger.info(f"  {split_name.capitalize():5s}:")
            logger.info(f"    Avg pseudo-note length:  {stats['avg_text_length']:.0f} chars")
            logger.info(f"    Avg enhanced note length: {stats['avg_enhanced_text_length']:.0f} chars")

        logger.info(f"\nView Position Distribution:")
        all_views = defaultdict(int)
        for stats in all_stats.values():
            for view, count in stats['view_position_counts'].items():
                all_views[view] += count

        for view, count in sorted(all_views.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {view:10s}: {count:,} ({count/total_valid*100:.1f}%)")

        logger.info("\n" + "=" * 60)
        logger.info("Datasets ready for model training!")
        logger.info("=" * 60)

    def _log_validation_stats(self, split_name: str, stats: ValidationStats):
        """Log validation statistics"""
        logger.info(f"\nValidation Results for {split_name}:")
        logger.info(f"  Total records:      {stats.total_records:,}")
        logger.info(f"  Valid records:      {stats.valid_records:,}")
        logger.info(f"  Validation rate:    {stats.valid_records/stats.total_records*100:.2f}%")

        if stats.missing_images > 0:
            logger.warning(f"  Missing images:     {stats.missing_images:,}")
        if stats.missing_clinical > 0:
            logger.warning(f"  Missing clinical:   {stats.missing_clinical:,}")
        if stats.missing_text > 0:
            logger.warning(f"  Missing text:       {stats.missing_text:,}")

    def _log_sample_record(self, split_name: str, record: Dict):
        """Log a sample integrated record"""
        logger.info(f"\nSample integrated record from {split_name}:")
        logger.info(f"  Subject ID:         {record['subject_id']}")
        logger.info(f"  Study ID:           {record['study_id']}")
        logger.info(f"  DICOM ID:           {record['dicom_id']}")
        logger.info(f"  Image shape:        {record['image'].shape}")
        logger.info(f"  Text tokens shape:  {record['text_input_ids'].shape}")
        logger.info(f"  Clinical features:  {record['clinical_features'].shape if torch.is_tensor(record['clinical_features']) else 'N/A'}")
        logger.info(f"  View position:      {record['view_position']}")
        logger.info(f"  Pseudo-note (first 150 chars):")
        logger.info(f"    {record['pseudo_note'][:150]}...")


def main():
    """Main entry point for Phase 3"""
    parser = argparse.ArgumentParser(
        description='Phase 3: Multi-Modal Integration and Final Dataset Preparation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output paths
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Path to Phase 2 output directory (local or GCS path)'
    )

    # GCS settings
    parser.add_argument(
        '--gcs-bucket',
        type=str,
        default=None,
        help='GCS bucket name (if using GCS)'
    )

    parser.add_argument(
        '--gcs-project-id',
        type=str,
        default=None,
        help='GCP project ID for requester pays'
    )

    # Sample data option
    parser.add_argument(
        '--use-small-sample',
        action='store_true',
        help='Process small sample files (e.g., train_small_enhanced.pt) for testing'
    )

    args = parser.parse_args()

    # Create configuration
    config = DataConfig()

    # Set paths
    config.output_path = args.input_path

    # Set GCS settings
    config.use_gcs = args.gcs_bucket is not None
    config.gcs_bucket = args.gcs_bucket
    config.output_gcs_bucket = args.gcs_bucket
    config.gcs_project_id = args.gcs_project_id

    logger.info("=" * 60)
    logger.info("Phase 3: Multi-Modal Integration and Final Dataset Preparation")
    logger.info("=" * 60)
    logger.info(f"Mode: {'GCS' if config.use_gcs else 'Local'}")
    if config.use_gcs:
        logger.info(f"Bucket: {config.gcs_bucket}")
    logger.info(f"Input path: {config.output_path}")
    logger.info(f"Use small sample: {args.use_small_sample}")
    logger.info("=" * 60)

    # Create processor
    processor = Phase3Processor(config)

    # Process all splits
    all_stats = processor.process_all_splits(use_small_sample=args.use_small_sample)

    # Save metadata
    processor.save_final_metadata(all_stats)

    # Generate final report
    processor.generate_dataset_report(all_stats)

    logger.info("Phase 3 processing complete!")


if __name__ == "__main__":
    main()
