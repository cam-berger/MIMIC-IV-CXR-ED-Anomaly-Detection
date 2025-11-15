"""
Phase 1 CXR-PRO Adapter: Integrate cleaned radiology impressions into preprocessing pipeline

Enhances the existing Phase 1 preprocessing by adding CXR-PRO cleaned radiology impressions.

Data Flow:
1. Load Phase 1 preprocessed data (train/val/test .pt files)
2. Match study_ids to CXR-PRO cleaned impressions
3. Add radiology impressions as a fourth text modality
4. Save enhanced data with radiology context

New Record Structure:
{
    # Existing Phase 1 fields
    'subject_id': int,
    'study_id': int,
    'dicom_id': str,
    'image': torch.Tensor,
    'clinical_features': torch.Tensor,
    'labels': Dict,

    # NEW: CXR-PRO radiology impressions
    'radiology_impression': str,              # Original impression
    'radiology_impression_cleaned': str,      # Prior-free impression
    'radiology_tokens': {                      # Tokenized for text encoder
        'input_ids': torch.Tensor,
        'attention_mask': torch.Tensor
    },
    'has_radiology_report': bool,             # Whether impression was found
    'prior_removal_stats': Dict               # Metadata about prior removal
}
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from cxr_pro_integration import CXRProIntegrator, CXRProConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Phase1CXRProAdapter:
    """
    Adapts CXR-PRO data to integrate with existing Phase 1 preprocessing

    Workflow:
    1. Load existing Phase 1 .pt files (train/val/test)
    2. Load CXR-PRO cleaned impressions (CSV or process with GILBERT)
    3. Match impressions to study_ids
    4. Tokenize impressions for text encoder
    5. Save enhanced .pt files with radiology context
    """

    def __init__(self,
                 phase1_data_path: str,
                 cxr_pro_config: CXRProConfig,
                 text_encoder: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 max_length: int = 512):
        """
        Initialize adapter

        Args:
            phase1_data_path: Path to Phase 1 .pt files
            cxr_pro_config: CXR-PRO configuration
            text_encoder: Text encoder model for tokenization
            max_length: Max sequence length for radiology impressions
        """
        self.phase1_data_path = phase1_data_path
        self.cxr_pro_config = cxr_pro_config
        self.max_length = max_length

        # Initialize tokenizer for radiology impressions
        logger.info(f"Loading tokenizer: {text_encoder}")
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)

        # Initialize CXR-PRO integrator
        self.cxr_pro_integrator = CXRProIntegrator(cxr_pro_config)

        # Statistics
        self.stats = {
            "total_records": 0,
            "records_with_radiology": 0,
            "records_without_radiology": 0,
            "average_impression_length": 0.0,
            "impressions_with_priors_removed": 0
        }

    def load_phase1_data(self, split: str) -> List[Dict]:
        """
        Load Phase 1 preprocessed data

        Args:
            split: "train", "val", or "test"

        Returns:
            List of data records
        """
        file_path = os.path.join(self.phase1_data_path, f"{split}_data.pt")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Phase 1 data not found: {file_path}")

        logger.info(f"Loading Phase 1 {split} data from {file_path}")
        data = torch.load(file_path)

        logger.info(f"Loaded {len(data)} records from {split} split")
        return data

    def load_cxr_pro_impressions(self, split: str) -> pd.DataFrame:
        """
        Load CXR-PRO cleaned impressions

        Args:
            split: "train" or "test" (CXR-PRO only has these two)

        Returns:
            DataFrame with study_id, subject_id, impression columns
        """
        logger.info(f"Loading CXR-PRO {split} impressions...")

        if split in ["train", "val"]:
            # Both train and val come from CXR-PRO train set
            df = self.cxr_pro_integrator.load_cxr_pro_preprocessed("train")
        else:  # test
            df = self.cxr_pro_integrator.load_cxr_pro_preprocessed("test")

        logger.info(f"Loaded {len(df)} CXR-PRO impressions for {split}")
        return df

    def match_impression_to_record(self,
                                   record: Dict,
                                   impressions_df: pd.DataFrame) -> Optional[str]:
        """
        Match a Phase 1 record to its CXR-PRO cleaned impression

        Args:
            record: Phase 1 data record
            impressions_df: CXR-PRO impressions DataFrame

        Returns:
            Cleaned impression text or None if not found
        """
        study_id = record.get('study_id')
        if study_id is None:
            return None

        # Find matching impression
        matched = impressions_df[impressions_df['study_id'] == study_id]

        if len(matched) == 0:
            return None

        # Get the impression (CXR-PRO uses 'impression' column for cleaned text)
        impression = matched.iloc[0].get('impression', None)
        return impression

    def tokenize_impression(self, impression: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize radiology impression for text encoder

        Args:
            impression: Radiology impression text

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        if not impression or len(impression.strip()) == 0:
            # Return empty tokens
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long)
            }

        # Tokenize
        encoded = self.tokenizer(
            impression,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }

    def enhance_record_with_radiology(self,
                                      record: Dict,
                                      impressions_df: pd.DataFrame) -> Dict:
        """
        Add CXR-PRO cleaned impression to a Phase 1 record

        Args:
            record: Original Phase 1 record
            impressions_df: CXR-PRO impressions

        Returns:
            Enhanced record with radiology fields
        """
        # Match impression
        cleaned_impression = self.match_impression_to_record(record, impressions_df)

        if cleaned_impression:
            # Tokenize
            radiology_tokens = self.tokenize_impression(cleaned_impression)

            # Add to record
            record['radiology_impression_cleaned'] = cleaned_impression
            record['radiology_tokens'] = radiology_tokens
            record['has_radiology_report'] = True

            # Update stats
            self.stats['records_with_radiology'] += 1
            self.stats['average_impression_length'] += len(cleaned_impression)

        else:
            # No impression found
            record['radiology_impression_cleaned'] = ""
            record['radiology_tokens'] = self.tokenize_impression("")
            record['has_radiology_report'] = False

            # Update stats
            self.stats['records_without_radiology'] += 1

        self.stats['total_records'] += 1

        return record

    def process_split(self, split: str) -> List[Dict]:
        """
        Process a complete data split (train/val/test)

        Args:
            split: "train", "val", or "test"

        Returns:
            Enhanced data records
        """
        logger.info(f"Processing {split} split...")

        # Load Phase 1 data
        phase1_data = self.load_phase1_data(split)

        # Load CXR-PRO impressions
        # Note: CXR-PRO only has train/test, map val to train
        cxr_pro_split = "train" if split in ["train", "val"] else "test"
        impressions_df = self.load_cxr_pro_impressions(cxr_pro_split)

        # Enhance each record
        enhanced_data = []
        for record in tqdm(phase1_data, desc=f"Enhancing {split}"):
            enhanced_record = self.enhance_record_with_radiology(record, impressions_df)
            enhanced_data.append(enhanced_record)

        return enhanced_data

    def save_enhanced_data(self, data: List[Dict], split: str, output_path: str):
        """
        Save enhanced data with radiology impressions

        Args:
            data: Enhanced data records
            split: "train", "val", or "test"
            output_path: Output directory
        """
        os.makedirs(output_path, exist_ok=True)

        # Save .pt file
        output_file = os.path.join(output_path, f"{split}_data_with_radiology.pt")
        torch.save(data, output_file)
        logger.info(f"Saved {len(data)} records to {output_file}")

    def process_all_splits(self, output_path: str):
        """
        Process all splits (train/val/test) and save enhanced data

        Args:
            output_path: Output directory for enhanced data
        """
        for split in ["train", "val", "test"]:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {split.upper()} split")
            logger.info(f"{'='*60}")

            # Reset stats for this split
            split_stats = {
                "total_records": 0,
                "records_with_radiology": 0,
                "records_without_radiology": 0,
                "average_impression_length": 0.0
            }
            self.stats = split_stats

            # Process split
            enhanced_data = self.process_split(split)

            # Calculate average impression length
            if self.stats['records_with_radiology'] > 0:
                self.stats['average_impression_length'] /= self.stats['records_with_radiology']

            # Save enhanced data
            self.save_enhanced_data(enhanced_data, split, output_path)

            # Report statistics
            logger.info(f"\n{split.upper()} Split Statistics:")
            logger.info(f"  Total records: {self.stats['total_records']}")
            logger.info(f"  With radiology: {self.stats['records_with_radiology']} "
                       f"({100*self.stats['records_with_radiology']/max(self.stats['total_records'],1):.1f}%)")
            logger.info(f"  Without radiology: {self.stats['records_without_radiology']} "
                       f"({100*self.stats['records_without_radiology']/max(self.stats['total_records'],1):.1f}%)")
            logger.info(f"  Avg impression length: {self.stats['average_impression_length']:.1f} chars")

            # Save statistics
            stats_file = os.path.join(output_path, f"{split}_radiology_stats.json")
            with open(stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("CXR-PRO integration complete!")
        logger.info(f"Enhanced data saved to: {output_path}")
        logger.info(f"{'='*60}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Integrate CXR-PRO cleaned radiology impressions into Phase 1 data"
    )

    parser.add_argument('--phase1-data-path', type=str, required=True,
                       help='Path to Phase 1 preprocessed data (.pt files)')
    parser.add_argument('--cxr-pro-train', type=str, required=True,
                       help='Path to CXR-PRO mimic_train_impressions.csv')
    parser.add_argument('--cxr-pro-test', type=str, required=True,
                       help='Path to CXR-PRO mimic_test_impressions.csv')
    parser.add_argument('--output-path', type=str, required=True,
                       help='Output path for enhanced data')

    # Text encoder settings
    parser.add_argument('--text-encoder', type=str,
                       default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
                       help='Text encoder for tokenization')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Max sequence length for impressions')

    # GCS settings
    parser.add_argument('--use-gcs', action='store_true',
                       help='Use Google Cloud Storage')
    parser.add_argument('--gcs-bucket', type=str,
                       help='GCS bucket name')
    parser.add_argument('--gcs-project-id', type=str,
                       help='GCP project ID')

    args = parser.parse_args()

    # Configure CXR-PRO
    cxr_pro_config = CXRProConfig(
        cxr_pro_train_path=args.cxr_pro_train,
        cxr_pro_test_path=args.cxr_pro_test,
        use_gcs=args.use_gcs,
        gcs_bucket=args.gcs_bucket,
        gcs_project_id=args.gcs_project_id
    )

    # Initialize adapter
    adapter = Phase1CXRProAdapter(
        phase1_data_path=args.phase1_data_path,
        cxr_pro_config=cxr_pro_config,
        text_encoder=args.text_encoder,
        max_length=args.max_length
    )

    # Process all splits
    adapter.process_all_splits(args.output_path)


if __name__ == "__main__":
    main()
