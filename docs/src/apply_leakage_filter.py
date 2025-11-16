#!/usr/bin/env python3
"""
Apply diagnosis leakage filtering to preprocessed Phase 1 data

This script:
1. Loads preprocessed data from GCS (train/val/test splits)
2. Loads CheXpert labels from MIMIC-CXR metadata
3. Applies leakage filter to remove diagnosis information from pseudo-notes
4. Saves filtered data back to GCS

Usage:
    python src/apply_leakage_filter.py \
        --gcs-bucket bergermimiciv \
        --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
        --gcs-project-id YOUR_PROJECT_ID \
        --input-path processed/phase1_preprocess \
        --output-path processed/phase1_filtered \
        --aggressive
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import logging
from tqdm import tqdm
from io import BytesIO

# Import leakage filter
from leakage_filt_util import DiagnosisLeakageFilter

# Import Google Cloud Storage
from google.cloud import storage

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GCSHelper:
    """Simple GCS helper for reading/writing data"""

    def __init__(self, gcs_bucket: str, gcs_cxr_bucket: str, project_id: str):
        self.gcs_client = storage.Client(project=project_id)
        self.bucket = self.gcs_client.bucket(gcs_bucket)
        self.cxr_bucket = self.gcs_client.bucket(gcs_cxr_bucket, user_project=project_id)
        logger.info(f"Initialized GCS client for buckets: {gcs_bucket}, {gcs_cxr_bucket}")

    def read_pickle(self, path: str):
        """Read pickle file from GCS"""
        logger.info(f"Reading pickle from: gs://{self.bucket.name}/{path}")
        blob = self.bucket.blob(path)
        data = blob.download_as_bytes()
        return pickle.loads(data)

    def write_pickle(self, data, path: str):
        """Write pickle file to GCS"""
        logger.info(f"Writing pickle to: gs://{self.bucket.name}/{path}")
        blob = self.bucket.blob(path)
        pickled_data = pickle.dumps(data)
        blob.upload_from_string(pickled_data)

    def read_csv(self, path: str, from_cxr_bucket: bool = False, **kwargs):
        """Read CSV from GCS"""
        bucket = self.cxr_bucket if from_cxr_bucket else self.bucket
        logger.info(f"Reading CSV from: gs://{bucket.name}/{path}")
        blob = bucket.blob(path)
        data = blob.download_as_bytes()
        return pd.read_csv(BytesIO(data), **kwargs)

    def write_json(self, data, path: str):
        """Write JSON to GCS"""
        logger.info(f"Writing JSON to: gs://{self.bucket.name}/{path}")
        blob = self.bucket.blob(path)
        blob.upload_from_string(json.dumps(data, indent=2))


class LeakageFilterPipeline:
    """Apply leakage filtering to preprocessed data"""

    def __init__(self,
                 gcs_helper: GCSHelper,
                 use_nlp_model: bool = False,
                 aggressive: bool = True):
        self.gcs_helper = gcs_helper
        self.filter = DiagnosisLeakageFilter(use_nlp_model=use_nlp_model)
        self.aggressive = aggressive
        self.chexpert_labels = None

    def load_chexpert_labels(self) -> pd.DataFrame:
        """Load CheXpert labels from MIMIC-CXR metadata"""
        logger.info("Loading CheXpert labels from MIMIC-CXR...")

        # Load CheXpert labels
        chexpert_df = self.gcs_helper.read_csv(
            'mimic-cxr-2.0.0-chexpert.csv.gz',
            from_cxr_bucket=True,
            compression='gzip'
        )

        # CheXpert label columns
        label_columns = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]

        logger.info(f"Loaded CheXpert labels for {len(chexpert_df)} studies")
        self.chexpert_labels = chexpert_df
        return chexpert_df

    def get_positive_findings(self, study_id: int) -> List[str]:
        """Get list of positive findings for a study"""
        if self.chexpert_labels is None:
            return []

        # Find study in CheXpert labels
        study_labels = self.chexpert_labels[
            self.chexpert_labels['study_id'] == study_id
        ]

        if study_labels.empty:
            return []

        # Get positive findings (value = 1.0)
        positive_findings = []
        label_columns = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
            'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
            'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
            'Pneumonia', 'Pneumothorax', 'Support Devices'
        ]

        for col in label_columns:
            if col in study_labels.columns:
                value = study_labels[col].values[0]
                if value == 1.0:
                    positive_findings.append(col)

        return positive_findings

    def filter_record(self, record: Dict) -> Dict:
        """Apply leakage filtering to a single record"""

        # Get positive findings for this study
        study_id = record.get('study_id')
        positive_findings = self.get_positive_findings(study_id)

        # Filter the enhanced note
        original_note = record.get('enhanced_note', '')
        filtered_note, filter_stats = self.filter.filter_clinical_note(
            original_note,
            positive_findings,
            aggressive=self.aggressive
        )

        # Update record with filtered note
        filtered_record = record.copy()
        filtered_record['enhanced_note'] = filtered_note
        filtered_record['original_note'] = original_note  # Keep for comparison
        filtered_record['filter_stats'] = filter_stats
        filtered_record['positive_findings'] = positive_findings

        # Update labels with CheXpert findings
        if 'labels' not in filtered_record:
            filtered_record['labels'] = {}
        filtered_record['labels']['chexpert_labels'] = positive_findings

        return filtered_record

    def filter_split(self, split_name: str, records: List[Dict]) -> List[Dict]:
        """Filter all records in a split"""
        logger.info(f"Filtering {split_name} split ({len(records)} records)...")

        filtered_records = []
        stats_summary = {
            'total_records': len(records),
            'avg_reduction_rate': 0,
            'records_with_findings': 0,
            'avg_sentence_removal_rate': 0
        }

        for record in tqdm(records, desc=f"Filtering {split_name}"):
            try:
                filtered_record = self.filter_record(record)
                filtered_records.append(filtered_record)

                # Collect statistics
                if filtered_record.get('filter_stats'):
                    stats = filtered_record['filter_stats']
                    stats_summary['avg_reduction_rate'] += (
                        1 - stats['filtered_length'] / max(stats['original_length'], 1)
                    )
                    stats_summary['avg_sentence_removal_rate'] += stats['removal_rate']

                if filtered_record.get('positive_findings'):
                    stats_summary['records_with_findings'] += 1

            except Exception as e:
                logger.error(f"Error filtering record {record.get('study_id')}: {e}")
                # Keep original record on error
                filtered_records.append(record)

        # Calculate averages
        n = len(records)
        if n > 0:
            stats_summary['avg_reduction_rate'] /= n
            stats_summary['avg_sentence_removal_rate'] /= n

        logger.info(f"{split_name} filtering complete!")
        logger.info(f"  Records with findings: {stats_summary['records_with_findings']}/{n}")
        logger.info(f"  Avg text reduction: {stats_summary['avg_reduction_rate']:.2%}")
        logger.info(f"  Avg sentence removal: {stats_summary['avg_sentence_removal_rate']:.2%}")

        return filtered_records, stats_summary

    def process_all_splits(self, input_path: str, output_path: str):
        """Process all data splits (train, val, test)"""

        # Load CheXpert labels first
        self.load_chexpert_labels()

        # Process each split
        splits = ['train', 'val', 'test']
        all_stats = {}

        for split_name in splits:
            logger.info("=" * 60)
            logger.info(f"Processing {split_name} split")
            logger.info("=" * 60)

            # Load split data
            input_file = f"{input_path}/{split_name}_data.pkl"
            try:
                records = self.gcs_helper.read_pickle(input_file)
                logger.info(f"Loaded {len(records)} records from {split_name} split")
            except Exception as e:
                logger.error(f"Could not load {split_name} split: {e}")
                continue

            # Filter records
            filtered_records, stats = self.filter_split(split_name, records)

            # Save filtered data
            output_file = f"{output_path}/{split_name}_data.pkl"
            self.gcs_helper.write_pickle(filtered_records, output_file)

            all_stats[split_name] = stats

        # Save overall statistics
        metadata = {
            'filtering_config': {
                'aggressive': self.aggressive,
                'use_nlp_model': self.filter.use_nlp_model
            },
            'split_statistics': all_stats,
            'total_records': sum(s['total_records'] for s in all_stats.values()),
            'total_with_findings': sum(s['records_with_findings'] for s in all_stats.values())
        }

        metadata_file = f"{output_path}/filtering_metadata.json"
        self.gcs_helper.write_json(metadata, metadata_file)

        logger.info("=" * 60)
        logger.info("Filtering pipeline complete!")
        logger.info(f"Total records processed: {metadata['total_records']}")
        logger.info(f"Records with CheXpert findings: {metadata['total_with_findings']}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Apply diagnosis leakage filtering to preprocessed data'
    )

    # GCS settings
    parser.add_argument('--gcs-bucket', type=str, required=True,
                       help='Main GCS bucket (e.g., bergermimiciv)')
    parser.add_argument('--gcs-cxr-bucket', type=str, required=True,
                       help='MIMIC-CXR bucket (e.g., mimic-cxr-jpg-2.1.0.physionet.org)')
    parser.add_argument('--gcs-project-id', type=str, required=True,
                       help='GCP project ID for requester pays')

    # Paths
    parser.add_argument('--input-path', type=str,
                       default='processed/phase1_preprocess',
                       help='Input path in GCS bucket')
    parser.add_argument('--output-path', type=str,
                       default='processed/phase1_filtered',
                       help='Output path in GCS bucket')

    # Filtering options
    parser.add_argument('--aggressive', action='store_true',
                       help='Use aggressive filtering mode')
    parser.add_argument('--use-nlp-model', action='store_true',
                       help='Use BioBERT for semantic filtering (slower)')

    args = parser.parse_args()

    # Initialize GCS helper
    gcs_helper = GCSHelper(
        args.gcs_bucket,
        args.gcs_cxr_bucket,
        args.gcs_project_id
    )

    # Initialize and run filtering pipeline
    pipeline = LeakageFilterPipeline(
        gcs_helper,
        use_nlp_model=args.use_nlp_model,
        aggressive=args.aggressive
    )

    pipeline.process_all_splits(args.input_path, args.output_path)

    logger.info("Done!")


if __name__ == "__main__":
    main()
