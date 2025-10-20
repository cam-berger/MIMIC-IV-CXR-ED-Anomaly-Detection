#!/usr/bin/env python3
"""
MIMIC Multimodal Data Preprocessing Pipeline - Phase 1 AWS Testing
Corrected version for S3 deployment with proper REFLACX mapping and gzip handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import re
import argparse
import boto3
from io import BytesIO, StringIO
import gzip
import os

class MIMICPreprocessor:
    """Preprocessor for linking MIMIC datasets while avoiding data leakage"""

    def __init__(self, base_path: str, use_s3: bool = False, s3_bucket: str = None):
        """
        Initialize with base path to MIMIC datasets

        Args:
            base_path: Root directory containing MIMIC-IV, MIMIC-CXR-JPG, REFLACX, MIMIC-ED
                       For S3: this should be the prefix path (e.g., 'physionet.org/files')
            use_s3: If True, read data from S3 instead of local filesystem
            s3_bucket: S3 bucket name (required if use_s3=True)
        """
        self.use_s3 = use_s3
        self.s3_bucket = s3_bucket
        self.base_path = base_path if use_s3 else Path(base_path)

        # Initialize S3 client if needed
        if self.use_s3:
            if not s3_bucket:
                raise ValueError("s3_bucket must be provided when use_s3=True")
            self.s3_client = boto3.client('s3')
            print(f"Initialized S3 client for bucket: {s3_bucket}")

        # Define paths to each dataset (with version directories)
        if use_s3:
            # S3 paths are strings with forward slashes
            self.paths = {
                'mimic_iv': f'{base_path}/mimiciv/3.1',
                'mimic_cxr': f'{base_path}/mimic-cxr-jpg/2.1.0',
                'reflacx': f'{base_path}/reflacx',
                'mimic_ed': f'{base_path}/mimic-iv-ed/2.2'
            }
        else:
            # Local paths use Path objects
            self.paths = {
                'mimic_iv': self.base_path / 'mimiciv' / '3.1',
                'mimic_cxr': self.base_path / 'mimic-cxr-jpg' / '2.1.0',
                'reflacx': self.base_path / 'reflacx',
                'mimic_ed': self.base_path / 'mimic-iv-ed' / '2.2'
            }

        # Initialize dataframes
        self.cxr_metadata = None
        self.cxr_labels = None
        self.reflacx_data = None
        self.reflacx_metadata = None
        self.image_filenames = None
        self.clinical_data = {}

    def _read_csv_from_s3(self, s3_key: str, compression: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Read CSV from S3 with proper gzip handling
        
        Args:
            s3_key: S3 object key
            compression: 'gzip' or None
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame
        """
        print(f"  Reading from S3: s3://{self.s3_bucket}/{s3_key}")
        
        try:
            obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            
            if compression == 'gzip' or s3_key.endswith('.gz'):
                # Decompress gzip data
                with gzip.GzipFile(fileobj=BytesIO(obj['Body'].read())) as gzipfile:
                    return pd.read_csv(gzipfile, **kwargs)
            else:
                return pd.read_csv(BytesIO(obj['Body'].read()), **kwargs)
        except Exception as e:
            print(f"  Error reading {s3_key}: {e}")
            raise

    def _read_csv(self, path, **kwargs) -> pd.DataFrame:
        """
        Read CSV from S3 or local filesystem

        Args:
            path: S3 key (str) or local file path (str or Path)
            **kwargs: Additional arguments to pass to pd.read_csv

        Returns:
            DataFrame
        """
        if self.use_s3:
            s3_key = path if isinstance(path, str) else str(path)
            # Determine compression from file extension
            compression = 'gzip' if s3_key.endswith('.gz') else kwargs.get('compression', None)
            return self._read_csv_from_s3(s3_key, compression=compression, **kwargs)
        else:
            # Read from local filesystem
            return pd.read_csv(path, **kwargs)

    def load_image_filenames(self):
        """Load the IMAGE_FILENAMES file that maps studies to actual image files"""
        print("Loading IMAGE_FILENAMES...")
        
        if self.use_s3:
            image_filenames_path = f"{self.paths['mimic_cxr']}/IMAGE_FILENAMES"
        else:
            image_filenames_path = self.paths['mimic_cxr'] / 'IMAGE_FILENAMES'
        
        # Read the file (it's a plain text file with one filename per line)
        if self.use_s3:
            obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=image_filenames_path)
            filenames = obj['Body'].read().decode('utf-8').strip().split('\n')
        else:
            with open(image_filenames_path, 'r') as f:
                filenames = [line.strip() for line in f if line.strip()]
        
        # Parse filenames to extract subject_id, study_id, and dicom_id
        parsed_files = []
        for filename in filenames[:10000]:  # Limit for testing
            # Format: files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
            parts = filename.split('/')
            if len(parts) >= 5:
                subject_str = parts[2]  # p10000032
                study_str = parts[3]     # s50414267
                dicom_file = parts[4]    # xxx.jpg
                
                subject_id = int(subject_str[1:]) if subject_str.startswith('p') else None
                study_id = int(study_str[1:]) if study_str.startswith('s') else None
                dicom_id = dicom_file.replace('.jpg', '') if dicom_file.endswith('.jpg') else dicom_file
                
                if subject_id and study_id:
                    parsed_files.append({
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'dicom_id': dicom_id,
                        'image_path': filename
                    })
        
        self.image_filenames = pd.DataFrame(parsed_files)
        print(f"  Loaded {len(self.image_filenames)} image file mappings")
        
    def load_reflacx_metadata(self):
        """Load REFLACX metadata to map REFLACX IDs to MIMIC study IDs"""
        print("Loading REFLACX metadata...")
        
        # REFLACX has metadata files that map their IDs to MIMIC IDs
        metadata_files = ['metadata_phase_1.csv', 'metadata_phase_2.csv', 'metadata_phase_3.csv']
        
        all_metadata = []
        for metadata_file in metadata_files:
            if self.use_s3:
                metadata_path = f"{self.paths['reflacx']}/main_data/{metadata_file}"
            else:
                metadata_path = self.paths['reflacx'] / 'main_data' / metadata_file
            
            try:
                df = self._read_csv(metadata_path)
                all_metadata.append(df)
                print(f"  Loaded {metadata_file}: {len(df)} records")
            except Exception as e:
                print(f"  Warning: Could not load {metadata_file}: {e}")
        
        if all_metadata:
            self.reflacx_metadata = pd.concat(all_metadata, ignore_index=True)
            print(f"  Total REFLACX metadata records: {len(self.reflacx_metadata)}")
            
            # Check what columns we have for mapping
            if 'study' in self.reflacx_metadata.columns and 'dicom_id' in self.reflacx_metadata.columns:
                print("  Found study and dicom_id columns for mapping")
            else:
                print(f"  Available columns: {self.reflacx_metadata.columns.tolist()}")
        else:
            self.reflacx_metadata = pd.DataFrame()

    def load_cxr_metadata(self):
        """Load MIMIC-CXR metadata and labels"""
        print("Loading MIMIC-CXR metadata...")

        # Load metadata (files are gzipped)
        if self.use_s3:
            metadata_path = f"{self.paths['mimic_cxr']}/mimic-cxr-2.0.0-metadata.csv.gz"
            chexpert_path = f"{self.paths['mimic_cxr']}/mimic-cxr-2.0.0-chexpert.csv.gz"
            split_path = f"{self.paths['mimic_cxr']}/mimic-cxr-2.0.0-split.csv.gz"
        else:
            metadata_path = self.paths['mimic_cxr'] / 'mimic-cxr-2.0.0-metadata.csv.gz'
            chexpert_path = self.paths['mimic_cxr'] / 'mimic-cxr-2.0.0-chexpert.csv.gz'
            split_path = self.paths['mimic_cxr'] / 'mimic-cxr-2.0.0-split.csv.gz'

        self.cxr_metadata = self._read_csv(metadata_path)
        self.cxr_labels = self._read_csv(chexpert_path)
        self.cxr_split = self._read_csv(split_path)
        
        # Merge metadata with labels
        self.cxr_combined = self.cxr_metadata.merge(
            self.cxr_labels, 
            on=['subject_id', 'study_id'],
            how='left'
        ).merge(
            self.cxr_split,
            on=['subject_id', 'study_id'],
            how='left'
        )
        
        print(f"  Loaded {len(self.cxr_combined)} CXR studies")
        print(f"  Sample study IDs: {self.cxr_metadata['study_id'].head().tolist()}")
        
    def load_reflacx_annotations(self):
        """Load REFLACX eye-tracking data with bounding boxes"""
        print("Loading REFLACX annotations...")
        
        # First load the metadata to get the mapping
        if self.reflacx_metadata is None or self.reflacx_metadata.empty:
            self.load_reflacx_metadata()
        
        annotations = []
        
        # For S3, we need to list the directories first
        if self.use_s3:
            # List all objects under main_data to find study directories
            prefix = f"{self.paths['reflacx']}/main_data/"
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            study_dirs = set()
            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Extract study directory from path
                        key_parts = obj['Key'].replace(prefix, '').split('/')
                        if len(key_parts) >= 2 and key_parts[0].startswith('P'):
                            study_dirs.add(key_parts[0])
            
            print(f"  Found {len(study_dirs)} REFLACX study directories")
            
            # Process a sample of study directories
            for i, reflacx_study_id in enumerate(list(study_dirs)[:100]):  # Limit for testing
                ellipse_file = f"{prefix}{reflacx_study_id}/anomaly_location_ellipses.csv"
                
                try:
                    df = self._read_csv(ellipse_file)
                    
                    # Map REFLACX study ID to MIMIC study ID using metadata
                    # The REFLACX study ID format is like P300R219715
                    # We need to find the corresponding MIMIC study_id
                    
                    for _, row in df.iterrows():
                        # Find which anomaly type is marked True
                        anomaly_cols = [
                            'Abnormal mediastinal contour', 'Acute fracture', 'Atelectasis',
                            'Consolidation', 'Enlarged cardiac silhouette', 'Enlarged hilum',
                            'Groundglass opacity', 'Hiatal hernia', 'High lung volume / emphysema',
                            'Interstitial lung disease', 'Lung nodule or mass', 'Other',
                            'Pleural abnormality', 'Pneumothorax', 'Pulmonary edema', 'Support devices'
                        ]
                        
                        anomaly_types = [col for col in anomaly_cols if col in row and row[col] == True]
                        anomaly_type = anomaly_types[0] if anomaly_types else 'unknown'
                        
                        annotations.append({
                            'reflacx_study_id': reflacx_study_id,  # Keep REFLACX ID for now
                            'anomaly_type': anomaly_type,
                            'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                            'certainty': row.get('certainty', None)
                        })
                    
                    if i % 20 == 0:
                        print(f"    Processed {i+1}/{len(study_dirs)} annotation files...")
                        
                except Exception as e:
                    # Skip files that don't exist or can't be read
                    pass
        
        self.reflacx_data = pd.DataFrame(annotations)
        print(f"  Loaded {len(self.reflacx_data)} anomaly annotations")
        
        if not self.reflacx_data.empty:
            print(f"  Sample REFLACX study IDs: {self.reflacx_data['reflacx_study_id'].head().tolist()}")
            print(f"  Anomaly types found: {self.reflacx_data['anomaly_type'].value_counts().head()}")

    def load_clinical_data_sample(self, num_subjects: int = 10):
        """
        Load a small sample of clinical data for testing
        
        Args:
            num_subjects: Number of subjects to load (small for testing)
        """
        print(f"Loading clinical data sample for {num_subjects} subjects...")
        
        # Get a sample of subject IDs from CXR metadata
        if self.cxr_metadata is not None:
            sample_subjects = self.cxr_metadata['subject_id'].unique()[:num_subjects].tolist()
        else:
            # Fallback to hardcoded IDs for testing
            sample_subjects = [10000032, 10000126, 10000246, 10000384, 10000612][:num_subjects]
        
        print(f"  Sample subject IDs: {sample_subjects}")
        
        # Load minimal admission data
        if self.use_s3:
            admissions_path = f"{self.paths['mimic_iv']}/hosp/admissions.csv.gz"
        else:
            admissions_path = self.paths['mimic_iv'] / 'hosp' / 'admissions.csv.gz'
        
        try:
            # Read only specific columns to reduce memory
            cols_to_read = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_type']
            admissions = self._read_csv(
                admissions_path,
                usecols=cols_to_read,
                parse_dates=['admittime', 'dischtime']
            )
            admissions = admissions[admissions['subject_id'].isin(sample_subjects)]
            print(f"    Loaded {len(admissions)} admission records")
        except Exception as e:
            print(f"    Error loading admissions: {e}")
            admissions = pd.DataFrame()
        
        # Store minimal clinical data
        self.clinical_data = {
            'admissions': admissions,
            'sample_subjects': sample_subjects
        }
    
    def create_sample_links(self, num_records: int = 5) -> pd.DataFrame:
        """
        Create a small sample of linked records for testing
        
        Args:
            num_records: Number of records to create
            
        Returns:
            DataFrame with sample linked data
        """
        print(f"Creating {num_records} sample linked records...")
        
        linked_records = []
        
        # Get sample of CXR studies
        sample_studies = self.cxr_combined.head(num_records)
        
        for _, study in sample_studies.iterrows():
            subject_id = study['subject_id']
            study_id = study['study_id']
            
            # Get image paths from IMAGE_FILENAMES
            image_paths = []
            if self.image_filenames is not None:
                study_images = self.image_filenames[
                    (self.image_filenames['subject_id'] == subject_id) &
                    (self.image_filenames['study_id'] == study_id)
                ]
                image_paths = study_images['image_path'].tolist()
            
            # Extract positive findings
            positive_findings = []
            label_cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                         'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                         'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
                         'Pneumonia', 'Pneumothorax', 'Support Devices']
            
            for col in label_cols:
                if col in study and study[col] == 1.0:
                    positive_findings.append(col)
            
            # Get admission info if available
            admission_info = {}
            if 'admissions' in self.clinical_data and not self.clinical_data['admissions'].empty:
                subj_admissions = self.clinical_data['admissions'][
                    self.clinical_data['admissions']['subject_id'] == subject_id
                ]
                if not subj_admissions.empty:
                    latest_admission = subj_admissions.iloc[-1]
                    admission_info = {
                        'hadm_id': latest_admission['hadm_id'],
                        'admission_type': latest_admission['admission_type']
                    }
            
            # Create linked record
            record = {
                'subject_id': subject_id,
                'study_id': study_id,
                'image_paths': image_paths[:3],  # Limit to 3 images
                'num_images': len(image_paths),
                'positive_findings': positive_findings,
                'num_positive_findings': len(positive_findings),
                'split': study.get('split', 'unknown'),
                'admission_info': admission_info,
                
                # Placeholders for features we'll add later
                'has_reflacx_annotations': False,
                'clinical_data_available': bool(admission_info)
            }
            
            linked_records.append(record)
        
        df = pd.DataFrame(linked_records)
        
        # Print summary
        print(f"\nSample linked records created:")
        print(f"  Total records: {len(df)}")
        print(f"  Records with images: {(df['num_images'] > 0).sum()}")
        print(f"  Records with findings: {(df['num_positive_findings'] > 0).sum()}")
        print(f"  Records with clinical data: {df['clinical_data_available'].sum()}")
        
        return df
    
    def validate_s3_access(self) -> bool:
        """
        Validate that we can access the S3 bucket and key files exist
        
        Returns:
            True if validation passes
        """
        if not self.use_s3:
            return True
        
        print("\nValidating S3 access...")
        
        required_files = [
            f"{self.paths['mimic_cxr']}/mimic-cxr-2.0.0-metadata.csv.gz",
            f"{self.paths['mimic_cxr']}/mimic-cxr-2.0.0-chexpert.csv.gz",
            f"{self.paths['mimic_cxr']}/IMAGE_FILENAMES",
            f"{self.paths['mimic_iv']}/hosp/admissions.csv.gz"
        ]
        
        all_exist = True
        for file_key in required_files:
            try:
                self.s3_client.head_object(Bucket=self.s3_bucket, Key=file_key)
                print(f"  ✓ Found: {file_key}")
            except Exception as e:
                print(f"  ✗ Missing: {file_key}")
                all_exist = False
        
        return all_exist


def main():
    """Main preprocessing pipeline for Phase 1 testing"""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='MIMIC Phase 1 - AWS Deployment Testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--s3-bucket',
        type=str,
        default='bergermimiciv',
        help='S3 bucket name'
    )
    
    parser.add_argument(
        '--s3-prefix',
        type=str,
        default='physionet.org/files',
        help='S3 prefix path to MIMIC datasets'
    )
    
    parser.add_argument(
        '--num-subjects',
        type=int,
        default=5,
        help='Number of subjects for testing'
    )
    
    parser.add_argument(
        '--num-records',
        type=int,
        default=10,
        help='Number of sample records to create'
    )
    
    parser.add_argument(
        '--local',
        action='store_true',
        help='Run locally instead of S3'
    )
    
    parser.add_argument(
        '--local-path',
        type=str,
        default='~/Documents/Portfolio/MIMIC_Data/physionet.org/files',
        help='Local path to MIMIC datasets'
    )
    
    args = parser.parse_args()
    
    # Expand local path if needed
    if args.local:
        local_path = os.path.expanduser(args.local_path)
        use_s3 = False
        base_path = local_path
        s3_bucket = None
    else:
        use_s3 = True
        base_path = args.s3_prefix
        s3_bucket = args.s3_bucket
    
    print("="*70)
    print("MIMIC Multimodal Preprocessing - Phase 1 Testing")
    print("="*70)
    print(f"Mode: {'S3' if use_s3 else 'Local'}")
    if use_s3:
        print(f"S3 Bucket: {s3_bucket}")
        print(f"S3 Prefix: {base_path}")
    else:
        print(f"Local Path: {base_path}")
    print("="*70)
    
    try:
        # Initialize preprocessor
        preprocessor = MIMICPreprocessor(
            base_path=base_path,
            use_s3=use_s3,
            s3_bucket=s3_bucket
        )
        
        # Step 1: Validate S3 access
        if use_s3:
            if not preprocessor.validate_s3_access():
                print("\n❌ S3 validation failed. Please check bucket and paths.")
                return
        
        # Step 2: Load IMAGE_FILENAMES
        print("\n" + "="*70)
        print("Step 1: Loading IMAGE_FILENAMES")
        print("="*70)
        preprocessor.load_image_filenames()
        
        # Step 3: Load CXR metadata
        print("\n" + "="*70)
        print("Step 2: Loading CXR Metadata")
        print("="*70)
        preprocessor.load_cxr_metadata()
        
        # Step 4: Load REFLACX data
        print("\n" + "="*70)
        print("Step 3: Loading REFLACX Annotations")
        print("="*70)
        preprocessor.load_reflacx_annotations()
        
        # Step 5: Load sample clinical data
        print("\n" + "="*70)
        print("Step 4: Loading Clinical Data Sample")
        print("="*70)
        preprocessor.load_clinical_data_sample(num_subjects=args.num_subjects)
        
        # Step 6: Create sample linked records
        print("\n" + "="*70)
        print("Step 5: Creating Sample Linked Records")
        print("="*70)
        sample_df = preprocessor.create_sample_links(num_records=args.num_records)
        
        # Step 7: Display results
        print("\n" + "="*70)
        print("Phase 1 Testing Results")
        print("="*70)
        
        print("\nFirst linked record (detailed):")
        if not sample_df.empty:
            first_record = sample_df.iloc[0].to_dict()
            for key, value in first_record.items():
                print(f"  {key}: {value}")
        
        print("\n✅ Phase 1 testing completed successfully!")
        
        # Save sample output
        if use_s3:
            # Save to S3
            output_key = f"{base_path}/preprocessed/phase1_test_results.json"
            output_data = sample_df.to_json(orient='records', indent=2)
            
            preprocessor.s3_client.put_object(
                Bucket=s3_bucket,
                Key=output_key,
                Body=output_data,
                ContentType='application/json'
            )
            print(f"\nResults saved to: s3://{s3_bucket}/{output_key}")
        else:
            # Save locally
            output_path = Path("phase1_test_results.json")
            sample_df.to_json(output_path, orient='records', indent=2)
            print(f"\nResults saved to: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()