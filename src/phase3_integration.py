"""
Phase 3: Integration
Creates patient-level folder structure and integrates all modalities
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
from tqdm import tqdm

from .config_manager import get_config
from .utils import S3Handler


class DataIntegrator:
    """Integrate all modalities into patient-level folders"""
    
    def __init__(self):
        """Initialize Data Integrator"""
        self.config = get_config()
        self.s3 = S3Handler(
            region=self.config.get('aws.region'),
            profile=self.config.get('aws.profile')
        )
        self.copy_images = self.config.get(
            'preprocessing.phase3.copy_images', True
        )
        
        logger.info("Data Integrator initialized")
    
    def create_patient_folder_structure(
        self,
        bucket: str,
        subject_id: int
    ):
        """
        Create folder structure for a patient in S3
        
        Args:
            bucket: S3 bucket name
            subject_id: Patient ID
        """
        prefix = f"processed/patient_{subject_id}/"
        
        # Create placeholder files to establish folder structure
        folders = [
            'ED/',
            'Hosp/',
            'CXR-JPG/',
            'metadata/'
        ]
        
        for folder in folders:
            key = f"{prefix}{folder}.placeholder"
            try:
                self.s3.s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=b''
                )
            except Exception as e:
                logger.warning(f"Could not create folder {folder}: {str(e)}")
    
    def save_patient_metadata(
        self,
        bucket: str,
        subject_id: int,
        patient_records: List[Dict]
    ):
        """
        Save patient metadata as JSON
        
        Args:
            bucket: S3 bucket name
            subject_id: Patient ID
            patient_records: List of patient records
        """
        prefix = f"processed/patient_{subject_id}/"
        key = f"{prefix}metadata/patient_data.json"
        
        try:
            # Convert to JSON-serializable format
            serializable_records = []
            for record in patient_records:
                serializable_record = {}
                for k, v in record.items():
                    # Handle pandas/numpy types
                    if pd.isna(v):
                        serializable_record[k] = None
                    elif isinstance(v, (pd.Timestamp, pd.DatetimeTZDtype)):
                        serializable_record[k] = str(v)
                    elif hasattr(v, 'item'):  # numpy types
                        serializable_record[k] = v.item()
                    else:
                        serializable_record[k] = v
                serializable_records.append(serializable_record)
            
            # Convert to JSON
            json_data = json.dumps(serializable_records, indent=2, default=str)
            
            # Upload to S3
            self.s3.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_data.encode('utf-8')
            )
            
            logger.debug(f"Saved metadata for patient {subject_id}")
            
        except Exception as e:
            logger.error(f"Error saving metadata for patient {subject_id}: {str(e)}")
            raise
    
    def copy_cxr_images(
        self,
        subject_id: int,
        patient_records: List[Dict],
        source_bucket: str,
        dest_bucket: str
    ):
        """
        Copy CXR images to patient folder
        
        Args:
            subject_id: Patient ID
            patient_records: List of patient records
            source_bucket: Source S3 bucket (MIMIC)
            dest_bucket: Destination S3 bucket
        """
        images_copied = 0
        images_failed = 0
        
        for record in patient_records:
            dicom_id = None  # Initialize to avoid unbound variable
            try:
                study_id = record['study_id']
                dicom_id = record['dicom_id']

                # Construct source path
                # MIMIC-CXR-JPG structure: files/p10/p10000032/s50414267/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.jpg
                p_dir = f"p{str(subject_id)[:2]}"
                source_key = (
                    f"mimic-cxr-jpg/2.0.0/files/"
                    f"{p_dir}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
                )

                # Construct destination path
                dest_key = (
                    f"processed/patient_{subject_id}/"
                    f"CXR-JPG/s{study_id}_{dicom_id}.jpg"
                )

                # Check if source exists
                if not self.s3.object_exists(source_bucket, source_key):
                    logger.warning(f"Source image not found: {source_key}")
                    images_failed += 1
                    continue

                # Copy image
                self.s3.copy_object(
                    source_bucket,
                    source_key,
                    dest_bucket,
                    dest_key
                )

                images_copied += 1

            except Exception as e:
                logger.error(f"Error copying image {dicom_id if dicom_id else 'unknown'}: {str(e)}")
                images_failed += 1
                continue
        
        logger.info(f"Patient {subject_id}: Copied {images_copied} images, {images_failed} failed")
    
    def save_patient_csv_data(
        self,
        bucket: str,
        subject_id: int,
        patient_records_df: pd.DataFrame
    ):
        """
        Save patient data as CSV for easier access
        
        Args:
            bucket: S3 bucket name
            subject_id: Patient ID
            patient_records_df: DataFrame with patient records
        """
        prefix = f"processed/patient_{subject_id}/"
        
        try:
            # Save main records CSV
            records_key = f"{prefix}metadata/records.csv"
            self.s3.write_csv(patient_records_df, bucket, records_key)
            
            # Extract and save clinical features if available
            if 'clinical_features' in patient_records_df.columns:
                clinical_data = []
                for idx, row in patient_records_df.iterrows():
                    if row['clinical_features'] and isinstance(row['clinical_features'], dict):
                        clinical_data.append(row['clinical_features'])
                
                if clinical_data:
                    clinical_df = pd.DataFrame(clinical_data)
                    clinical_key = f"{prefix}metadata/clinical_features.csv"
                    self.s3.write_csv(clinical_df, bucket, clinical_key)
                    logger.debug(f"Saved clinical features for patient {subject_id}")
            
        except Exception as e:
            logger.error(f"Error saving CSV data for patient {subject_id}: {str(e)}")
    
    def process_patient(
        self,
        subject_id: int,
        patient_records: List[Dict]
    ):
        """
        Process and integrate data for a single patient
        
        Args:
            subject_id: Patient ID
            patient_records: List of records for this patient
        """
        output_bucket = self.config.get('aws.s3.output_bucket')
        mimic_bucket = self.config.get('aws.s3.mimic_bucket')
        
        logger.debug(f"Processing patient {subject_id} with {len(patient_records)} records")
        
        try:
            # Create folder structure
            self.create_patient_folder_structure(output_bucket, subject_id)
            
            # Save metadata as JSON
            self.save_patient_metadata(output_bucket, subject_id, patient_records)
            
            # Save as CSV
            patient_df = pd.DataFrame(patient_records)
            self.save_patient_csv_data(output_bucket, subject_id, patient_df)
            
            # Copy images if enabled
            if self.copy_images:
                self.copy_cxr_images(
                    subject_id,
                    patient_records,
                    mimic_bucket,
                    output_bucket
                )
            
            logger.debug(f"Successfully processed patient {subject_id}")
            
        except Exception as e:
            logger.error(f"Error processing patient {subject_id}: {str(e)}")
            raise
    
    def create_master_index(
        self,
        all_records: pd.DataFrame
    ):
        """
        Create master index of all patients and records
        
        Args:
            all_records: DataFrame with all patient records
        """
        output_bucket = self.config.get('aws.s3.output_bucket')
        
        try:
            # Create summary statistics
            summary = {
                'total_patients': int(all_records['subject_id'].nunique()),
                'total_records': int(len(all_records)),
                'total_stays': int(all_records['stay_id'].nunique()),
                'processing_date': pd.Timestamp.now().isoformat(),
                'patients_with_clinical_data': int(
                    all_records['clinical_features'].notna().sum()
                ),
                'average_records_per_patient': float(
                    len(all_records) / all_records['subject_id'].nunique()
                )
            }
            
            # Save summary
            summary_key = 'processed/summary.json'
            self.s3.s3_client.put_object(
                Bucket=output_bucket,
                Key=summary_key,
                Body=json.dumps(summary, indent=2).encode('utf-8')
            )
            
            logger.info(f"Created summary with {summary['total_patients']} patients")
            
            # Save master index
            # Remove complex nested columns for CSV
            index_df = all_records.copy()
            
            # Flatten clinical features
            if 'clinical_features' in index_df.columns:
                index_df['has_clinical_data'] = index_df['clinical_features'].notna()
                index_df = index_df.drop(columns=['clinical_features'])
            
            # Flatten other complex columns
            for col in ['lab_summary', 'medication_summary']:
                if col in index_df.columns:
                    index_df = index_df.drop(columns=[col])
            
            index_key = 'processed/master_index.csv'
            self.s3.write_csv(index_df, output_bucket, index_key)
            
            logger.info(f"Created master index with {len(all_records)} records")
            
        except Exception as e:
            logger.error(f"Error creating master index: {str(e)}")
            raise
    
    def create_patient_list(
        self,
        all_records: pd.DataFrame
    ):
        """
        Create a simple list of all patients with counts
        
        Args:
            all_records: DataFrame with all patient records
        """
        output_bucket = self.config.get('aws.s3.output_bucket')
        
        try:
            # Group by patient
            patient_summary = all_records.groupby('subject_id').agg({
                'dicom_id': 'count',
                'stay_id': 'nunique',
                'study_datetime': ['min', 'max']
            }).reset_index()
            
            patient_summary.columns = [
                'subject_id',
                'total_cxrs',
                'total_stays',
                'first_cxr',
                'last_cxr'
            ]
            
            # Save patient list
            patient_list_key = 'processed/patient_list.csv'
            self.s3.write_csv(patient_summary, output_bucket, patient_list_key)
            
            logger.info(f"Created patient list with {len(patient_summary)} patients")
            
        except Exception as e:
            logger.error(f"Error creating patient list: {str(e)}")
    
    def run(self):
        """Execute Phase 3: Integration"""
        logger.info("="*60)
        logger.info("Starting Phase 3: Data Integration")
        logger.info("="*60)
        
        # Load Phase 2 results
        output_bucket = self.config.get('aws.s3.output_bucket')
        input_key = 'processing/patient_clinical_data.csv'
        
        logger.info(f"Loading Phase 2 results from s3://{output_bucket}/{input_key}")
        
        try:
            patient_data = self.s3.read_csv(output_bucket, input_key)
        except Exception as e:
            logger.error(f"Failed to load Phase 2 results: {str(e)}")
            raise
        
        if len(patient_data) == 0:
            logger.error("No data found in Phase 2 results")
            raise ValueError("Phase 2 results are empty")
        
        # Group by patient
        unique_patients = patient_data['subject_id'].nunique()
        logger.info(f"Processing {unique_patients} patients with {len(patient_data)} total records")
        
        # Process each patient
        processed_count = 0
        failed_count = 0
        
        for subject_id, records_df in tqdm(
            patient_data.groupby('subject_id'),
            desc="Integrating patient data",
            total=unique_patients
        ):
            try:
                # Convert to list of dicts
                patient_records = records_df.to_dict('records')

                # Process patient - ensure subject_id is int
                self.process_patient(int(subject_id), patient_records)
                processed_count += 1

            except Exception as e:
                logger.error(f"Failed to process patient {subject_id}: {str(e)}")
                failed_count += 1
                continue
        
        logger.info(f"Processed {processed_count} patients, {failed_count} failed")
        
        # Create master index
        logger.info("Creating master index...")
        self.create_master_index(patient_data)
        
        # Create patient list
        logger.info("Creating patient list...")
        self.create_patient_list(patient_data)
        
        # Log statistics
        self._log_statistics(patient_data, processed_count, failed_count)
        
        logger.info("Phase 3 complete!")
    
    def _log_statistics(self, data: pd.DataFrame, processed: int, failed: int):
        """Log integration statistics"""
        total_patients = data['subject_id'].nunique()
        total_records = len(data)
        total_stays = data['stay_id'].nunique()
        
        logger.info("="*60)
        logger.info("Phase 3 Statistics:")
        logger.info(f"  Total patients:        {total_patients:,}")
        logger.info(f"  Processed successfully: {processed:,}")
        logger.info(f"  Failed:                {failed:,}")
        logger.info(f"  Total records:         {total_records:,}")
        logger.info(f"  Total ED stays:        {total_stays:,}")
        logger.info(f"  Avg records/patient:   {total_records/total_patients:.2f}")
        
        if 'clinical_features' in data.columns:
            with_clinical = data['clinical_features'].notna().sum()
            logger.info(f"  Records with clinical: {with_clinical:,} ({with_clinical/total_records*100:.1f}%)")
        
        logger.info("="*60)


def main():
    """Main entry point for Phase 3"""
    import argparse
    from .utils import setup_logging
    
    parser = argparse.ArgumentParser(
        description='Phase 3: Integrate data into patient folders'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run - do not copy images'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Run Phase 3
    integrator = DataIntegrator()
    
    # Override copy_images if dry run
    if args.dry_run:
        logger.info("DRY RUN: Images will not be copied")
        integrator.copy_images = False
    
    integrator.run()


if __name__ == '__main__':
    main()