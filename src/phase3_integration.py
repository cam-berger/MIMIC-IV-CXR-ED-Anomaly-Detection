"""
Phase 3: Integration
Creates patient-level folder structure and integrates all modalities with comprehensive clinical data
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
from tqdm import tqdm
from datetime import datetime

from .config_manager import get_config
from .utils import S3Handler


class DataIntegrator:
    """Integrate all modalities and clinical data into patient-level folders"""
    
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
        self.save_json = self.config.get(
            'preprocessing.phase3.save_json', True
        )
        self.save_csv = self.config.get(
            'preprocessing.phase3.save_csv', True
        )
        
        logger.info("Data Integrator initialized for comprehensive data")
    
    def create_patient_folder_structure(
        self,
        bucket: str,
        subject_id: int
    ):
        """
        Create comprehensive folder structure for a patient in S3
        
        Args:
            bucket: S3 bucket name
            subject_id: Patient ID
        """
        prefix = f"processed/patient_{subject_id}/"
        
        # Create comprehensive folder structure
        folders = [
            'ED/',                    # ED visit data
            'Hosp/',                  # Hospital admission data
            'Labs/',                  # Lab results
            'Medications/',           # Medication data
            'Diagnoses/',            # Diagnosis codes
            'Procedures/',           # Procedure codes
            'Vitals/',               # Vital signs
            'CXR-JPG/',              # Chest X-ray images
            'Reports/',              # Radiology reports
            'metadata/',             # Integrated metadata
            'timeline/'              # Time-based view of data
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
        Save comprehensive patient metadata as JSON
        
        Args:
            bucket: S3 bucket name
            subject_id: Patient ID
            patient_records: List of patient records with clinical data
        """
        prefix = f"processed/patient_{subject_id}/"
        
        try:
            # Save complete records with all nested data
            if self.save_json:
                key = f"{prefix}metadata/patient_data_complete.json"
                
                # Convert to JSON-serializable format
                serializable_records = self._make_serializable(patient_records)
                
                # Convert to JSON with nice formatting
                json_data = json.dumps(
                    serializable_records, 
                    indent=2, 
                    default=str,
                    sort_keys=True
                )
                
                # Upload to S3
                self.s3.s3_client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=json_data.encode('utf-8')
                )
                
                logger.debug(f"Saved complete metadata for patient {subject_id}")
            
            # Also save individual data components
            self._save_data_components(bucket, subject_id, patient_records)
            
        except Exception as e:
            logger.error(f"Error saving metadata for patient {subject_id}: {str(e)}")
            raise
    
    def _make_serializable(self, obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeTZDtype, datetime)):
            return str(obj)
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj
    
    def _save_data_components(
        self,
        bucket: str,
        subject_id: int,
        patient_records: List[Dict]
    ):
        """Save individual data components in separate files"""
        prefix = f"processed/patient_{subject_id}/"
        
        try:
            # Extract and save lab results
            lab_data = []
            for record in patient_records:
                if 'lab_results' in record and record['lab_results']:
                    lab_entry = {
                        'study_datetime': record['study_datetime'],
                        'stay_id': record['stay_id'],
                        **record['lab_results']
                    }
                    lab_data.append(lab_entry)
            
            if lab_data:
                # Save important labs as CSV
                important_labs = []
                for entry in lab_data:
                    if 'important_labs' in entry:
                        for lab_name, lab_info in entry['important_labs'].items():
                            important_labs.append({
                                'study_datetime': entry['study_datetime'],
                                'lab_name': lab_name,
                                'value': lab_info.get('value'),
                                'valuenum': lab_info.get('valuenum'),
                                'charttime': lab_info.get('charttime'),
                                'flag': lab_info.get('flag')
                            })
                
                if important_labs:
                    labs_df = pd.DataFrame(important_labs)
                    labs_key = f"{prefix}Labs/important_labs.csv"
                    self.s3.write_csv(labs_df, bucket, labs_key)
            
            # Extract and save medications
            med_data = []
            for record in patient_records:
                if 'medications' in record and record['medications']:
                    if record['medications'].get('medications'):
                        for med in record['medications']['medications']:
                            med_entry = {
                                'study_datetime': record['study_datetime'],
                                'stay_id': record['stay_id'],
                                **med
                            }
                            med_data.append(med_entry)
            
            if med_data:
                meds_df = pd.DataFrame(med_data)
                meds_key = f"{prefix}Medications/medications.csv"
                self.s3.write_csv(meds_df, bucket, meds_key)
            
            # Extract and save diagnoses
            diag_data = []
            for record in patient_records:
                if 'diagnoses' in record and record['diagnoses']:
                    if record['diagnoses'].get('all_diagnoses'):
                        for diag in record['diagnoses']['all_diagnoses']:
                            diag_entry = {
                                'study_datetime': record['study_datetime'],
                                'stay_id': record['stay_id'],
                                **diag
                            }
                            diag_data.append(diag_entry)
            
            if diag_data:
                diag_df = pd.DataFrame(diag_data)
                diag_key = f"{prefix}Diagnoses/diagnoses.csv"
                self.s3.write_csv(diag_df, bucket, diag_key)
            
            # Extract and save vital signs
            vitals_data = []
            for record in patient_records:
                if 'vital_signs' in record and record['vital_signs']:
                    vitals_entry = {
                        'study_datetime': record['study_datetime'],
                        'stay_id': record['stay_id'],
                        **{k: v for k, v in record['vital_signs'].items() 
                           if k not in ['note', 'error']}
                    }
                    vitals_data.append(vitals_entry)
            
            if vitals_data:
                vitals_df = pd.DataFrame(vitals_data)
                vitals_key = f"{prefix}Vitals/vital_signs.csv"
                self.s3.write_csv(vitals_df, bucket, vitals_key)
            
        except Exception as e:
            logger.error(f"Error saving data components for patient {subject_id}: {str(e)}")
    
    def create_patient_timeline(
        self,
        bucket: str,
        subject_id: int,
        patient_records: List[Dict]
    ):
        """
        Create a timeline view of patient data
        
        Args:
            bucket: S3 bucket name
            subject_id: Patient ID
            patient_records: List of patient records
        """
        prefix = f"processed/patient_{subject_id}/"
        
        try:
            timeline = []
            
            for record in patient_records:
                study_dt = pd.to_datetime(record['study_datetime'])
                
                # Add CXR event
                timeline.append({
                    'datetime': str(study_dt),
                    'event_type': 'CXR',
                    'description': f"Chest X-ray ({record.get('ViewPosition', 'Unknown view')})",
                    'stay_id': record.get('stay_id'),
                    'hadm_id': record.get('hadm_id')
                })
                
                # Add lab events
                if record.get('lab_results', {}).get('most_recent_charttime'):
                    timeline.append({
                        'datetime': record['lab_results']['most_recent_charttime'],
                        'event_type': 'Labs',
                        'description': f"{record['lab_results'].get('lab_count', 0)} lab results",
                        'stay_id': record.get('stay_id'),
                        'abnormal_count': record['lab_results'].get('abnormal_count', 0)
                    })
                
                # Add medication events
                if record.get('medications', {}).get('medications'):
                    timeline.append({
                        'datetime': str(study_dt),
                        'event_type': 'Medications',
                        'description': f"{record['medications'].get('medication_count', 0)} medications active",
                        'stay_id': record.get('stay_id'),
                        'has_antibiotics': record['medications'].get('has_antibiotics', False)
                    })
            
            # Sort timeline
            timeline_df = pd.DataFrame(timeline)
            if not timeline_df.empty:
                timeline_df['datetime'] = pd.to_datetime(timeline_df['datetime'])
                timeline_df = timeline_df.sort_values('datetime')
                
                # Save timeline
                timeline_key = f"{prefix}timeline/patient_timeline.csv"
                self.s3.write_csv(timeline_df, bucket, timeline_key)
                
                logger.debug(f"Created timeline for patient {subject_id}")
            
        except Exception as e:
            logger.error(f"Error creating timeline for patient {subject_id}: {str(e)}")
    
    def save_patient_summary(
        self,
        bucket: str,
        subject_id: int,
        patient_records: List[Dict]
    ):
        """
        Create and save a patient summary
        
        Args:
            bucket: S3 bucket name
            subject_id: Patient ID
            patient_records: List of patient records
        """
        prefix = f"processed/patient_{subject_id}/"
        
        try:
            # Calculate summary statistics
            summary = {
                'subject_id': int(subject_id),
                'total_cxrs': len(patient_records),
                'total_stays': len(set(r['stay_id'] for r in patient_records if r.get('stay_id'))),
                'total_admissions': len(set(r['hadm_id'] for r in patient_records if r.get('hadm_id'))),
                'date_range': {
                    'first_cxr': min(r['study_datetime'] for r in patient_records),
                    'last_cxr': max(r['study_datetime'] for r in patient_records)
                },
                'data_availability': {
                    'has_clinical': any(r.get('clinical_features') for r in patient_records),
                    'has_labs': any(r.get('lab_results', {}).get('labs_available') for r in patient_records),
                    'has_medications': any(r.get('medications', {}).get('medications_available') for r in patient_records),
                    'has_diagnoses': any(r.get('diagnoses', {}).get('diagnoses_available') for r in patient_records),
                    'has_procedures': any(r.get('procedures', {}).get('procedures_available') for r in patient_records)
                },
                'statistics': {
                    'total_lab_results': sum(r.get('lab_results', {}).get('lab_count', 0) for r in patient_records),
                    'total_medications': sum(r.get('medications', {}).get('unique_medications', 0) for r in patient_records),
                    'total_diagnoses': sum(r.get('diagnoses', {}).get('diagnosis_count', 0) for r in patient_records),
                    'total_procedures': sum(r.get('procedures', {}).get('procedure_count', 0) for r in patient_records)
                }
            }
            
            # Add demographics if available
            for record in patient_records:
                if record.get('clinical_features'):
                    cf = record['clinical_features']
                    summary['demographics'] = {
                        'age': cf.get('age'),
                        'gender': cf.get('gender')
                    }
                    break
            
            # Save summary
            summary_key = f"{prefix}metadata/patient_summary.json"
            self.s3.s3_client.put_object(
                Bucket=bucket,
                Key=summary_key,
                Body=json.dumps(summary, indent=2, default=str).encode('utf-8')
            )
            
            logger.debug(f"Saved summary for patient {subject_id}")
            
        except Exception as e:
            logger.error(f"Error saving summary for patient {subject_id}: {str(e)}")
    
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
            dicom_id = None
            try:
                study_id = record['study_id']
                dicom_id = record['dicom_id']
                
                # Construct source path
                p_dir = f"p{str(subject_id)[:2]}"
                source_key = (
                    f"mimic-cxr-jpg/2.0.0/files/"
                    f"{p_dir}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
                )
                
                # Construct destination path with metadata
                view = record.get('ViewPosition', 'unknown').replace(' ', '_')
                dest_key = (
                    f"processed/patient_{subject_id}/"
                    f"CXR-JPG/s{study_id}_{dicom_id}_{view}.jpg"
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
    
    def process_patient(
        self,
        subject_id: int,
        patient_records: List[Dict]
    ):
        """
        Process and integrate comprehensive data for a single patient
        
        Args:
            subject_id: Patient ID
            patient_records: List of records for this patient
        """
        output_bucket = self.config.get('aws.s3.output_bucket')
        mimic_bucket = self.config.get('aws.s3.mimic_bucket')
        
        logger.debug(f"Processing patient {subject_id} with {len(patient_records)} records")
        
        try:
            # Create comprehensive folder structure
            self.create_patient_folder_structure(output_bucket, subject_id)
            
            # Save complete metadata
            self.save_patient_metadata(output_bucket, subject_id, patient_records)
            
            # Create patient summary
            self.save_patient_summary(output_bucket, subject_id, patient_records)
            
            # Create timeline view
            self.create_patient_timeline(output_bucket, subject_id, patient_records)
            
            # Save as CSV if requested
            if self.save_csv:
                patient_df = pd.DataFrame(patient_records)
                flattened_df = self._flatten_for_csv(patient_df)
                csv_key = f"processed/patient_{subject_id}/metadata/patient_data_flat.csv"
                self.s3.write_csv(flattened_df, output_bucket, csv_key)
            
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
        all_records: List[Dict]
    ):
        """
        Create comprehensive master index of all patients and records
        
        Args:
            all_records: List of all patient records
        """
        output_bucket = self.config.get('aws.s3.output_bucket')
        
        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(all_records)
            
            # Create comprehensive summary statistics
            summary = {
                'processing_info': {
                    'processing_date': pd.Timestamp.now().isoformat(),
                    'total_records': len(all_records),
                    'total_patients': df['subject_id'].nunique(),
                    'total_stays': df['stay_id'].nunique(),
                    'total_admissions': df['hadm_id'].nunique() if 'hadm_id' in df else 0
                },
                'data_completeness': {
                    'records_with_clinical': sum(1 for r in all_records if r.get('clinical_features')),
                    'records_with_labs': sum(1 for r in all_records if r.get('lab_results', {}).get('labs_available')),
                    'records_with_medications': sum(1 for r in all_records if r.get('medications', {}).get('medications_available')),
                    'records_with_diagnoses': sum(1 for r in all_records if r.get('diagnoses', {}).get('diagnoses_available')),
                    'records_with_procedures': sum(1 for r in all_records if r.get('procedures', {}).get('procedures_available'))
                },
                'aggregate_statistics': {
                    'total_lab_results': sum(r.get('lab_results', {}).get('lab_count', 0) for r in all_records),
                    'total_unique_labs': len(set(
                        lab_name 
                        for r in all_records 
                        if r.get('lab_results', {}).get('important_labs')
                        for lab_name in r['lab_results']['important_labs'].keys()
                    )),
                    'total_medications': sum(r.get('medications', {}).get('medication_count', 0) for r in all_records),
                    'total_diagnoses': sum(r.get('diagnoses', {}).get('diagnosis_count', 0) for r in all_records),
                    'patients_with_antibiotics': sum(
                        1 for _, patient_df in df.groupby('subject_id')
                        if any(r.get('medications', {}).get('has_antibiotics') for _, r in patient_df.iterrows())
                    )
                },
                'temporal_range': {
                    'earliest_study': df['study_datetime'].min(),
                    'latest_study': df['study_datetime'].max()
                }
            }
            
            # Save comprehensive summary
            summary_key = 'processed/master_summary.json'
            self.s3.s3_client.put_object(
                Bucket=output_bucket,
                Key=summary_key,
                Body=json.dumps(summary, indent=2, default=str).encode('utf-8')
            )
            
            logger.info(f"Created master summary with {summary['processing_info']['total_patients']} patients")
            
            # Save flattened master index for querying
            index_df = self._flatten_for_csv(df)
            index_key = 'processed/master_index.csv'
            self.s3.write_csv(index_df, output_bucket, index_key)
            
            logger.info(f"Created master index with {len(all_records)} records")
            
        except Exception as e:
            logger.error(f"Error creating master index: {str(e)}")
            raise
    
    def _flatten_for_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested dictionaries for CSV export"""
        flat_df = df[['subject_id', 'stay_id', 'dicom_id', 
                     'study_id', 'study_datetime', 'ViewPosition']].copy()
        
        # Add hadm_id if present
        if 'hadm_id' in df.columns:
            flat_df['hadm_id'] = df['hadm_id']
        
        # Flatten clinical features
        for _, row in df.iterrows():
            if row.get('clinical_features') and isinstance(row['clinical_features'], dict):
                for key, value in row['clinical_features'].items():
                    if key not in ['subject_id', 'stay_id']:  # Avoid duplicates
                        flat_df.loc[_, f'clinical_{key}'] = value
        
        # Add summary metrics from nested structures
        for _, row in df.iterrows():
            # Lab metrics
            if row.get('lab_results') and isinstance(row['lab_results'], dict):
                flat_df.loc[_, 'lab_count'] = row['lab_results'].get('lab_count', 0)
                flat_df.loc[_, 'unique_labs'] = row['lab_results'].get('unique_labs', 0)
                flat_df.loc[_, 'abnormal_labs'] = row['lab_results'].get('abnormal_count', 0)
            
            # Medication metrics
            if row.get('medications') and isinstance(row['medications'], dict):
                flat_df.loc[_, 'medication_count'] = row['medications'].get('medication_count', 0)
                flat_df.loc[_, 'unique_medications'] = row['medications'].get('unique_medications', 0)
                flat_df.loc[_, 'has_antibiotics'] = row['medications'].get('has_antibiotics', False)
            
            # Diagnosis metrics
            if row.get('diagnoses') and isinstance(row['diagnoses'], dict):
                flat_df.loc[_, 'diagnosis_count'] = row['diagnoses'].get('diagnosis_count', 0)
                flat_df.loc[_, 'unique_diagnoses'] = row['diagnoses'].get('unique_diagnoses', 0)
            
            # Procedure metrics
            if row.get('procedures') and isinstance(row['procedures'], dict):
                flat_df.loc[_, 'procedure_count'] = row['procedures'].get('procedure_count', 0)
                flat_df.loc[_, 'unique_procedures'] = row['procedures'].get('unique_procedures', 0)
        
        return flat_df
    
    def run(self):
        """Execute Phase 3: Comprehensive Integration"""
        logger.info("="*60)
        logger.info("Starting Phase 3: Comprehensive Data Integration")
        logger.info("="*60)
        
        # Load Phase 2 results (JSON format for nested data)
        output_bucket = self.config.get('aws.s3.output_bucket')
        input_key = 'processing/patient_clinical_data_comprehensive.json'
        
        logger.info(f"Loading Phase 2 comprehensive results from s3://{output_bucket}/{input_key}")
        
        try:
            # Load JSON data
            response = self.s3.s3_client.get_object(Bucket=output_bucket, Key=input_key)
            all_records = json.loads(response['Body'].read())
            
            if not all_records:
                logger.error("No data found in Phase 2 results")
                raise ValueError("Phase 2 results are empty")
            
            logger.info(f"Loaded {len(all_records)} records")
            
            # Group by patient
            patient_groups = {}
            for record in all_records:
                subject_id = record['subject_id']
                if subject_id not in patient_groups:
                    patient_groups[subject_id] = []
                patient_groups[subject_id].append(record)
            
            unique_patients = len(patient_groups)
            logger.info(f"Processing {unique_patients} patients with {len(all_records)} total records")
            
            # Process each patient
            processed_count = 0
            failed_count = 0
            
            for subject_id, patient_records in tqdm(
                patient_groups.items(),
                desc="Integrating comprehensive patient data",
                total=unique_patients
            ):
                try:
                    self.process_patient(int(subject_id), patient_records)
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process patient {subject_id}: {str(e)}")
                    failed_count += 1
                    continue
            
            logger.info(f"Processed {processed_count} patients, {failed_count} failed")
            
            # Create master index
            logger.info("Creating comprehensive master index...")
            self.create_master_index(all_records)
            
            # Log comprehensive statistics
            self._log_statistics(all_records, processed_count, failed_count)
            
            logger.info("Phase 3 complete!")
            
        except Exception as e:
            logger.error(f"Failed to load Phase 2 results: {str(e)}")
            logger.info("Attempting to load CSV fallback...")
            
            # Fallback to CSV if JSON not available
            try:
                csv_key = 'processing/patient_clinical_data_flat.csv'
                patient_data = self.s3.read_csv(output_bucket, csv_key)
                
                # Process with limited data from CSV
                self._process_from_csv(patient_data)
                
            except Exception as csv_error:
                logger.error(f"Failed to load CSV fallback: {str(csv_error)}")
                raise
    
    def _process_from_csv(self, patient_data: pd.DataFrame):
        """Fallback processing from CSV data"""
        logger.warning("Using CSV fallback - some nested data may not be available")
        
        # Convert DataFrame records to dict format
        all_records = patient_data.to_dict('records')
        
        # Continue with regular processing
        patient_groups = {}
        for record in all_records:
            subject_id = record['subject_id']
            if subject_id not in patient_groups:
                patient_groups[subject_id] = []
            patient_groups[subject_id].append(record)
        
        for subject_id, patient_records in tqdm(patient_groups.items(), desc="Processing from CSV"):
            try:
                self.process_patient(int(subject_id), patient_records)
            except Exception as e:
                logger.error(f"Failed to process patient {subject_id}: {str(e)}")
    
    def _log_statistics(self, all_records: List[Dict], processed: int, failed: int):
        """Log comprehensive integration statistics"""
        # Calculate statistics
        total_patients = len(set(r['subject_id'] for r in all_records))
        total_records = len(all_records)
        total_stays = len(set(r['stay_id'] for r in all_records if r.get('stay_id')))
        total_admissions = len(set(r['hadm_id'] for r in all_records if r.get('hadm_id')))
        
        # Data completeness
        with_clinical = sum(1 for r in all_records if r.get('clinical_features'))
        with_labs = sum(1 for r in all_records if r.get('lab_results', {}).get('labs_available'))
        with_meds = sum(1 for r in all_records if r.get('medications', {}).get('medications_available'))
        with_diag = sum(1 for r in all_records if r.get('diagnoses', {}).get('diagnoses_available'))
        with_proc = sum(1 for r in all_records if r.get('procedures', {}).get('procedures_available'))
        
        logger.info("="*60)
        logger.info("Phase 3 Comprehensive Statistics:")
        logger.info(f"  Total patients:           {total_patients:,}")
        logger.info(f"  Processed successfully:   {processed:,}")
        logger.info(f"  Failed:                   {failed:,}")
        logger.info(f"  Total records:            {total_records:,}")
        logger.info(f"  Total ED stays:           {total_stays:,}")
        logger.info(f"  Total hospital admissions: {total_admissions:,}")
        logger.info(f"  Avg records/patient:      {total_records/total_patients:.2f}")
        logger.info("")
        logger.info("Data Completeness:")
        logger.info(f"  With clinical features:   {with_clinical:,} ({with_clinical/total_records*100:.1f}%)")
        logger.info(f"  With lab results:         {with_labs:,} ({with_labs/total_records*100:.1f}%)")
        logger.info(f"  With medications:         {with_meds:,} ({with_meds/total_records*100:.1f}%)")
        logger.info(f"  With diagnoses:           {with_diag:,} ({with_diag/total_records*100:.1f}%)")
        logger.info(f"  With procedures:          {with_proc:,} ({with_proc/total_records*100:.1f}%)")
        logger.info("="*60)


def main():
    """Main entry point for Phase 3"""
    import argparse
    from .utils import setup_logging
    
    parser = argparse.ArgumentParser(
        description='Phase 3: Integrate comprehensive clinical data into patient folders'
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
    parser.add_argument(
        '--skip-images',
        action='store_true',
        help='Skip copying CXR images'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Run Phase 3
    integrator = DataIntegrator()
    
    # Override settings based on arguments
    if args.dry_run or args.skip_images:
        logger.info("Images will not be copied")
        integrator.copy_images = False
    
    integrator.run()


if __name__ == '__main__':
    main()