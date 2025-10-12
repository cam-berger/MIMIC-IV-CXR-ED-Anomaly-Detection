"""
Phase 2: Clinical Data Extraction
Extracts clinical features, labs, medications, etc.
"""

import pandas as pd
from typing import Dict, Optional, List
from datetime import timedelta
from loguru import logger
from tqdm import tqdm

from .config_manager import get_config
from .utils import S3Handler, DataValidator


class ClinicalExtractor:
    """Extract clinical data for patients"""
    
    def __init__(self):
        """Initialize Clinical Extractor"""
        self.config = get_config()
        self.s3 = S3Handler(
            region=self.config.get('aws.region'),
            profile=self.config.get('aws.profile')
        )
        self.lab_lookback_hours = self.config.get(
            'preprocessing.phase2.lab_lookback_hours', 24
        )
        self.med_lookback_hours = self.config.get(
            'preprocessing.phase2.medication_lookback_hours', 48
        )
        
        # Cache for large datasets
        self._patients_cache = None
        self._triage_cache = None
        
        logger.info("Clinical Extractor initialized")
    
    def load_patients(self) -> pd.DataFrame:
        """Load patients data (with caching)"""
        if self._patients_cache is None:
            mimic_bucket = self.config.get('aws.s3.mimic_bucket')
            patients_path = self.config.get_data_path('mimic_iv', 'patients')
            logger.info(f"Loading patients data from s3://{mimic_bucket}/{patients_path}")
            self._patients_cache = self.s3.read_csv(
                mimic_bucket, 
                patients_path, 
                compression='gzip'
            )
        return self._patients_cache
    
    def load_triage(self) -> pd.DataFrame:
        """Load triage data (with caching)"""
        if self._triage_cache is None:
            mimic_bucket = self.config.get('aws.s3.mimic_bucket')
            triage_path = self.config.get_data_path('mimic_ed', 'triage')
            logger.info(f"Loading triage data from s3://{mimic_bucket}/{triage_path}")
            self._triage_cache = self.s3.read_csv(
                mimic_bucket,
                triage_path,
                compression='gzip'
            )
        return self._triage_cache
    
    def calculate_age(
        self,
        patient_row: pd.Series,
        visit_datetime: pd.Timestamp
    ) -> Optional[int]:
        """
        Calculate patient age at time of visit
        
        Args:
            patient_row: Row from patients DataFrame
            visit_datetime: DateTime of visit
            
        Returns:
            Age in years
        """
        try:
            anchor_age = patient_row['anchor_age']
            anchor_year = patient_row['anchor_year']
            visit_year = visit_datetime.year
            
            age = anchor_age + (visit_year - anchor_year)
            return int(age) if age >= 0 else None
        except:
            return None
    
    def extract_clinical_features(
        self,
        subject_id: int,
        stay_id: int,
        study_datetime: pd.Timestamp
    ) -> Optional[Dict]:
        """
        Extract clinical features from ED triage and patient data
        
        Args:
            subject_id: Patient ID
            stay_id: ED stay ID
            study_datetime: DateTime of CXR
            
        Returns:
            Dictionary of clinical features
        """
        try:
            # Get triage data
            triage = self.load_triage()
            stay_triage = triage[triage['stay_id'] == stay_id]
            
            if len(stay_triage) == 0:
                logger.debug(f"No triage data for stay {stay_id}")
                return None
            
            triage_row = stay_triage.iloc[0]
            
            # Get patient demographics
            patients = self.load_patients()
            patient = patients[patients['subject_id'] == subject_id]
            
            if len(patient) == 0:
                logger.warning(f"No patient data for subject {subject_id}")
                return None
            
            patient_row = patient.iloc[0]
            
            # Extract features
            features = {
                'subject_id': int(subject_id),
                'stay_id': int(stay_id),
                'age': self.calculate_age(patient_row, study_datetime),
                'gender': patient_row['gender'],
                'temperature': self._safe_float(triage_row.get('temperature')),
                'heartrate': self._safe_float(triage_row.get('heartrate')),
                'resprate': self._safe_float(triage_row.get('resprate')),
                'o2sat': self._safe_float(triage_row.get('o2sat')),
                'sbp': self._safe_float(triage_row.get('sbp')),
                'dbp': self._safe_float(triage_row.get('dbp')),
                'pain': self._safe_float(triage_row.get('pain')),
                'acuity': self._safe_int(triage_row.get('acuity'))
            }
            
            # Validate
            if not DataValidator.validate_clinical_features(features):
                logger.warning(f"Incomplete clinical features for stay {stay_id}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting clinical features: {str(e)}")
            return None
    
    def _safe_float(self, value) -> Optional[float]:
        """Safely convert to float"""
        try:
            return float(value) if pd.notna(value) else None
        except:
            return None
    
    def _safe_int(self, value) -> Optional[int]:
        """Safely convert to int"""
        try:
            return int(value) if pd.notna(value) else None
        except:
            return None
    
    def extract_lab_summary(
        self,
        subject_id: int,
        study_datetime: pd.Timestamp
    ) -> Dict:
        """
        Extract summary of lab results within time window
        Note: Full labevents is too large to load entirely
        This returns a summary instead
        
        Args:
            subject_id: Patient ID
            study_datetime: CXR timestamp
            
        Returns:
            Dictionary with lab summary
        """
        try:
            # In production, you would:
            # 1. Use chunked reading or
            # 2. Query a database or
            # 3. Use AWS Athena for querying large files
            
            # For now, return placeholder
            return {
                'lab_available': False,
                'lab_count': 0,
                'note': 'Lab extraction requires chunked processing or database query'
            }
            
        except Exception as e:
            logger.error(f"Error extracting labs: {str(e)}")
            return {'error': str(e)}
    
    def extract_medication_summary(
        self,
        subject_id: int,
        study_datetime: pd.Timestamp
    ) -> Dict:
        """
        Extract summary of medications
        
        Args:
            subject_id: Patient ID
            study_datetime: CXR timestamp
            
        Returns:
            Dictionary with medication summary
        """
        try:
            # Similar to labs, prescriptions file is large
            # In production, use chunked reading or database
            
            return {
                'medications_available': False,
                'medication_count': 0,
                'note': 'Medication extraction requires chunked processing or database query'
            }
            
        except Exception as e:
            logger.error(f"Error extracting medications: {str(e)}")
            return {'error': str(e)}
    
    def process_patient(
        self,
        subject_id: int,
        patient_cxrs: pd.DataFrame
    ) -> List[Dict]:
        """
        Process all CXRs for a single patient
        
        Args:
            subject_id: Patient ID
            patient_cxrs: DataFrame of CXRs for this patient
            
        Returns:
            List of dictionaries with patient data
        """
        records = []
        
        for _, cxr_row in patient_cxrs.iterrows():
            # Skip if no stay_id
            if pd.isna(cxr_row['stay_id']):
                continue
            
            try:
                study_datetime = pd.to_datetime(cxr_row['study_datetime'])
                
                record = {
                    'subject_id': int(subject_id),
                    'stay_id': int(cxr_row['stay_id']),
                    'dicom_id': cxr_row['dicom_id'],
                    'study_id': cxr_row['study_id'],
                    'study_datetime': str(study_datetime),
                    'ViewPosition': cxr_row.get('ViewPosition')
                }
                
                # Extract clinical features
                clinical = self.extract_clinical_features(
                    subject_id,
                    int(cxr_row['stay_id']),
                    study_datetime
                )
                record['clinical_features'] = clinical
                
                # Extract lab summary
                lab_summary = self.extract_lab_summary(subject_id, study_datetime)
                record['lab_summary'] = lab_summary
                
                # Extract medication summary
                med_summary = self.extract_medication_summary(subject_id, study_datetime)
                record['medication_summary'] = med_summary
                
                records.append(record)
                
            except Exception as e:
                logger.error(
                    f"Error processing CXR {cxr_row.get('dicom_id')}: {str(e)}"
                )
                continue
        
        return records
    
    def run(self, chunk_id: Optional[int] = None):
        """
        Execute Phase 2: Clinical data extraction
        
        Args:
            chunk_id: If provided, process only this chunk
        """
        logger.info("="*60)
        logger.info("Starting Phase 2: Clinical Data Extraction")
        logger.info("="*60)
        
        # Load Phase 1 results
        output_bucket = self.config.get('aws.s3.output_bucket')
        
        if chunk_id is not None:
            input_key = f'processing/phase1_results/chunk_{chunk_id}.csv'
        else:
            input_key = 'processing/cxr_with_stays.csv'
        
        logger.info(f"Loading Phase 1 results from s3://{output_bucket}/{input_key}")
        cxr_with_stays = self.s3.read_csv(output_bucket, input_key)
        
        # Filter records with stay_id
        cxr_with_stays = cxr_with_stays[cxr_with_stays['stay_id'].notna()]
        logger.info(f"Processing {len(cxr_with_stays)} CXR records with stay_id")
        
        # Process by patient
        all_records = []
        
        patients_to_process = cxr_with_stays.groupby('subject_id')
        
        for subject_id, patient_cxrs in tqdm(
            patients_to_process,
            desc="Processing patients",
            total=len(patients_to_process)
        ):
            patient_records = self.process_patient(subject_id, patient_cxrs)
            all_records.extend(patient_records)
        
        # Convert to DataFrame
        if all_records:
            results_df = pd.DataFrame(all_records)
            
            # Save results
            if chunk_id is not None:
                output_key = f'processing/phase2_results/chunk_{chunk_id}.csv'
            else:
                output_key = 'processing/patient_clinical_data.csv'
            
            logger.info(f"Saving results to s3://{output_bucket}/{output_key}")
            self.s3.write_csv(results_df, output_bucket, output_key)
            
            # Log statistics
            self._log_statistics(results_df)
        else:
            logger.warning("No records to save!")
        
        logger.info("Phase 2 complete!")
        return all_records
    
    def _log_statistics(self, results: pd.DataFrame):
        """Log processing statistics"""
        total = len(results)
        unique_patients = results['subject_id'].nunique()
        with_clinical = results['clinical_features'].notna().sum()
        
        logger.info("="*60)
        logger.info("Phase 2 Statistics:")
        logger.info(f"  Total records:              {total:,}")
        logger.info(f"  Unique patients:            {unique_patients:,}")
        logger.info(f"  Records with clinical data: {with_clinical:,}")
        logger.info("="*60)


def main():
    """Main entry point for Phase 2"""
    import argparse
    from .utils import setup_logging
    
    parser = argparse.ArgumentParser(
        description='Phase 2: Extract clinical data'
    )
    parser.add_argument(
        '--chunk-id',
        type=int,
        help='Process specific chunk (for parallel processing)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Run Phase 2
    extractor = ClinicalExtractor()
    extractor.run(chunk_id=args.chunk_id)


if __name__ == '__main__':
    main()