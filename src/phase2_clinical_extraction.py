"""
Phase 2: Clinical Data Extraction
Extracts comprehensive clinical features, labs, medications, diagnoses, procedures, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import timedelta
from loguru import logger
from tqdm import tqdm
import json

from .config_manager import get_config
from .utils import S3Handler, DataValidator


class ClinicalExtractor:
    """Extract comprehensive clinical data for patients"""
    
    def __init__(self):
        """Initialize Clinical Extractor"""
        self.config = get_config()
        self.s3 = S3Handler(
            region=self.config.get('aws.region'),
            profile=self.config.get('aws.profile')
        )
        
        # Time windows for data extraction
        self.lab_lookback_hours = self.config.get(
            'preprocessing.phase2.lab_lookback_hours', 24
        )
        self.med_lookback_hours = self.config.get(
            'preprocessing.phase2.medication_lookback_hours', 48
        )
        self.vitals_lookback_hours = self.config.get(
            'preprocessing.phase2.vitals_lookback_hours', 24
        )
        
        # Chunk sizes for large files
        self.lab_chunk_size = self.config.get(
            'preprocessing.phase2.lab_chunk_size', 100000
        )
        self.chart_chunk_size = self.config.get(
            'preprocessing.phase2.chart_chunk_size', 100000
        )
        
        # Cache for frequently used smaller datasets
        self._patients_cache = None
        self._triage_cache = None
        self._d_labitems_cache = None
        self._d_items_cache = None
        self._diagnoses_cache = None
        self._procedures_cache = None
        
        # Important lab item IDs (can be configured)
        self.important_lab_items = {
            51221: 'Hematocrit',
            51222: 'Hemoglobin',
            51265: 'Platelet Count',
            51301: 'White Blood Cells',
            50912: 'Creatinine',
            50971: 'Potassium',
            50983: 'Sodium',
            50902: 'Chloride',
            50882: 'Bicarbonate',
            50931: 'Glucose',
            51006: 'Urea Nitrogen',
            50813: 'Lactate',
            51003: 'Troponin T',
            50889: 'C-Reactive Protein',
            51237: 'INR',
            51274: 'PT',
            51275: 'PTT'
        }
        
        logger.info("Clinical Extractor initialized with comprehensive extraction")
    
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
            # Check if custom bucket is specified for ED data
            custom_bucket = self.config.get('data_paths.mimic_ed.bucket')
            mimic_bucket = custom_bucket if custom_bucket else self.config.get('aws.s3.mimic_bucket')

            triage_path = self.config.get_data_path('mimic_ed', 'triage')
            logger.info(f"Loading triage data from s3://{mimic_bucket}/{triage_path}")
            self._triage_cache = self.s3.read_csv(
                mimic_bucket,
                triage_path,
                compression='gzip'
            )
        return self._triage_cache
    
    def extract_labs_chunked(
        self,
        subject_id: int,
        study_datetime: pd.Timestamp,
        hadm_id: Optional[int] = None
    ) -> Dict:
        """
        Extract lab results using chunked reading for large files
        
        Args:
            subject_id: Patient ID
            study_datetime: CXR timestamp
            hadm_id: Optional hospital admission ID
            
        Returns:
            Dictionary with lab results
        """
        try:
            lookback = timedelta(hours=self.lab_lookback_hours)
            start_time = study_datetime - lookback
            
            mimic_bucket = self.config.get('aws.s3.mimic_bucket')
            labevents_path = 'hosp/labevents.csv.gz'
            
            logger.debug(f"Extracting labs for subject {subject_id} from {start_time} to {study_datetime}")
            
            # Process in chunks
            relevant_labs = []
            total_chunks = 0
            
            for chunk in pd.read_csv(
                f"s3://{mimic_bucket}/{labevents_path}",
                chunksize=self.lab_chunk_size,
                compression='gzip',
                usecols=['subject_id', 'hadm_id', 'charttime', 'itemid', 
                        'value', 'valuenum', 'valueuom', 'flag'],
                dtype={'subject_id': int, 'hadm_id': 'float64', 'itemid': int}
            ):
                total_chunks += 1
                
                # Filter for this patient
                patient_chunk = chunk[chunk['subject_id'] == subject_id]
                
                if len(patient_chunk) == 0:
                    continue
                
                # Parse datetime and filter time window
                patient_chunk['charttime'] = pd.to_datetime(patient_chunk['charttime'])
                
                mask = (
                    (patient_chunk['charttime'] >= start_time) &
                    (patient_chunk['charttime'] <= study_datetime)
                )
                
                if hadm_id is not None:
                    mask = mask & (patient_chunk['hadm_id'] == hadm_id)
                
                filtered_chunk = patient_chunk[mask]
                
                if len(filtered_chunk) > 0:
                    relevant_labs.append(filtered_chunk)
                
                # Stop after processing reasonable number of chunks
                if total_chunks >= 50:  # Adjust based on file size
                    logger.warning(f"Stopped lab extraction after {total_chunks} chunks")
                    break
            
            if not relevant_labs:
                return {
                    'lab_count': 0,
                    'labs_available': False,
                    'important_labs': {},
                    'all_labs': []
                }
            
            # Combine all relevant labs
            labs_df = pd.concat(relevant_labs, ignore_index=True)
            
            # Extract important labs
            important_lab_values = {}
            for itemid, name in self.important_lab_items.items():
                lab_values = labs_df[labs_df['itemid'] == itemid]
                if len(lab_values) > 0:
                    # Get most recent value
                    most_recent = lab_values.nlargest(1, 'charttime').iloc[0]
                    important_lab_values[name] = {
                        'value': most_recent['value'],
                        'valuenum': self._safe_float(most_recent['valuenum']),
                        'charttime': str(most_recent['charttime']),
                        'flag': most_recent.get('flag', None)
                    }
            
            return {
                'lab_count': len(labs_df),
                'unique_labs': labs_df['itemid'].nunique(),
                'labs_available': True,
                'important_labs': important_lab_values,
                'most_recent_charttime': str(labs_df['charttime'].max()),
                'abnormal_count': (labs_df['flag'] == 'abnormal').sum() if 'flag' in labs_df else 0
            }
            
        except Exception as e:
            logger.error(f"Error extracting labs: {str(e)}")
            return {
                'lab_count': 0,
                'labs_available': False,
                'error': str(e)
            }
    
    def extract_medications_chunked(
        self,
        subject_id: int,
        study_datetime: pd.Timestamp,
        hadm_id: Optional[int] = None
    ) -> Dict:
        """
        Extract medications using chunked reading
        
        Args:
            subject_id: Patient ID
            study_datetime: CXR timestamp
            hadm_id: Optional hospital admission ID
            
        Returns:
            Dictionary with medication data
        """
        try:
            lookback = timedelta(hours=self.med_lookback_hours)
            start_time = study_datetime - lookback
            
            mimic_bucket = self.config.get('aws.s3.mimic_bucket')
            prescriptions_path = 'hosp/prescriptions.csv.gz'
            
            logger.debug(f"Extracting medications for subject {subject_id}")
            
            # Process prescriptions in chunks
            relevant_meds = []
            total_chunks = 0
            
            for chunk in pd.read_csv(
                f"s3://{mimic_bucket}/{prescriptions_path}",
                chunksize=self.lab_chunk_size,
                compression='gzip',
                usecols=['subject_id', 'hadm_id', 'starttime', 'stoptime', 
                        'drug', 'drug_type', 'dose_val_rx', 'dose_unit_rx', 'route'],
                dtype={'subject_id': int, 'hadm_id': 'float64'}
            ):
                total_chunks += 1
                
                # Filter for this patient
                patient_chunk = chunk[chunk['subject_id'] == subject_id]
                
                if len(patient_chunk) == 0:
                    continue
                
                # Parse datetime
                patient_chunk['starttime'] = pd.to_datetime(patient_chunk['starttime'], errors='coerce')
                patient_chunk['stoptime'] = pd.to_datetime(patient_chunk['stoptime'], errors='coerce')
                
                # Filter for medications active during time window
                mask = (
                    (patient_chunk['starttime'] <= study_datetime) &
                    ((patient_chunk['stoptime'].isna()) | (patient_chunk['stoptime'] >= start_time))
                )
                
                if hadm_id is not None:
                    mask = mask & (patient_chunk['hadm_id'] == hadm_id)
                
                filtered_chunk = patient_chunk[mask]
                
                if len(filtered_chunk) > 0:
                    relevant_meds.append(filtered_chunk)
                
                # Stop after reasonable number of chunks
                if total_chunks >= 50:
                    logger.warning(f"Stopped medication extraction after {total_chunks} chunks")
                    break
            
            if not relevant_meds:
                return {
                    'medication_count': 0,
                    'medications_available': False,
                    'medications': []
                }
            
            # Combine all relevant medications
            meds_df = pd.concat(relevant_meds, ignore_index=True)
            
            # Get unique medications
            unique_meds = meds_df.groupby('drug').agg({
                'dose_val_rx': 'last',
                'dose_unit_rx': 'last',
                'route': 'last',
                'starttime': 'min'
            }).reset_index()
            
            # Create medication list
            med_list = []
            for _, row in unique_meds.head(20).iterrows():
                med_list.append({
                    'drug': row['drug'],
                    'dose': self._safe_float(row['dose_val_rx']),
                    'unit': row['dose_unit_rx'],
                    'route': row['route'],
                    'started': str(row['starttime']) if pd.notna(row['starttime']) else None
                })
            
            # Identify antibiotics
            antibiotics = meds_df[
                meds_df['drug'].str.contains(
                    'cillin|mycin|floxacin|cephalexin|azithromycin|metronidazole',
                    case=False, na=False
                )
            ]['drug'].unique().tolist()
            
            return {
                'medication_count': len(meds_df),
                'unique_medications': len(unique_meds),
                'medications_available': True,
                'medications': med_list,
                'antibiotics': antibiotics,
                'has_antibiotics': len(antibiotics) > 0
            }
            
        except Exception as e:
            logger.error(f"Error extracting medications: {str(e)}")
            return {
                'medication_count': 0,
                'medications_available': False,
                'error': str(e)
            }
    
    def extract_diagnoses(
        self,
        subject_id: int,
        hadm_id: Optional[int] = None
    ) -> Dict:
        """Extract diagnosis codes"""
        try:
            diagnoses = self.load_diagnoses_for_patient(subject_id, hadm_id)
            
            if len(diagnoses) == 0:
                return {
                    'diagnoses_available': False,
                    'diagnosis_count': 0
                }
            
            # Group by ICD version
            icd9_diagnoses = diagnoses[diagnoses['icd_version'] == 9]
            icd10_diagnoses = diagnoses[diagnoses['icd_version'] == 10]
            
            # Get primary diagnoses (seq_num = 1)
            primary_diagnoses = diagnoses[diagnoses['seq_num'] == 1]
            
            return {
                'diagnoses_available': True,
                'diagnosis_count': len(diagnoses),
                'unique_diagnoses': diagnoses['icd_code'].nunique(),
                'icd9_count': len(icd9_diagnoses),
                'icd10_count': len(icd10_diagnoses),
                'primary_diagnoses': primary_diagnoses['icd_code'].tolist(),
                'all_diagnoses': diagnoses[['icd_code', 'icd_version']].to_dict('records')[:50]
            }
            
        except Exception as e:
            logger.error(f"Error extracting diagnoses: {str(e)}")
            return {
                'diagnoses_available': False,
                'error': str(e)
            }
    
    def load_diagnoses_for_patient(self, subject_id: int, hadm_id: Optional[int] = None) -> pd.DataFrame:
        """Load diagnoses for a specific patient"""
        if self._diagnoses_cache is None:
            mimic_bucket = self.config.get('aws.s3.mimic_bucket')
            diagnoses_path = 'hosp/diagnoses_icd.csv.gz'
            logger.debug(f"Loading diagnoses from s3://{mimic_bucket}/{diagnoses_path}")
            self._diagnoses_cache = self.s3.read_csv(
                mimic_bucket,
                diagnoses_path,
                compression='gzip'
            )
        
        patient_diagnoses = self._diagnoses_cache[
            self._diagnoses_cache['subject_id'] == subject_id
        ]
        
        if hadm_id is not None:
            patient_diagnoses = patient_diagnoses[
                patient_diagnoses['hadm_id'] == hadm_id
            ]
        
        return patient_diagnoses
    
    def load_procedures_for_patient(self, subject_id: int, hadm_id: Optional[int] = None) -> pd.DataFrame:
        """Load procedures for a specific patient"""
        if self._procedures_cache is None:
            mimic_bucket = self.config.get('aws.s3.mimic_bucket')
            procedures_path = 'hosp/procedures_icd.csv.gz'
            logger.debug(f"Loading procedures from s3://{mimic_bucket}/{procedures_path}")
            self._procedures_cache = self.s3.read_csv(
                mimic_bucket,
                procedures_path,
                compression='gzip'
            )
        
        patient_procedures = self._procedures_cache[
            self._procedures_cache['subject_id'] == subject_id
        ]
        
        if hadm_id is not None:
            patient_procedures = patient_procedures[
                patient_procedures['hadm_id'] == hadm_id
            ]
        
        return patient_procedures
    
    def extract_procedures(
        self,
        subject_id: int,
        hadm_id: Optional[int] = None
    ) -> Dict:
        """Extract procedure codes"""
        try:
            procedures = self.load_procedures_for_patient(subject_id, hadm_id)
            
            if len(procedures) == 0:
                return {
                    'procedures_available': False,
                    'procedure_count': 0
                }
            
            # Group by ICD version
            icd9_procedures = procedures[procedures['icd_version'] == 9]
            icd10_procedures = procedures[procedures['icd_version'] == 10]
            
            return {
                'procedures_available': True,
                'procedure_count': len(procedures),
                'unique_procedures': procedures['icd_code'].nunique(),
                'icd9_count': len(icd9_procedures),
                'icd10_count': len(icd10_procedures),
                'procedures': procedures[['icd_code', 'icd_version', 'chartdate']].to_dict('records')[:20]
            }
            
        except Exception as e:
            logger.error(f"Error extracting procedures: {str(e)}")
            return {
                'procedures_available': False,
                'error': str(e)
            }
    
    def get_hospital_admission_id(
        self,
        subject_id: int,
        study_datetime: pd.Timestamp
    ) -> Optional[int]:
        """Find hospital admission ID for the given time"""
        try:
            mimic_bucket = self.config.get('aws.s3.mimic_bucket')
            admissions_path = 'hosp/admissions.csv.gz'
            
            # Load admissions (this is a smaller file, can load fully)
            admissions = self.s3.read_csv(
                mimic_bucket,
                admissions_path,
                compression='gzip'
            )
            
            # Filter for patient
            patient_admissions = admissions[admissions['subject_id'] == subject_id].copy()
            
            if len(patient_admissions) == 0:
                return None
            
            # Parse dates
            patient_admissions['admittime'] = pd.to_datetime(patient_admissions['admittime'])
            patient_admissions['dischtime'] = pd.to_datetime(patient_admissions['dischtime'])
            
            # Find admission containing the study datetime
            mask = (
                (patient_admissions['admittime'] <= study_datetime) &
                (patient_admissions['dischtime'] >= study_datetime)
            )
            
            matches = patient_admissions[mask]
            
            if len(matches) > 0:
                return int(matches.iloc[0]['hadm_id'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding hospital admission: {str(e)}")
            return None
    
    def calculate_age(
        self,
        patient_row: pd.Series,
        visit_datetime: pd.Timestamp
    ) -> Optional[int]:
        """Calculate patient age at time of visit"""
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
        """Extract clinical features from ED triage and patient data"""
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
                'acuity': self._safe_int(triage_row.get('acuity')),
                'chiefcomplaint': str(triage_row.get('chiefcomplaint', ''))
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting clinical features: {str(e)}")
            return None
    
    def extract_vital_signs_detailed(
        self,
        subject_id: int,
        stay_id: int,
        study_datetime: pd.Timestamp
    ) -> Dict:
        """Extract detailed vital signs from chartevents"""
        try:
            # For now, return enhanced triage vitals
            # Full chartevents extraction would require chunked processing
            triage = self.load_triage()
            stay_triage = triage[triage['stay_id'] == stay_id]
            
            if len(stay_triage) > 0:
                triage_row = stay_triage.iloc[0]
                return {
                    'vitals_source': 'triage',
                    'heart_rate': self._safe_float(triage_row.get('heartrate')),
                    'respiratory_rate': self._safe_float(triage_row.get('resprate')),
                    'spo2': self._safe_float(triage_row.get('o2sat')),
                    'systolic_bp': self._safe_float(triage_row.get('sbp')),
                    'diastolic_bp': self._safe_float(triage_row.get('dbp')),
                    'temperature': self._safe_float(triage_row.get('temperature')),
                    'detailed_available': False,
                    'note': 'Full chartevents extraction requires database or Athena'
                }
            
            return {
                'vitals_source': None,
                'detailed_available': False
            }
            
        except Exception as e:
            logger.error(f"Error extracting vital signs: {str(e)}")
            return {'error': str(e)}
    
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
    
    def process_patient(
        self,
        subject_id: int,
        patient_cxrs: pd.DataFrame
    ) -> List[Dict]:
        """Process all CXRs for a single patient with comprehensive data extraction"""
        records = []
        
        for _, cxr_row in patient_cxrs.iterrows():
            # Skip if no stay_id
            if pd.isna(cxr_row['stay_id']):
                continue
            
            try:
                study_datetime = pd.to_datetime(cxr_row['study_datetime'])
                
                # Find hospital admission if exists
                hadm_id = self.get_hospital_admission_id(subject_id, study_datetime)
                
                record = {
                    'subject_id': int(subject_id),
                    'stay_id': int(cxr_row['stay_id']),
                    'hadm_id': hadm_id,
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
                
                # Extract comprehensive labs
                lab_data = self.extract_labs_chunked(
                    subject_id, 
                    study_datetime,
                    hadm_id
                )
                record['lab_results'] = lab_data
                
                # Extract medications
                med_data = self.extract_medications_chunked(
                    subject_id,
                    study_datetime,
                    hadm_id
                )
                record['medications'] = med_data
                
                # Extract detailed vitals
                vitals_data = self.extract_vital_signs_detailed(
                    subject_id,
                    int(cxr_row['stay_id']),
                    study_datetime
                )
                record['vital_signs'] = vitals_data
                
                # Extract diagnoses
                diagnoses_data = self.extract_diagnoses(subject_id, hadm_id)
                record['diagnoses'] = diagnoses_data
                
                # Extract procedures
                procedures_data = self.extract_procedures(subject_id, hadm_id)
                record['procedures'] = procedures_data
                
                records.append(record)
                
            except Exception as e:
                logger.error(
                    f"Error processing CXR {cxr_row.get('dicom_id')}: {str(e)}"
                )
                continue
        
        return records
    
    def run(self, chunk_id: Optional[int] = None, sample_size: Optional[int] = None):
        """Execute Phase 2: Comprehensive clinical data extraction"""
        logger.info("="*60)
        logger.info("Starting Phase 2: Comprehensive Clinical Data Extraction")
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
        
        # Apply sample size if specified
        if sample_size:
            patient_ids = list(patients_to_process.groups.keys())[:sample_size]
            logger.info(f"Processing sample of {sample_size} patients")
        else:
            patient_ids = list(patients_to_process.groups.keys())
        
        for subject_id in tqdm(patient_ids, desc="Processing patients"):
            patient_cxrs = patients_to_process.get_group(subject_id)
            patient_records = self.process_patient(subject_id, patient_cxrs)
            all_records.extend(patient_records)
        
        # Convert to DataFrame for easier handling
        if all_records:
            results_df = pd.DataFrame(all_records)
            
            # Save full results as JSON (preserves nested structure)
            if chunk_id is not None:
                json_key = f'processing/phase2_results/chunk_{chunk_id}.json'
            else:
                json_key = 'processing/patient_clinical_data_comprehensive.json'
            
            logger.info(f"Saving comprehensive results to s3://{output_bucket}/{json_key}")
            
            # Convert to JSON and save
            json_data = json.dumps(all_records, indent=2, default=str)
            self.s3.s3_client.put_object(
                Bucket=output_bucket,
                Key=json_key,
                Body=json_data.encode('utf-8')
            )
            
            # Also save a flattened CSV for easier querying
            flattened_df = self._flatten_for_csv(results_df)
            
            if chunk_id is not None:
                csv_key = f'processing/phase2_results/chunk_{chunk_id}_flat.csv'
            else:
                csv_key = 'processing/patient_clinical_data_flat.csv'
            
            self.s3.write_csv(flattened_df, output_bucket, csv_key)
            
            # Log statistics
            self._log_statistics(results_df)
        else:
            logger.warning("No records to save!")
        
        logger.info("Phase 2 complete!")
        return all_records
    
    def _flatten_for_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested dictionaries for CSV export"""
        flat_df = df[['subject_id', 'stay_id', 'hadm_id', 'dicom_id', 
                     'study_id', 'study_datetime', 'ViewPosition']].copy()
        
        # Add clinical features
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
    
    def _log_statistics(self, results: pd.DataFrame):
        """Log comprehensive processing statistics"""
        total = len(results)
        unique_patients = results['subject_id'].nunique()
        
        # Count available data types
        with_clinical = sum(1 for _, r in results.iterrows() 
                          if r.get('clinical_features') and r['clinical_features'])
        with_labs = sum(1 for _, r in results.iterrows()
                       if r.get('lab_results', {}).get('labs_available', False))
        with_meds = sum(1 for _, r in results.iterrows()
                       if r.get('medications', {}).get('medications_available', False))
        with_diagnoses = sum(1 for _, r in results.iterrows()
                           if r.get('diagnoses', {}).get('diagnoses_available', False))
        with_procedures = sum(1 for _, r in results.iterrows()
                            if r.get('procedures', {}).get('procedures_available', False))
        
        logger.info("="*60)
        logger.info("Phase 2 Comprehensive Statistics:")
        logger.info(f"  Total records:              {total:,}")
        logger.info(f"  Unique patients:            {unique_patients:,}")
        logger.info(f"  Records with clinical data: {with_clinical:,} ({with_clinical/total*100:.1f}%)")
        logger.info(f"  Records with lab results:   {with_labs:,} ({with_labs/total*100:.1f}%)")
        logger.info(f"  Records with medications:   {with_meds:,} ({with_meds/total*100:.1f}%)")
        logger.info(f"  Records with diagnoses:     {with_diagnoses:,} ({with_diagnoses/total*100:.1f}%)")
        logger.info(f"  Records with procedures:    {with_procedures:,} ({with_procedures/total*100:.1f}%)")
        logger.info("="*60)


def main():
    """Main entry point for Phase 2"""
    import argparse
    from .utils import setup_logging
    
    parser = argparse.ArgumentParser(
        description='Phase 2: Extract comprehensive clinical data'
    )
    parser.add_argument(
        '--chunk-id',
        type=int,
        help='Process specific chunk (for parallel processing)'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Process only N patients (for testing)'
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
    extractor.run(chunk_id=args.chunk_id, sample_size=args.sample_size)


if __name__ == '__main__':
    main()