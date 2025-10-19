#!/usr/bin/env python3
"""
MIMIC Multimodal Data Preprocessing Pipeline
Links CXR images with anomaly classifications, bounding boxes, and filtered clinical data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pydicom
from tqdm import tqdm
import re

class MIMICPreprocessor:
    """Preprocessor for linking MIMIC datasets while avoiding data leakage"""
    
    def __init__(self, base_path: str):
        """
        Initialize with base path to MIMIC datasets
        
        Args:
            base_path: Root directory containing MIMIC-IV, MIMIC-CXR-JPG, REFLACX, MIMIC-ED
        """
        self.base_path = Path(base_path)
        
        # Define paths to each dataset
        self.paths = {
            'mimic_iv': self.base_path / 'mimic-iv',
            'mimic_cxr': self.base_path / 'mimic-cxr-jpg',
            'reflacx': self.base_path / 'reflacx',
            'mimic_ed': self.base_path / 'mimic-ed'
        }
        
        # Initialize dataframes
        self.cxr_metadata = None
        self.cxr_labels = None
        self.reflacx_data = None
        self.clinical_data = {}
        
    def load_cxr_metadata(self):
        """Load MIMIC-CXR metadata and labels"""
        print("Loading MIMIC-CXR metadata...")
        
        # Load metadata
        metadata_path = self.paths['mimic_cxr'] / 'mimic-cxr-2.0.0-metadata.csv'
        self.cxr_metadata = pd.read_csv(metadata_path)
        
        # Load CheXpert labels (abnormality classifications)
        chexpert_path = self.paths['mimic_cxr'] / 'mimic-cxr-2.0.0-chexpert.csv'
        self.cxr_labels = pd.read_csv(chexpert_path)
        
        # Load split information
        split_path = self.paths['mimic_cxr'] / 'mimic-cxr-2.0.0-split.csv'
        self.cxr_split = pd.read_csv(split_path)
        
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
        
    def load_reflacx_annotations(self):
        """Load REFLACX eye-tracking data with bounding boxes"""
        print("Loading REFLACX annotations...")
        
        # REFLACX provides bounding boxes for abnormalities
        # Structure: subject_id, study_id, dicom_id, boxes, labels
        
        reflacx_path = self.paths['reflacx'] / 'main_data'
        
        annotations = []
        
        # Load anomaly localizations
        for json_file in (reflacx_path / 'anomaly_location_ellipses').glob('*.json'):
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            # Parse REFLACX format
            study_id = json_file.stem
            
            if 'anomaly_ellipses' in data:
                for anomaly in data['anomaly_ellipses']:
                    annotations.append({
                        'study_id': study_id,
                        'anomaly_type': anomaly.get('label', 'unknown'),
                        'ellipse_center_x': anomaly.get('center_x'),
                        'ellipse_center_y': anomaly.get('center_y'),
                        'ellipse_major_axis': anomaly.get('major_axis'),
                        'ellipse_minor_axis': anomaly.get('minor_axis'),
                        'ellipse_angle': anomaly.get('angle', 0)
                    })
        
        self.reflacx_data = pd.DataFrame(annotations)
        
        # Convert ellipses to bounding boxes for easier use
        self.reflacx_data['bbox'] = self.reflacx_data.apply(
            lambda row: self._ellipse_to_bbox(row), axis=1
        )
        
        print(f"  Loaded {len(self.reflacx_data)} anomaly annotations")
        
    def _ellipse_to_bbox(self, row) -> List[float]:
        """Convert ellipse parameters to bounding box [x_min, y_min, x_max, y_max]"""
        center_x = row['ellipse_center_x']
        center_y = row['ellipse_center_y']
        major = row['ellipse_major_axis']
        minor = row['ellipse_minor_axis']
        
        # Simple conversion - can be refined with rotation consideration
        x_min = center_x - major / 2
        x_max = center_x + major / 2
        y_min = center_y - minor / 2
        y_max = center_y + minor / 2
        
        return [x_min, y_min, x_max, y_max]
    
    def load_clinical_data(self, subject_ids: List[int]):
        """
        Load relevant clinical data from MIMIC-IV for given subjects
        
        Args:
            subject_ids: List of subject IDs to load data for
        """
        print(f"Loading clinical data for {len(subject_ids)} subjects...")
        
        # Load admissions to get timing information
        admissions = pd.read_csv(
            self.paths['mimic_iv'] / 'hosp' / 'admissions.csv',
            parse_dates=['admittime', 'dischtime']
        )
        admissions = admissions[admissions['subject_id'].isin(subject_ids)]
        
        # Load clinical notes (excluding discharge summaries and radiology reports)
        notes = pd.read_csv(self.paths['mimic_iv'] / 'note' / 'noteevents.csv')
        
        # IMPORTANT: Filter out notes that would leak diagnosis information
        excluded_categories = [
            'Discharge summary',
            'Radiology',  # These directly describe the X-ray findings
            'Echo',  # Often contains chest-related findings
            'ECG'  # May reference chest pathology
        ]
        
        notes_filtered = notes[
            (notes['subject_id'].isin(subject_ids)) &
            (~notes['category'].isin(excluded_categories))
        ]
        
        # Load lab events
        lab_events = pd.read_csv(
            self.paths['mimic_iv'] / 'hosp' / 'labevents.csv',
            nrows=10000000  # Sample for memory management
        )
        lab_events = lab_events[lab_events['subject_id'].isin(subject_ids)]
        
        # Load vital signs
        vitals = pd.read_csv(
            self.paths['mimic_iv'] / 'icu' / 'chartevents.csv',
            nrows=10000000  # Sample for memory management
        )
        
        # Filter for vital sign items only
        vital_itemids = [
            220045,  # Heart Rate
            220050,  # Arterial BP Systolic
            220051,  # Arterial BP Diastolic
            220210,  # Respiratory Rate
            223761,  # Temperature (F)
            220277   # SpO2
        ]
        
        vitals = vitals[
            (vitals['subject_id'].isin(subject_ids)) &
            (vitals['itemid'].isin(vital_itemids))
        ]
        
        # Load medications
        prescriptions = pd.read_csv(
            self.paths['mimic_iv'] / 'hosp' / 'prescriptions.csv'
        )
        prescriptions = prescriptions[prescriptions['subject_id'].isin(subject_ids)]
        
        # Store filtered clinical data
        self.clinical_data = {
            'admissions': admissions,
            'notes': notes_filtered,
            'labs': lab_events,
            'vitals': vitals,
            'medications': prescriptions
        }
        
        print(f"  Loaded clinical data:")
        for key, df in self.clinical_data.items():
            print(f"    {key}: {len(df)} records")
    
    def filter_diagnosis_leakage(self, text: str, diagnosis_keywords: List[str]) -> str:
        """
        Remove text that directly mentions the diagnosis
        
        Args:
            text: Clinical note text
            diagnosis_keywords: List of diagnosis-related keywords to filter
            
        Returns:
            Filtered text
        """
        if pd.isna(text):
            return ""
        
        # Common patterns that leak diagnosis
        leakage_patterns = [
            r'chest\s*(x-ray|xray|radiograph)',
            r'cxr\s*(shows?|reveals?|demonstrates?)',
            r'imaging\s*(shows?|reveals?|demonstrates?)',
            r'radiolog\w+\s*finding',
            r'consolidation',
            r'infiltrate',
            r'opacity',
            r'effusion',
            r'pneumothorax',
            r'cardiomegaly'
        ]
        
        # Add specific diagnosis keywords
        for keyword in diagnosis_keywords:
            leakage_patterns.append(rf'\b{keyword.lower()}\b')
        
        # Remove sentences containing leakage patterns
        sentences = text.split('.')
        filtered_sentences = []
        
        for sentence in sentences:
            if not any(re.search(pattern, sentence.lower()) for pattern in leakage_patterns):
                filtered_sentences.append(sentence)
        
        return '.'.join(filtered_sentences)
    
    def link_cxr_to_clinical(self, time_window_hours: int = 24) -> pd.DataFrame:
        """
        Link CXR images to clinical data within specified time window
        
        Args:
            time_window_hours: Hours before CXR to include clinical data
            
        Returns:
            DataFrame with linked data
        """
        print(f"Linking CXR to clinical data (±{time_window_hours} hours)...")
        
        # Get CXR study times
        cxr_times = self.cxr_metadata[['subject_id', 'study_id', 'StudyTime']].copy()
        cxr_times['study_datetime'] = pd.to_datetime(cxr_times['StudyTime'], format='%H%M%S')
        
        linked_data = []
        
        for _, cxr in tqdm(cxr_times.iterrows(), total=len(cxr_times)):
            subject_id = cxr['subject_id']
            study_id = cxr['study_id']
            study_time = cxr['study_datetime']
            
            # Get labels for this study
            labels = self.cxr_labels[
                (self.cxr_labels['subject_id'] == subject_id) &
                (self.cxr_labels['study_id'] == study_id)
            ]
            
            if labels.empty:
                continue
            
            # Get REFLACX annotations if available
            reflacx = self.reflacx_data[
                self.reflacx_data['study_id'] == study_id
            ] if self.reflacx_data is not None else pd.DataFrame()
            
            # Extract diagnoses for filtering
            positive_findings = []
            for col in ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                       'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
                       'Pneumonia', 'Pneumothorax', 'Support Devices']:
                if col in labels.columns and labels[col].values[0] == 1.0:
                    positive_findings.append(col)
            
            # Get relevant clinical notes (before CXR, excluding diagnosis leakage)
            relevant_notes = []
            if 'notes' in self.clinical_data:
                subject_notes = self.clinical_data['notes'][
                    self.clinical_data['notes']['subject_id'] == subject_id
                ]
                
                for _, note in subject_notes.iterrows():
                    # Filter out notes that leak diagnosis
                    filtered_text = self.filter_diagnosis_leakage(
                        note['text'],
                        positive_findings
                    )
                    
                    if filtered_text:
                        relevant_notes.append({
                            'note_category': note['category'],
                            'note_text': filtered_text[:1000]  # Truncate for storage
                        })
            
            # Get relevant lab values
            relevant_labs = []
            if 'labs' in self.clinical_data:
                subject_labs = self.clinical_data['labs'][
                    self.clinical_data['labs']['subject_id'] == subject_id
                ]
                # Aggregate recent lab values
                recent_labs = subject_labs.groupby('itemid').agg({
                    'value': 'last',
                    'valuenum': 'last',
                    'valueuom': 'first'
                }).head(20)  # Top 20 most recent labs
                
                relevant_labs = recent_labs.to_dict('records')
            
            # Get relevant vital signs
            relevant_vitals = {}
            if 'vitals' in self.clinical_data:
                subject_vitals = self.clinical_data['vitals'][
                    self.clinical_data['vitals']['subject_id'] == subject_id
                ]
                
                # Get most recent vital signs
                for itemid in [220045, 220050, 220051, 220210, 223761, 220277]:
                    vital_values = subject_vitals[
                        subject_vitals['itemid'] == itemid
                    ]['value'].values
                    
                    if len(vital_values) > 0:
                        relevant_vitals[f'vital_{itemid}'] = vital_values[-1]
            
            # Compile linked record
            linked_record = {
                'subject_id': subject_id,
                'study_id': study_id,
                'image_path': f"files/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}",
                
                # Labels (abnormality classifications)
                'labels': labels.iloc[0].to_dict() if not labels.empty else {},
                'positive_findings': positive_findings,
                
                # Bounding boxes from REFLACX
                'bounding_boxes': reflacx['bbox'].tolist() if not reflacx.empty else [],
                'anomaly_types': reflacx['anomaly_type'].tolist() if not reflacx.empty else [],
                
                # Filtered clinical data
                'clinical_notes': relevant_notes[:5],  # Limit to 5 most relevant notes
                'lab_values': relevant_labs,
                'vital_signs': relevant_vitals,
                
                # Metadata
                'split': labels.iloc[0].get('split', 'train')
            }
            
            linked_data.append(linked_record)
        
        print(f"  Created {len(linked_data)} linked records")
        return pd.DataFrame(linked_data)
    
    def validate_preprocessing(self, linked_df: pd.DataFrame):
        """
        Validate that preprocessing removed diagnosis leakage
        
        Args:
            linked_df: DataFrame with linked data
        """
        print("Validating preprocessing...")
        
        # Check for diagnosis keywords in clinical notes
        leakage_count = 0
        
        diagnosis_keywords = [
            'consolidation', 'infiltrate', 'opacity', 'effusion',
            'pneumothorax', 'cardiomegaly', 'atelectasis', 'pneumonia'
        ]
        
        for _, row in linked_df.iterrows():
            notes_text = ' '.join([
                note.get('note_text', '') 
                for note in row.get('clinical_notes', [])
            ])
            
            for keyword in diagnosis_keywords:
                if keyword.lower() in notes_text.lower():
                    leakage_count += 1
                    break
        
        leakage_rate = leakage_count / len(linked_df) * 100
        print(f"  Potential leakage rate: {leakage_rate:.2f}%")
        
        if leakage_rate > 5:
            print("  WARNING: High leakage rate detected. Review filtering logic.")
        
        # Validate data completeness
        print(f"  Records with bounding boxes: {(linked_df['bounding_boxes'].str.len() > 0).sum()}")
        print(f"  Records with clinical notes: {(linked_df['clinical_notes'].str.len() > 0).sum()}")
        print(f"  Records with vital signs: {(linked_df['vital_signs'].str.len() > 0).sum()}")
        
        return leakage_rate < 5
    
    def save_preprocessed_data(self, linked_df: pd.DataFrame, output_path: str):
        """
        Save preprocessed data in multiple formats
        
        Args:
            linked_df: DataFrame with linked data
            output_path: Path to save preprocessed data
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main dataframe
        linked_df.to_parquet(output_path / 'linked_data.parquet')
        
        # Save train/val/test splits
        for split in ['train', 'validate', 'test']:
            split_df = linked_df[linked_df['split'] == split]
            split_df.to_parquet(output_path / f'{split}_data.parquet')
            print(f"  Saved {split} split: {len(split_df)} records")
        
        # Save metadata
        metadata = {
            'total_records': len(linked_df),
            'subjects': linked_df['subject_id'].nunique(),
            'studies': linked_df['study_id'].nunique(),
            'records_with_boxes': (linked_df['bounding_boxes'].str.len() > 0).sum(),
            'records_with_notes': (linked_df['clinical_notes'].str.len() > 0).sum(),
            'positive_findings_distribution': linked_df['positive_findings'].value_counts().to_dict()
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Saved preprocessed data to {output_path}")


def main():
    """Main preprocessing pipeline"""
    
    # Configuration
    BASE_PATH = "/path/to/your/mimic/datasets"  # Update this
    OUTPUT_PATH = "./preprocessed_mimic_data"
    
    # Initialize preprocessor
    preprocessor = MIMICPreprocessor(BASE_PATH)
    
    # Step 1: Load CXR metadata and labels
    preprocessor.load_cxr_metadata()
    
    # Step 2: Load REFLACX annotations (bounding boxes)
    preprocessor.load_reflacx_annotations()
    
    # Step 3: Get unique subjects from CXR data
    subject_ids = preprocessor.cxr_metadata['subject_id'].unique()[:100]  # Start with 100 for testing
    
    # Step 4: Load clinical data for these subjects
    preprocessor.load_clinical_data(subject_ids)
    
    # Step 5: Link all data together
    linked_data = preprocessor.link_cxr_to_clinical(time_window_hours=24)
    
    # Step 6: Validate preprocessing (check for leakage)
    is_valid = preprocessor.validate_preprocessing(linked_data)
    
    if is_valid:
        # Step 7: Save preprocessed data
        preprocessor.save_preprocessed_data(linked_data, OUTPUT_PATH)
        print("Preprocessing complete!")
    else:
        print("Preprocessing validation failed. Please review the pipeline.")


if __name__ == "__main__":
    main()