"""
Fix for attaching CheXpert labels during Phase 1 preprocessing

This module provides functions to properly load and attach CheXpert labels 
to each record during the preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from io import BytesIO

logger = logging.getLogger(__name__)

# CheXpert label columns as defined in MIMIC-CXR-JPG
CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]

class CheXpertLabelProcessor:
    """
    Handles loading and processing of CheXpert labels from MIMIC-CXR-JPG
    """
    
    def __init__(self, gcs_helper=None, use_gcs=False):
        """
        Initialize label processor
        
        Args:
            gcs_helper: GCS helper object if using Google Cloud Storage
            use_gcs: Whether to use GCS for loading labels
        """
        self.gcs_helper = gcs_helper
        self.use_gcs = use_gcs
        self.chexpert_df = None
        self.label_cache = {}  # Cache for faster lookups
        
    def load_chexpert_labels(self, chexpert_path: str = None) -> pd.DataFrame:
        """
        Load CheXpert labels from CSV file
        
        Args:
            chexpert_path: Path to mimic-cxr-2.0.0-chexpert.csv.gz
                          If using GCS, this should be the blob path
        
        Returns:
            DataFrame with CheXpert labels
        """
        if self.use_gcs and self.gcs_helper:
            logger.info(f"Loading CheXpert labels from GCS: {chexpert_path}")
            # Assuming your GCS helper has a method to read CSV
            self.chexpert_df = self.gcs_helper.read_csv(
                chexpert_path, 
                from_cxr_bucket=True, 
                compression='gzip'
            )
        else:
            logger.info(f"Loading CheXpert labels from local path: {chexpert_path}")
            self.chexpert_df = pd.read_csv(chexpert_path, compression='gzip')
        
        logger.info(f"Loaded {len(self.chexpert_df)} CheXpert label records")
        
        # Build cache for faster lookups
        self._build_label_cache()
        
        return self.chexpert_df
    
    def _build_label_cache(self):
        """Build a cache mapping study_id to labels for faster lookup"""
        self.label_cache = {}
        for _, row in self.chexpert_df.iterrows():
            study_id = row['study_id']
            labels = {}
            for label in CHEXPERT_LABELS:
                if label in row:
                    # Store the actual value (1.0, 0.0, -1.0, or NaN)
                    labels[label] = row[label] if pd.notna(row[label]) else None
            self.label_cache[study_id] = labels
    
    def get_labels_for_study(self, study_id: int) -> Dict:
        """
        Get CheXpert labels for a specific study
        
        Args:
            study_id: Study ID to get labels for
        
        Returns:
            Dictionary with label values:
            - 1.0: Positive (present)
            - 0.0: Negative (absent) 
            - -1.0: Uncertain
            - None: Not mentioned
        """
        if study_id in self.label_cache:
            return self.label_cache[study_id]
        
        # Fallback to DataFrame lookup if not in cache
        study_labels = self.chexpert_df[self.chexpert_df['study_id'] == study_id]
        
        if study_labels.empty:
            logger.warning(f"No CheXpert labels found for study_id: {study_id}")
            return {label: None for label in CHEXPERT_LABELS}
        
        labels = {}
        row = study_labels.iloc[0]
        for label in CHEXPERT_LABELS:
            if label in row:
                labels[label] = row[label] if pd.notna(row[label]) else None
            else:
                labels[label] = None
                
        return labels
    
    def format_labels_for_training(self, study_id: int, 
                                  format_type: str = 'multi_label') -> Dict:
        """
        Format CheXpert labels for model training
        
        Args:
            study_id: Study ID to get labels for
            format_type: How to format labels
                - 'multi_label': Binary multi-label format (0 or 1 for each disease)
                - 'multi_class': Include uncertainty (-1, 0, 1)
                - 'positive_only': List of positive findings only
                - 'full': Complete label dictionary with all values
        
        Returns:
            Formatted label dictionary
        """
        raw_labels = self.get_labels_for_study(study_id)
        
        if format_type == 'multi_label':
            # Binary classification: treat positive as 1, everything else as 0
            formatted = {
                'disease_labels': [],
                'disease_binary': []
            }
            for label in CHEXPERT_LABELS:
                value = raw_labels.get(label)
                if value == 1.0:
                    formatted['disease_labels'].append(label)
                    formatted['disease_binary'].append(1)
                else:
                    formatted['disease_binary'].append(0)
            
            # Also return as numpy array for easier model training
            formatted['label_array'] = np.array(formatted['disease_binary'], dtype=np.float32)
            
        elif format_type == 'multi_class':
            # Include uncertainty: -1, 0, 1, or 0 for not mentioned
            formatted = {
                'disease_labels': CHEXPERT_LABELS,
                'disease_values': []
            }
            for label in CHEXPERT_LABELS:
                value = raw_labels.get(label)
                if value is None:
                    formatted['disease_values'].append(0)  # Treat not mentioned as negative
                else:
                    formatted['disease_values'].append(value)
            
            formatted['label_array'] = np.array(formatted['disease_values'], dtype=np.float32)
            
        elif format_type == 'positive_only':
            # List only positive findings (backward compatible with your current approach)
            formatted = {
                'disease_labels': [
                    label for label in CHEXPERT_LABELS 
                    if raw_labels.get(label) == 1.0
                ]
            }
            
        else:  # 'full'
            formatted = {
                'disease_labels': CHEXPERT_LABELS,
                'disease_values': raw_labels
            }
        
        return formatted


def enhance_record_with_labels(record: Dict, 
                              label_processor: CheXpertLabelProcessor,
                              format_type: str = 'multi_label') -> Dict:
    """
    Enhance a preprocessing record with CheXpert labels
    
    Args:
        record: The preprocessing record (must have 'study_id')
        label_processor: CheXpertLabelProcessor instance with loaded labels
        format_type: How to format the labels
    
    Returns:
        Enhanced record with proper labels
    """
    study_id = record.get('study_id')
    
    if study_id is None:
        logger.error(f"Record missing study_id, cannot attach labels")
        # Return with empty labels
        record['labels'] = {
            'disease_labels': [],
            'disease_binary': [0] * len(CHEXPERT_LABELS),
            'label_array': np.zeros(len(CHEXPERT_LABELS), dtype=np.float32)
        }
        return record
    
    # Get formatted labels
    labels = label_processor.format_labels_for_training(study_id, format_type)
    
    # Update the record's label field
    if 'labels' not in record:
        record['labels'] = {}
    
    # Merge with existing labels (like view_position)
    record['labels'].update(labels)
    
    return record


# Integration function for your preprocessing pipeline
def integrate_labels_into_preprocessing(preprocessor_instance, 
                                       chexpert_path: str,
                                       format_type: str = 'multi_label'):
    """
    Integrate this label processor into your existing preprocessing pipeline
    
    This should be called in your DataPreprocessor.__init__ or setup method
    
    Args:
        preprocessor_instance: Your DataPreprocessor instance
        chexpert_path: Path to CheXpert labels CSV
        format_type: Label format type
    """
    # Create label processor
    label_processor = CheXpertLabelProcessor(
        gcs_helper=getattr(preprocessor_instance, 'gcs_helper', None),
        use_gcs=getattr(preprocessor_instance, 'use_gcs', False)
    )
    
    # Load labels
    label_processor.load_chexpert_labels(chexpert_path)
    
    # Store processor in preprocessor instance
    preprocessor_instance.label_processor = label_processor
    preprocessor_instance.label_format_type = format_type
    
    logger.info(f"Label processor integrated with format: {format_type}")


# Example modification to your process_record method
def enhanced_process_record(self, row, image_data, text_data, clinical_features, retrieved_knowledge):
    """
    This should REPLACE the relevant part of your DataPreprocessor.process_record method
    
    The key change is properly attaching CheXpert labels to each record
    """
    try:
        # Original record creation (your existing code)
        record = {
            'subject_id': int(row['subject_id']),
            'study_id': int(row['study_id']),
            'dicom_id': row.get('dicom_id', 'unknown'),
            
            # Image data
            'image_data': image_data,
            
            # Text data
            'text_tokens': text_data,
            'clinical_features': clinical_features,
            
            # Knowledge
            'retrieved_knowledge': retrieved_knowledge,
            
            # Labels - START OF FIX
            'labels': {
                'view_position': row.get('ViewPosition', 'UNKNOWN'),
            }
        }
        
        # CRITICAL FIX: Add CheXpert labels
        if hasattr(self, 'label_processor') and self.label_processor:
            record = enhance_record_with_labels(
                record, 
                self.label_processor, 
                self.label_format_type
            )
        else:
            logger.warning("No label processor found, labels will be empty!")
            # Add empty structure so downstream code doesn't break
            record['labels'].update({
                'disease_labels': [],
                'disease_binary': [0] * len(CHEXPERT_LABELS),
                'label_array': np.zeros(len(CHEXPERT_LABELS), dtype=np.float32)
            })
        
        return record
        
    except Exception as e:
        logger.error(f"Error processing record {row.get('dicom_id', 'unknown')}: {e}")
        return None


# Training utility function
def get_label_weights(label_processor: CheXpertLabelProcessor, 
                     study_ids: List[int]) -> np.ndarray:
    """
    Calculate class weights for handling imbalanced labels
    
    Args:
        label_processor: CheXpertLabelProcessor with loaded labels
        study_ids: List of study IDs in your dataset
    
    Returns:
        Array of weights for each label class
    """
    # Count positive instances for each label
    label_counts = np.zeros(len(CHEXPERT_LABELS))
    total_samples = len(study_ids)
    
    for study_id in study_ids:
        labels = label_processor.format_labels_for_training(
            study_id, 
            format_type='multi_label'
        )
        label_counts += labels['label_array']
    
    # Calculate weights (inverse frequency)
    # Avoid division by zero
    label_counts = np.maximum(label_counts, 1)
    weights = total_samples / (len(CHEXPERT_LABELS) * label_counts)
    
    # Normalize weights
    weights = weights / weights.mean()
    
    logger.info("Label distribution and weights:")
    for i, label in enumerate(CHEXPERT_LABELS):
        logger.info(f"  {label}: {label_counts[i]}/{total_samples} "
                   f"({100*label_counts[i]/total_samples:.1f}%), weight={weights[i]:.2f}")
    
    return weights


if __name__ == "__main__":
    # Example usage
    print("CheXpert Label Processor Module")
    print("=" * 60)
    print("\nThis module provides:")
    print("1. CheXpertLabelProcessor class for loading/processing labels")
    print("2. enhance_record_with_labels() for adding labels to records")  
    print("3. integrate_labels_into_preprocessing() for pipeline integration")
    print("4. get_label_weights() for handling class imbalance")
    print("\nLabel formats supported:")
    print("- multi_label: Binary classification (0/1) for each disease")
    print("- multi_class: Includes uncertainty (-1/0/1)")
    print("- positive_only: List of positive findings only")
    print("- full: Complete raw label values")
    print("\nIntegration example:")
    print("-" * 40)
    print("""
# In your DataPreprocessor.__init__ method:
from fix_chexpert_labels import integrate_labels_into_preprocessing

# After initializing your preprocessor
integrate_labels_into_preprocessing(
    self, 
    chexpert_path='mimic-cxr-2.0.0-chexpert.csv.gz',
    format_type='multi_label'
)

# Then in process_record, the labels will be automatically attached
    """)
