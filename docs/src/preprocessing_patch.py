"""
Patch for phase1_preprocess_streaming.py to properly attach CheXpert labels

This patch shows the exact modifications needed to fix the label attachment issue
in your existing preprocessing pipeline.
"""

# ============================================================================
# MODIFICATION 1: Add to imports section (around line 30)
# ============================================================================

# Add after existing imports:
from fix_chexpert_labels import (
    CheXpertLabelProcessor, 
    enhance_record_with_labels,
    CHEXPERT_LABELS
)

# ============================================================================
# MODIFICATION 2: Update DataConfig class (around line 70)
# ============================================================================

@dataclass
class DataConfig:
    """Configuration for data preprocessing"""
    # ... existing fields ...
    
    # ADD THESE NEW FIELDS:
    # CheXpert label settings
    chexpert_labels_path: str = "mimic-cxr-2.0.0-chexpert.csv.gz"
    label_format_type: str = "multi_label"  # Options: multi_label, multi_class, positive_only, full
    handle_uncertain_as: str = "negative"  # How to handle uncertain (-1) labels: negative, positive, or keep

# ============================================================================
# MODIFICATION 3: Update DataPreprocessor.__init__ (around line 800)
# ============================================================================

class DataPreprocessor:
    def __init__(self, config: DataConfig):
        self.config = config
        self.use_gcs = config.use_gcs
        
        # ... existing initialization code ...
        
        # ADD THIS: Initialize CheXpert label processor
        self.label_processor = CheXpertLabelProcessor(
            gcs_helper=self.gcs_helper if self.use_gcs else None,
            use_gcs=self.use_gcs
        )
        
        # Load CheXpert labels
        if self.use_gcs:
            # For GCS, construct the full path
            chexpert_path = self.config.chexpert_labels_path
            # If using the PhysioNet requester-pays bucket
            if hasattr(self.gcs_helper, 'cxr_bucket'):
                logger.info(f"Loading CheXpert labels from CXR bucket: {chexpert_path}")
            else:
                chexpert_path = f"data/{chexpert_path}"
        else:
            # For local files
            chexpert_path = Path(self.config.mimic_cxr_path) / self.config.chexpert_labels_path
        
        logger.info(f"Loading CheXpert labels from: {chexpert_path}")
        self.label_processor.load_chexpert_labels(str(chexpert_path))
        logger.info("CheXpert labels loaded successfully")

# ============================================================================  
# MODIFICATION 4: Replace process_record method (around line 900)
# ============================================================================

def process_record(self, row) -> Optional[Dict]:
    """
    Process a single record with FIXED label attachment
    
    Args:
        row: Row from metadata containing study information
    
    Returns:
        Processed record or None if error
    """
    try:
        # Load and process image
        image_data = self.process_image(row)
        if image_data is None:
            return None
        
        # Process text
        clinical_text = self.create_clinical_text(row)
        text_data = self.process_text(clinical_text)
        
        # Extract clinical features
        clinical_features = self.extract_clinical_features(row)
        
        # Retrieve knowledge (if configured)
        retrieved_knowledge = None
        if self.config.use_rag:
            retrieved_knowledge = self.retrieve_knowledge(clinical_text)
        
        # Create base record
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
            
            # Basic metadata labels
            'labels': {
                'view_position': row.get('ViewPosition', 'UNKNOWN'),
            }
        }
        
        # CRITICAL FIX: Add CheXpert disease labels
        if self.label_processor:
            # Get formatted labels for this study
            study_labels = self.label_processor.format_labels_for_training(
                record['study_id'],
                format_type=self.config.label_format_type
            )
            
            # Merge disease labels into record
            record['labels'].update(study_labels)
            
            # Log if no labels found (for debugging)
            if not study_labels.get('disease_labels'):
                logger.debug(f"No positive findings for study {record['study_id']}")
        else:
            # Fallback: add empty structure so downstream code doesn't break
            logger.warning(f"No label processor available for study {record['study_id']}")
            record['labels'].update({
                'disease_labels': [],
                'disease_binary': [0] * len(CHEXPERT_LABELS),
                'label_array': np.zeros(len(CHEXPERT_LABELS), dtype=np.float32)
            })
        
        return record
        
    except Exception as e:
        logger.error(f"Error processing record {row.get('dicom_id', 'unknown')}: {e}")
        return None

# ============================================================================
# MODIFICATION 5: Update the main preprocessing loop to track label statistics
# ============================================================================

def process_batch_records_parallel(self, records_batch: pd.DataFrame, 
                                  batch_num: int) -> Optional[List[Dict]]:
    """
    Process a batch of records in parallel with label statistics
    """
    try:
        logger.info(f"Processing batch {batch_num} with {len(records_batch)} records...")
        
        processed_records = []
        
        # ADD: Track label statistics
        label_stats = {label: 0 for label in CHEXPERT_LABELS}
        records_with_findings = 0
        
        # Process records with progress bar
        for _, row in tqdm(records_batch.iterrows(), 
                          total=len(records_batch),
                          desc=f"Batch {batch_num}"):
            record = self.process_record(row)
            if record:
                processed_records.append(record)
                
                # ADD: Update label statistics
                if 'disease_labels' in record['labels']:
                    if record['labels']['disease_labels']:
                        records_with_findings += 1
                        for label in record['labels']['disease_labels']:
                            if label in label_stats:
                                label_stats[label] += 1
        
        # ADD: Log label statistics for this batch
        logger.info(f"Batch {batch_num} label statistics:")
        logger.info(f"  Records with findings: {records_with_findings}/{len(processed_records)}")
        logger.info(f"  Label distribution:")
        for label, count in label_stats.items():
            if count > 0:
                logger.info(f"    {label}: {count} ({100*count/len(processed_records):.1f}%)")
        
        return processed_records
        
    except Exception as e:
        logger.error(f"Error processing batch {batch_num}: {e}")
        return None

# ============================================================================
# MODIFICATION 6: Update command-line arguments in main()
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 1 Data Preprocessing with CheXpert Labels')
    
    # ... existing arguments ...
    
    # ADD THESE NEW ARGUMENTS:
    # CheXpert label arguments
    parser.add_argument('--chexpert-labels-path', type=str,
                       default='mimic-cxr-2.0.0-chexpert.csv.gz',
                       help='Path to CheXpert labels CSV file')
    parser.add_argument('--label-format', type=str,
                       default='multi_label',
                       choices=['multi_label', 'multi_class', 'positive_only', 'full'],
                       help='Format for disease labels')
    parser.add_argument('--handle-uncertain', type=str,
                       default='negative',
                       choices=['negative', 'positive', 'keep'],
                       help='How to handle uncertain (-1) labels')
    
    args = parser.parse_args()
    
    # Create config with label settings
    config = DataConfig(
        mimic_cxr_path=args.mimic_cxr_path,
        mimic_iv_path=args.mimic_iv_path,
        mimic_ed_path=args.mimic_ed_path,
        output_path=args.output_path,
        
        # ... other config fields ...
        
        # ADD: Label configuration
        chexpert_labels_path=args.chexpert_labels_path,
        label_format_type=args.label_format,
        handle_uncertain_as=args.handle_uncertain
    )
    
    # ... rest of main() ...

# ============================================================================
# VALIDATION FUNCTION: Add this to verify labels are correctly attached
# ============================================================================

def validate_preprocessed_data(data_path: str, split: str = 'train'):
    """
    Validate that CheXpert labels are properly attached to preprocessed data
    
    Args:
        data_path: Path to preprocessed data
        split: Which split to validate ('train', 'val', or 'test')
    """
    import pickle
    
    # Load preprocessed data
    file_path = f"{data_path}/{split}_data.pkl"
    logger.info(f"Loading {file_path} for validation...")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    logger.info(f"Loaded {len(data)} records")
    
    # Check label statistics
    records_with_labels = 0
    records_with_findings = 0
    label_counts = {label: 0 for label in CHEXPERT_LABELS}
    missing_labels = 0
    
    for record in data:
        if 'labels' in record:
            if 'disease_labels' in record['labels']:
                records_with_labels += 1
                if record['labels']['disease_labels']:
                    records_with_findings += 1
                    for label in record['labels']['disease_labels']:
                        if label in label_counts:
                            label_counts[label] += 1
            else:
                missing_labels += 1
        else:
            missing_labels += 1
    
    # Report findings
    logger.info("=" * 60)
    logger.info(f"VALIDATION RESULTS for {split} split:")
    logger.info("=" * 60)
    logger.info(f"Total records: {len(data)}")
    logger.info(f"Records with label structure: {records_with_labels} ({100*records_with_labels/len(data):.1f}%)")
    logger.info(f"Records with positive findings: {records_with_findings} ({100*records_with_findings/len(data):.1f}%)")
    logger.info(f"Records missing labels: {missing_labels}")
    
    logger.info("\nLabel distribution (positive findings only):")
    for label in CHEXPERT_LABELS:
        count = label_counts[label]
        if count > 0:
            logger.info(f"  {label}: {count} ({100*count/len(data):.1f}%)")
    
    # Check a sample record
    if data:
        logger.info("\nSample record structure:")
        sample = data[0]
        logger.info(f"  Keys: {list(sample.keys())}")
        if 'labels' in sample:
            logger.info(f"  Label keys: {list(sample['labels'].keys())}")
            if 'disease_labels' in sample['labels']:
                logger.info(f"  Disease labels: {sample['labels']['disease_labels']}")
                if 'label_array' in sample['labels']:
                    logger.info(f"  Label array shape: {sample['labels']['label_array'].shape}")
    
    return records_with_findings > 0  # Return True if labels are properly attached

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
To apply this patch to your existing pipeline:

1. Save the fix_chexpert_labels.py file in your src/ directory

2. Apply the modifications shown above to your phase1_preprocess_streaming.py file

3. Run preprocessing with label attachment:
   python phase1_preprocess_streaming.py \
       --mimic-cxr-path /path/to/mimic-cxr \
       --chexpert-labels-path mimic-cxr-2.0.0-chexpert.csv.gz \
       --label-format multi_label \
       --output-path /path/to/output

4. Validate that labels are attached:
   python -c "from phase1_preprocess_streaming_patched import validate_preprocessed_data; validate_preprocessed_data('/path/to/output', 'train')"

Expected output after fix:
- Each record will have properly populated 'disease_labels' 
- Binary label arrays for model training
- Positive finding counts that match MIMIC-CXR statistics (~50-60% with findings)

Common MIMIC-CXR label prevalences for reference:
- Support Devices: ~50%
- Lung Opacity: ~40%
- Pleural Effusion: ~30%
- Atelectasis: ~30%
- Cardiomegaly: ~25%
- No Finding: ~15-20%
"""
