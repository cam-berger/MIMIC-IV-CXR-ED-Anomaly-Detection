# MIMIC-CXR Label Attachment Issue - Analysis and Solution

## Executive Summary

Your preprocessed MIMIC-CXR data has **no disease labels attached**. The CheXpert labels (the 14 disease findings like Pneumonia, Cardiomegaly, etc.) are not being loaded or attached during preprocessing, resulting in empty `disease_labels` arrays for all samples. This makes supervised training impossible.

## The Problem

### Current Data Structure (BROKEN)
```python
{
    'subject_id': 10000032,
    'study_id': 56699142,
    'image_tensor': tensor([...]),
    'labels': {
        'disease_labels': [],        # ❌ EMPTY - Should have disease names
        'bbox_coordinates': [],       # Not used for classification
        'severity_scores': []         # Not used for classification
    }
}
```

### Expected Data Structure (FIXED)
```python
{
    'subject_id': 10000032,
    'study_id': 56699142, 
    'image_tensor': tensor([...]),
    'labels': {
        'view_position': 'PA',
        'disease_labels': ['Cardiomegaly', 'Pleural Effusion'],  # ✅ Positive findings
        'disease_binary': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Binary array
        'label_array': array([0., 1., 0., ...], dtype=float32)  # For loss computation
    }
}
```

## Root Cause Analysis

### 1. Label Loading Issue
- **phase1_stay_identification.py**: Loads CheXpert labels correctly but only extracts positive findings
- **phase1_preprocess_streaming.py**: Creates records but doesn't include the CheXpert labels
- **Result**: Labels exist in intermediate files but aren't attached to final training data

### 2. Missing Integration
The preprocessing pipeline has a placeholder comment:
```python
'labels': {
    'view_position': row.get('ViewPosition', 'UNKNOWN'),
    # Add more labels as needed  ← Never implemented!
}
```

## The Solution

### Step 1: Run Diagnostic
First, verify the issue exists:
```bash
python diagnose_labels.py /path/to/your/preprocessed/train_data.pkl
```

Expected output will show:
- ❌ CRITICAL: No records have disease labels attached!
- Records with positive findings: 0 (0.0%)

### Step 2: Apply the Fix

The fix involves three components:

1. **fix_chexpert_labels.py** - Core label processing module
2. **preprocessing_patch.py** - Shows exact modifications needed
3. **diagnose_labels.py** - Verification tool

### Step 3: Modify Your Pipeline

Add to your `phase1_preprocess_streaming.py`:

```python
# 1. Import the label processor
from fix_chexpert_labels import CheXpertLabelProcessor, CHEXPERT_LABELS

# 2. In DataPreprocessor.__init__, add:
self.label_processor = CheXpertLabelProcessor(
    gcs_helper=self.gcs_helper if self.use_gcs else None,
    use_gcs=self.use_gcs
)
self.label_processor.load_chexpert_labels('mimic-cxr-2.0.0-chexpert.csv.gz')

# 3. In process_record(), replace the labels section with:
# Get CheXpert labels for this study
study_labels = self.label_processor.format_labels_for_training(
    record['study_id'],
    format_type='multi_label'
)

# Add to record
record['labels'].update(study_labels)
```

### Step 4: Rerun Preprocessing

```bash
python phase1_preprocess_streaming.py \
    --gcs-bucket bergermimiciv \
    --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
    --chexpert-labels-path mimic-cxr-2.0.0-chexpert.csv.gz \
    --label-format multi_label \
    --output-path processed/phase1_with_labels
```

### Step 5: Verify Fix

```bash
python diagnose_labels.py processed/phase1_with_labels/train_data.pkl
```

Expected output after fix:
- ✅ Labels appear to be properly attached
- Records with positive findings: ~50-60%
- Distribution matching MIMIC-CXR statistics

## Label Statistics (Expected)

After fixing, you should see approximately:

| Finding | Expected Prevalence | Your Current | Status |
|---------|-------------------|--------------|---------|
| Support Devices | 45-55% | 0% | ❌ Missing |
| Lung Opacity | 35-45% | 0% | ❌ Missing |
| Pleural Effusion | 25-35% | 0% | ❌ Missing |
| Atelectasis | 25-35% | 0% | ❌ Missing |
| Cardiomegaly | 20-30% | 0% | ❌ Missing |
| No Finding | 15-25% | 0% | ❌ Missing |

## Label Formats Supported

The fix supports multiple label formats:

### 1. Multi-Label (Recommended)
```python
format_type='multi_label'
# Returns binary array [0,1,0,1,...] for 14 diseases
# Use with BCEWithLogitsLoss
```

### 2. Multi-Class with Uncertainty
```python
format_type='multi_class'  
# Returns [-1,0,1,...] preserving uncertain labels
# Use for uncertainty-aware models
```

### 3. Positive Only (Your Current Approach)
```python
format_type='positive_only'
# Returns list of positive disease names only
# Backward compatible but incomplete
```

## Training Considerations

### Class Imbalance
The labels are highly imbalanced. Use the provided weight calculator:
```python
from fix_chexpert_labels import get_label_weights
weights = get_label_weights(label_processor, study_ids)
# Use in loss: nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights))
```

### Handling Uncertain Labels
CheXpert includes uncertain (-1) labels. Options:
1. Treat as negative (default)
2. Treat as positive (aggressive)
3. Use uncertainty-aware loss
4. Ignore uncertain samples

### Multi-Label vs Multi-Class
- **Multi-Label**: Each disease is independent binary classification
- **Multi-Class**: Would be wrong for CXR (can have multiple diseases)

## Quick Validation Check

Run this to quickly check if labels are attached:
```python
import pickle
with open('train_data.pkl', 'rb') as f:
    data = pickle.load(f)
    
sample = data[0]
print("Labels in first record:", sample['labels'])
print("Disease labels:", sample['labels'].get('disease_labels', 'MISSING!'))
```

## Next Steps

1. ✅ Apply the fix to your preprocessing pipeline
2. ✅ Rerun preprocessing with label attachment
3. ✅ Verify labels are properly attached
4. ✅ Proceed with model training
5. ✅ Use class weights for imbalanced labels

## Files Provided

1. **fix_chexpert_labels.py** - Complete label processing module
2. **preprocessing_patch.py** - Detailed integration instructions
3. **diagnose_labels.py** - Diagnostic tool to verify labels
4. **label_fix_summary.md** - This documentation

## Need Help?

Common issues:
- Can't find chexpert.csv.gz → Check GCS bucket paths
- Study IDs don't match → Verify MIMIC-CXR version consistency
- Memory issues → Process in smaller batches
- GCS authentication → Ensure requester-pays is configured

The critical point is that **without these labels, you cannot train a disease classification model**. The images alone aren't sufficient - you need the ground truth labels from the radiologist reports that CheXpert provides.
