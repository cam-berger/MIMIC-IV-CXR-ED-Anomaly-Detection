# Phase 1 Preprocessing Pipeline - Complete Guide

## Overview

The integrated Phase 1 preprocessing pipeline now includes **full CheXpert disease label support**. This fix addresses the critical issue where disease labels were not being attached to preprocessed data, making supervised training impossible.

## What's New

✅ **CheXpert label loading** - Automatically loads and attaches 14 disease labels to each record
✅ **Multiple label formats** - Supports multi_label, multi_class, positive_only, and full formats
✅ **Diagnostic tools** - Verify labels are properly attached before training
✅ **Integrated pipeline** - All functionality in one streamlined codebase

## Quick Start

### 1. Basic Local Usage

```bash
python src/phase1_preprocess_streaming.py \
    --mimic-cxr-path ~/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-cxr-jpg-2.1.0 \
    --mimic-iv-path ~/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimiciv/3.1 \
    --mimic-ed-path ~/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-iv-ed/2.2 \
    --output-path processed/phase1_with_labels \
    --chexpert-labels-path mimic-cxr-2.0.0-chexpert.csv.gz \
    --label-format multi_label \
    --batch-size 100 \
    --num-workers 4
```

### 2. Google Cloud Storage Usage

```bash
python src/phase1_preprocess_streaming.py \
    --gcs-bucket bergermimiciv \
    --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
    --gcs-project-id your-project-id \
    --output-path processed/phase1_with_labels \
    --chexpert-labels-path mimic-cxr-2.0.0-chexpert.csv.gz \
    --label-format multi_label \
    --batch-size 100 \
    --num-workers 4
```

### 3. Verify Labels are Attached

After preprocessing, verify that labels are properly attached:

```bash
python src/diagnose_labels.py processed/phase1_with_labels/train_data.pkl
```

**Expected Output:**
```
✓ Labels appear to be properly attached
Records with positive findings: ~50-60%
Support Devices: ~50%
Lung Opacity: ~40%
Pleural Effusion: ~30%
```

## Label Formats

### Multi-Label (Recommended)

Binary classification for each disease independently. Best for multi-label classification.

```bash
--label-format multi_label
```

**Output structure:**
```python
{
    'disease_labels': ['Cardiomegaly', 'Pleural Effusion'],  # Positive findings
    'disease_binary': [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # Binary array
    'label_array': array([0., 1., 0., ...], dtype=float32)  # For loss computation
}
```

**Use with:**
```python
criterion = nn.BCEWithLogitsLoss()
```

### Multi-Class (With Uncertainty)

Preserves uncertain labels as -1. Useful for uncertainty-aware models.

```bash
--label-format multi_class
```

**Output structure:**
```python
{
    'disease_labels': ['Atelectasis', 'Cardiomegaly', ...],  # All 14 labels
    'disease_values': [-1, 1, 0, 1, 0, ...],  # -1=uncertain, 0=negative, 1=positive
    'label_array': array([-1., 1., 0., ...], dtype=float32)
}
```

### Positive Only

Legacy format - only positive findings listed.

```bash
--label-format positive_only
```

**Output structure:**
```python
{
    'disease_labels': ['Cardiomegaly', 'Pleural Effusion']  # Only positives
}
```

## Command-Line Arguments

### Required Paths

- `--mimic-cxr-path`: Path to MIMIC-CXR-JPG dataset
- `--mimic-iv-path`: Path to MIMIC-IV dataset
- `--mimic-ed-path`: Path to MIMIC-IV-ED dataset
- `--output-path`: Where to save preprocessed data

### GCS Arguments

- `--gcs-bucket`: Your GCS bucket for MIMIC-IV/ED data
- `--gcs-cxr-bucket`: GCS bucket for MIMIC-CXR images (PhysioNet)
- `--gcs-project-id`: GCP project ID (required for requester-pays)
- `--output-gcs-bucket`: Output bucket (defaults to gcs-bucket)

### Label Arguments

- `--chexpert-labels-path`: Path to CheXpert labels file (default: `mimic-cxr-2.0.0-chexpert.csv.gz`)
- `--label-format`: Label format - `multi_label` (default), `multi_class`, `positive_only`, `full`
- `--handle-uncertain`: How to handle uncertain (-1) labels - `negative` (default), `positive`, `keep`

### Processing Arguments

- `--batch-size`: Records per batch (default: 100)
- `--num-workers`: Parallel workers (default: 4)
- `--image-size`: Image size for BiomedCLIP (default: 518)
- `--max-text-length`: Max text tokens (default: 8192)

### Performance Arguments

- `--skip-to-combine`: Skip batch processing, combine existing batches
- `--skip-final-combine`: Keep data as chunks instead of single files
- `--max-batches`: Limit number of batches (for testing)
- `--create-small-samples`: Create small sample datasets
- `--small-sample-size`: Size of sample datasets (default: 100)

## Training with Labels

### Example: Binary Multi-Label Classification

```python
import torch
import torch.nn as nn
import pickle

# Load preprocessed data
with open('processed/phase1_with_labels/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)

# Access labels in your training loop
for sample in train_data:
    image = sample['image']  # Image tensor
    labels = sample['labels']['label_array']  # Shape: (14,)

    # Forward pass
    outputs = model(image)  # Shape: (14,)

    # Binary cross-entropy loss
    loss = criterion(outputs, torch.tensor(labels))

# Use class weights for imbalanced data
from src.fix_chexpert_labels import get_label_weights, CheXpertLabelProcessor

label_processor = CheXpertLabelProcessor()
label_processor.load_chexpert_labels('path/to/chexpert.csv.gz')

study_ids = [sample['study_id'] for sample in train_data]
weights = get_label_weights(label_processor, study_ids)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights))
```

## CheXpert Label Details

### 14 Disease Findings

1. **Atelectasis** - Lung collapse
2. **Cardiomegaly** - Enlarged heart
3. **Consolidation** - Lung solidification
4. **Edema** - Fluid accumulation
5. **Enlarged Cardiomediastinum** - Widened mediastinum
6. **Fracture** - Bone fracture
7. **Lung Lesion** - Abnormal lung tissue
8. **Lung Opacity** - Abnormal opacification
9. **No Finding** - No abnormalities detected
10. **Pleural Effusion** - Fluid around lungs
11. **Pleural Other** - Other pleural abnormalities
12. **Pneumonia** - Lung infection
13. **Pneumothorax** - Collapsed lung
14. **Support Devices** - Medical devices present

### Expected Prevalence

| Finding | Prevalence | Count (in 200k train) |
|---------|------------|----------------------|
| Support Devices | 45-55% | ~100k |
| Lung Opacity | 35-45% | ~80k |
| Pleural Effusion | 25-35% | ~60k |
| Atelectasis | 25-35% | ~60k |
| Cardiomegaly | 20-30% | ~50k |
| No Finding | 15-25% | ~40k |

## Handling Uncertain Labels

CheXpert includes uncertain (-1) labels when the labeler couldn't definitively determine presence/absence.

### Options:

**1. Treat as Negative (Default)**
```bash
--handle-uncertain negative
```
Conservative approach - only definite positives count as positive.

**2. Treat as Positive**
```bash
--handle-uncertain positive
```
Aggressive approach - maximizes sensitivity.

**3. Keep Uncertainty**
```bash
--label-format multi_class --handle-uncertain keep
```
Preserve uncertainty for uncertainty-aware loss functions.

## Common Issues

### Issue: "No CheXpert labels found"

**Solution:**
```bash
# Verify the file exists
ls -lh ~/path/to/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv.gz

# Use absolute path
--chexpert-labels-path /full/path/to/mimic-cxr-2.0.0-chexpert.csv.gz
```

### Issue: "Study IDs don't match"

**Solution:** Ensure MIMIC-CXR versions are consistent:
- CXR images: `mimic-cxr-jpg-2.1.0`
- CheXpert labels: `mimic-cxr-2.0.0-chexpert.csv.gz` (compatible)

### Issue: Memory errors during preprocessing

**Solution:**
```bash
# Reduce batch size
--batch-size 50

# Keep data as chunks
--skip-final-combine
```

### Issue: GCS authentication errors

**Solution:**
```bash
# Set up GCS authentication
gcloud auth application-default login

# Verify requester-pays setup
--gcs-project-id your-project-id
```

## File Structure

```
src/
├── __init__.py
├── phase1_preprocess_streaming.py  # Main preprocessing pipeline
├── fix_chexpert_labels.py          # Label loading and processing
└── diagnose_labels.py              # Label verification tool

docs/
├── PHASE1_USAGE.md                 # This file
└── label_fix_summary.md            # Technical details

processed/
└── phase1_with_labels/
    ├── train_data.pkl              # Training data with labels
    ├── val_data.pkl                # Validation data with labels
    ├── test_data.pkl               # Test data with labels
    └── metadata.json               # Processing metadata
```

## Next Steps

1. ✅ Run preprocessing with label support
2. ✅ Verify labels with `diagnose_labels.py`
3. ✅ Check label distribution matches expected prevalence
4. ✅ Implement model training with multi-label classification
5. ✅ Use class weights for imbalanced labels
6. ✅ Evaluate on all 14 disease findings

## Support

For issues or questions:
- Check `docs/label_fix_summary.md` for technical details
- Review diagnostic output from `diagnose_labels.py`
- Verify MIMIC-CXR file integrity

## References

- [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/)
- [CheXpert Labeling](https://arxiv.org/abs/1901.07031)
- [MIMIC-IV Database](https://physionet.org/content/mimiciv/3.1/)
