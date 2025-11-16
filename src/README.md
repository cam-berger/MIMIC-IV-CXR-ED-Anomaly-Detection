# Source Code - Integrated Phase 1 Pipeline

## Overview

This directory contains the complete, streamlined Phase 1 preprocessing pipeline with integrated CheXpert disease label support.

## Key Files

### `phase1_preprocess_streaming.py`

**Main preprocessing pipeline** - Processes MIMIC-CXR, MIMIC-IV, and MIMIC-IV-ED data into training-ready format.

**Features:**
- ✅ BiomedCLIP-CXR image processing (518x518)
- ✅ Clinical ModernBERT text processing (8192 tokens)
- ✅ RAG knowledge enhancement
- ✅ **CheXpert disease label attachment** (14 findings)
- ✅ Google Cloud Storage support
- ✅ Official MIMIC-CXR splits (patient-level)
- ✅ Parallel batch processing
- ✅ Memory-efficient chunked output

**Usage:**
```bash
python src/phase1_preprocess_streaming.py \
    --mimic-cxr-path /path/to/mimic-cxr \
    --output-path processed/phase1 \
    --label-format multi_label
```

See `docs/PHASE1_USAGE.md` for complete documentation.

### `fix_chexpert_labels.py`

**CheXpert label processor** - Loads and formats disease labels for training.

**Key Components:**
- `CheXpertLabelProcessor` - Main class for label loading
- `enhance_record_with_labels()` - Attach labels to records
- `get_label_weights()` - Calculate class weights for imbalanced data
- `CHEXPERT_LABELS` - List of 14 disease findings

**Label Formats:**
- `multi_label` - Binary (0/1) for each disease
- `multi_class` - Preserves uncertainty (-1/0/1)
- `positive_only` - List of positive findings
- `full` - Complete raw values

**Usage:**
```python
from src.fix_chexpert_labels import CheXpertLabelProcessor, CHEXPERT_LABELS

processor = CheXpertLabelProcessor()
processor.load_chexpert_labels('mimic-cxr-2.0.0-chexpert.csv.gz')

labels = processor.format_labels_for_training(
    study_id=12345,
    format_type='multi_label'
)
# Returns: {'disease_labels': [...], 'disease_binary': [...], 'label_array': array(...)}
```

### `diagnose_labels.py`

**Label diagnostic tool** - Verifies labels are properly attached to preprocessed data.

**Features:**
- Analyzes label statistics
- Compares to expected prevalence
- Identifies missing or malformed labels
- Generates diagnostic reports

**Usage:**
```bash
# Check single file
python src/diagnose_labels.py processed/phase1/train_data.pkl

# Check all splits
python src/diagnose_labels.py processed/phase1/ --check-all-splits

# Save JSON report
python src/diagnose_labels.py processed/phase1/train_data.pkl --save-report report.json
```

**Expected Output:**
```
✓ Labels appear to be properly attached
Records with positive findings: 52.3% (104,600/200,000)

LABEL DISTRIBUTION:
  Support Devices:              48.2% (96,400)
  Lung Opacity:                 39.1% (78,200)
  Pleural Effusion:             28.5% (57,000)
  Atelectasis:                  27.3% (54,600)
  Cardiomegaly:                 24.7% (49,400)
```

## Integration Details

The pipeline integrates labels through these key modifications:

1. **Import** (line 65-70):
```python
from fix_chexpert_labels import (
    CheXpertLabelProcessor,
    enhance_record_with_labels,
    CHEXPERT_LABELS
)
```

2. **Configuration** (line 130-133):
```python
# CheXpert label settings
chexpert_labels_path: str = "mimic-cxr-2.0.0-chexpert.csv.gz"
label_format_type: str = "multi_label"
handle_uncertain_as: str = "negative"
```

3. **Initialization** (line 775-782):
```python
# Initialize CheXpert label processor
self.label_processor = CheXpertLabelProcessor(
    gcs_helper=self.gcs_helper if config.use_gcs else None,
    use_gcs=config.use_gcs
)

# Load CheXpert labels
self._load_chexpert_labels()
```

4. **Label Attachment** (line 990-1016):
```python
# CRITICAL: Attach CheXpert disease labels
if self.label_processor is not None:
    study_labels = self.label_processor.format_labels_for_training(
        int(row['study_id']),
        format_type=self.config.label_format_type
    )
    record['labels'].update(study_labels)
```

## Data Structure

### Input Data

- **MIMIC-CXR**: Chest X-ray images + metadata
  - `files/p##/p#######/s########/*.jpg` - Images
  - `mimic-cxr-2.0.0-chexpert.csv.gz` - Disease labels
  - `mimic-cxr-2.0.0-split.csv.gz` - Official splits

- **MIMIC-IV**: Clinical data (vitals, labs, notes)
- **MIMIC-IV-ED**: Emergency department data

### Output Structure

Each preprocessed record contains:

```python
{
    # Identifiers
    'subject_id': int,
    'study_id': int,
    'dicom_id': str,

    # Image data
    'image': torch.Tensor,  # Shape: (3, 518, 518)
    'attention_regions': Dict,

    # Text data
    'text_tokens': Dict,
    'clinical_features': Dict,

    # Knowledge
    'retrieved_knowledge': List[str],

    # Labels
    'labels': {
        'view_position': str,
        'disease_labels': List[str],  # ['Cardiomegaly', 'Pleural Effusion']
        'disease_binary': List[int],  # [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
        'label_array': np.ndarray     # Shape: (14,), dtype: float32
    }
}
```

## Label Statistics

### Expected Prevalence in MIMIC-CXR

| Finding | Positive % | Train (~200k) | Val (~3k) | Test (~3k) |
|---------|-----------|--------------|-----------|------------|
| Support Devices | 48% | 96,000 | 1,440 | 1,440 |
| Lung Opacity | 39% | 78,000 | 1,170 | 1,170 |
| Pleural Effusion | 28% | 56,000 | 840 | 840 |
| Atelectasis | 27% | 54,000 | 810 | 810 |
| Cardiomegaly | 25% | 50,000 | 750 | 750 |
| Edema | 15% | 30,000 | 450 | 450 |
| Consolidation | 12% | 24,000 | 360 | 360 |
| No Finding | 19% | 38,000 | 570 | 570 |
| Pneumonia | 4% | 8,000 | 120 | 120 |
| Pneumothorax | 5% | 10,000 | 150 | 150 |
| Enlarged Cardiomediastinum | 10% | 20,000 | 300 | 300 |
| Fracture | 2% | 4,000 | 60 | 60 |
| Lung Lesion | 3% | 6,000 | 90 | 90 |
| Pleural Other | 3% | 6,000 | 90 | 90 |

## Quick Verification

After preprocessing, verify everything is working:

```bash
# 1. Check files were created
ls -lh processed/phase1/*.pkl

# 2. Run diagnostics
python src/diagnose_labels.py processed/phase1/train_data.pkl

# 3. Quick Python check
python -c "
import pickle
with open('processed/phase1/train_data.pkl', 'rb') as f:
    data = pickle.load(f)
    sample = data[0]
    print('Keys:', list(sample.keys()))
    print('Labels:', list(sample['labels'].keys()))
    print('Disease labels:', sample['labels'].get('disease_labels', []))
    print('Label array shape:', sample['labels'].get('label_array', []).shape)
"
```

**Expected output:**
```
Keys: ['subject_id', 'study_id', 'dicom_id', 'image', 'attention_regions', 'text_tokens', 'clinical_features', 'retrieved_knowledge', 'labels']
Labels: ['view_position', 'disease_labels', 'disease_binary', 'label_array']
Disease labels: ['Cardiomegaly', 'Support Devices']
Label array shape: (14,)
```

## Training Example

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle

class MIMICDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'image': sample['image'],
            'labels': torch.tensor(sample['labels']['label_array'], dtype=torch.float32)
        }

# Load data
train_dataset = MIMICDataset('processed/phase1/train_data.pkl')
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop
model = YourModel()  # 14 output units for binary classification
criterion = nn.BCEWithLogitsLoss()

for batch in train_loader:
    images = batch['image']  # Shape: (batch_size, 3, 518, 518)
    labels = batch['labels']  # Shape: (batch_size, 14)

    outputs = model(images)  # Shape: (batch_size, 14)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()
```

## Performance Notes

- **Processing time**: ~2-4 hours for full MIMIC-CXR dataset (local SSD)
- **Memory usage**: ~8-16 GB RAM (depends on batch size)
- **Storage**: ~500 GB for full preprocessed dataset
- **GCS**: Add ~20-30% time for network I/O

**Optimization tips:**
- Use `--batch-size 200` for faster processing (if memory allows)
- Use `--num-workers 8` for I/O bound operations
- Use `--skip-final-combine` to keep chunked output for large datasets
- Enable GCS for distributed processing

## Troubleshooting

See `docs/PHASE1_USAGE.md` for detailed troubleshooting guide.

## References

- Main documentation: `docs/PHASE1_USAGE.md`
- Technical details: `docs/label_fix_summary.md`
- Original preprocessing patch: `docs/src/preprocessing_patch.py`
