# Phase 3: Multi-Modal Integration and Final Dataset Preparation

## Overview

Phase 3 is the final stage of the MIMIC-IV-CXR-ED preprocessing pipeline. It takes the enhanced outputs from Phase 2 (pseudo-notes with RAG enhancement) and creates final, model-ready datasets with comprehensive quality validation and multi-modal integration.

## Purpose

Phase 3 serves several critical functions:

1. **Data Quality Validation**: Ensures all modalities are present, properly formatted, and ready for model training
2. **Multi-Modal Integration**: Combines vision, text, and clinical features into a unified format
3. **Quality Metrics**: Generates comprehensive statistics and quality reports
4. **Model-Ready Format**: Creates final datasets optimized for PyTorch DataLoader and model training

## Architecture

```
Phase 2 Enhanced Data                    Phase 3 Integration
├── train_data_enhanced.pt      →       ├── Validation
├── val_data_enhanced.pt        →       ├── Integration
└── test_data_enhanced.pt       →       └── Final Datasets
                                             ├── train_final.pt
                                             ├── val_final.pt
                                             ├── test_final.pt
                                             └── phase3_metadata.json
```

## Components

### 1. DataQualityValidator

Validates data completeness and quality across all modalities.

**Validation Checks:**
- Required Phase 1 fields: `subject_id`, `study_id`, `dicom_id`, `image`, `clinical_features`, `labels`
- Phase 2 fields: `pseudo_note`, `enhanced_note`, `enhanced_text_tokens`
- Image tensor shape: `[3, 518, 518]`
- Clinical features tensor format
- Enhanced text tokens structure: `{input_ids, attention_mask}`

**Statistics Collected:**
- Total records and validation rate
- Missing data counts (images, clinical, text, labels)
- Average text lengths (pseudo-note and enhanced note)
- View position distribution

### 2. MultiModalIntegrator

Integrates all modalities into a unified, model-ready format.

**Integration Process:**
1. Extract vision modality (BiomedCLIP input)
   - Image tensor `[3, 518, 518]`
   - Attention regions from edge detection

2. Prepare text modality (Clinical ModernBERT input)
   - Tokenized enhanced notes (`input_ids`, `attention_mask`)
   - Raw text (pseudo-note, enhanced note) for analysis

3. Include clinical features
   - Normalized tensor of structured clinical data

4. Preserve metadata
   - Labels, view position, RAG knowledge
   - Processing flags for all phases

**Output Format:**
```python
{
    # Identifiers
    'subject_id': int,
    'study_id': int,
    'dicom_id': str,

    # Vision modality
    'image': torch.Tensor,              # [3, 518, 518]
    'attention_regions': Dict,

    # Text modality
    'text_input_ids': torch.Tensor,
    'text_attention_mask': torch.Tensor,
    'pseudo_note': str,
    'enhanced_note': str,

    # Clinical features
    'clinical_features': torch.Tensor,

    # Metadata
    'labels': Dict,
    'view_position': str,
    'retrieved_knowledge': List[str],

    # Flags
    'phase1_processed': True,
    'phase2_processed': True,
    'phase3_integrated': True
}
```

### 3. Phase3Processor

Main orchestrator that coordinates validation, integration, and output generation.

**Processing Pipeline:**
1. Load Phase 2 enhanced splits
2. Validate data quality
3. Integrate modalities
4. Save final datasets
5. Generate metadata and reports

## Usage

### Basic Usage

```bash
# Process full datasets
python src/phase3_integration.py \
  --input-path processed/phase1_output \
  --gcs-bucket bergermimiciv \
  --gcs-project-id YOUR_PROJECT_ID
```

### Local Testing

```bash
# Test with small samples
python src/phase3_integration.py \
  --input-path processed/phase1_output \
  --use-small-sample
```

### GCS Mode

```bash
# Using Google Cloud Storage
python src/phase3_integration.py \
  --input-path gs://bergermimiciv/processed/phase1_output \
  --gcs-bucket bergermimiciv \
  --gcs-project-id YOUR_PROJECT_ID
```

## Command-Line Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--input-path` | str | Yes | Path to Phase 2 output directory (local or GCS) |
| `--gcs-bucket` | str | No | GCS bucket name (if using GCS) |
| `--gcs-project-id` | str | No | GCP project ID for requester pays |
| `--use-small-sample` | flag | No | Process small sample files for testing |

## Input Requirements

Phase 3 expects the following files from Phase 2:

### Full Dataset Mode
```
processed/phase1_output/
├── train_data_enhanced.pt
├── val_data_enhanced.pt
└── test_data_enhanced.pt
```

### Small Sample Mode
```
processed/phase1_output/
├── train_small_enhanced.pt
├── val_small_enhanced.pt
└── test_small_enhanced.pt
```

## Output Files

### Final Datasets

**Full Mode:**
- `train_final.pt` - Final training dataset
- `val_final.pt` - Final validation dataset
- `test_final.pt` - Final test dataset

**Small Sample Mode:**
- `train_small_final.pt`
- `val_small_final.pt`
- `test_small_final.pt`

### Metadata

**phase3_metadata.json:**
```json
{
  "phase": 3,
  "timestamp": "2025-11-14T12:00:00",
  "description": "Multi-modal integrated datasets ready for model training",
  "splits": {
    "train": {
      "total_records": 12345,
      "valid_records": 12340,
      "validation_rate": 0.9996,
      "avg_text_length": 245.3,
      "avg_enhanced_text_length": 1024.7,
      "view_position_counts": {
        "PA": 7500,
        "AP": 4840
      }
    },
    "val": { ... },
    "test": { ... }
  },
  "config": {
    "max_text_length": 8192,
    "image_size": 518,
    "top_k_retrieval": 5
  },
  "modalities": {
    "vision": {
      "encoder": "BiomedCLIP-CXR",
      "image_size": 518,
      "format": "tensor [3, 518, 518]"
    },
    "text": {
      "encoder": "Clinical ModernBERT",
      "max_length": 8192,
      "format": "tokenized (input_ids, attention_mask)"
    },
    "clinical": {
      "format": "tensor of normalized features",
      "features": [...]
    }
  },
  "data_quality": {
    "train": {
      "validation_rate": "99.96%",
      "valid_records": 12340,
      "total_records": 12345
    },
    "val": { ... },
    "test": { ... }
  }
}
```

## Quality Validation

Phase 3 performs comprehensive quality checks:

### Record-Level Validation

✅ **Required Fields Check**
- All Phase 1 fields present (subject_id, study_id, dicom_id, image, clinical_features, labels)
- All Phase 2 fields present (pseudo_note, enhanced_note, enhanced_text_tokens)
- Phase 2 processing flag set

✅ **Tensor Shape Validation**
- Image: `torch.Size([3, 518, 518])`
- Clinical features: Valid tensor
- Text tokens: `{input_ids: Tensor, attention_mask: Tensor}`

✅ **Data Type Validation**
- Images are PyTorch tensors
- Clinical features are PyTorch tensors
- Enhanced text tokens are dictionaries with tensor values

### Split-Level Statistics

For each split (train/val/test), Phase 3 computes:

- **Validation Rate**: Percentage of records passing all quality checks
- **Text Length Stats**: Average length of pseudo-notes and enhanced notes
- **View Position Distribution**: Count of each CXR view position
- **Missing Data Counts**: Records missing images, clinical data, text, or labels

## Console Output

During processing, Phase 3 provides detailed logging:

```
============================================================
Phase 3: Multi-Modal Integration and Final Dataset Preparation
============================================================
Mode: GCS
Bucket: bergermimiciv
Input path: processed/phase1_output
Use small sample: False
============================================================

Processing train split...
Loading enhanced train split from: processed/phase1_output/train_data_enhanced.pt
Loaded 12345 enhanced records from train split
Validating train split...
Validating train: 100%|████████████| 12345/12345 [00:15<00:00, 800.52it/s]

Validation Results for train:
  Total records:      12,345
  Valid records:      12,340
  Validation rate:    99.96%

Integrating modalities for train split...
Integrating train: 100%|████████████| 12340/12340 [00:08<00:00, 1543.21it/s]
Integrated 12340/12345 records
Saving integrated train split to: processed/phase1_output/train_final.pt

Sample integrated record from train:
  Subject ID:         10000032
  Study ID:           50000123
  DICOM ID:           12345678-abcd1234-12ab34cd-56ef78gh-12345678
  Image shape:        torch.Size([3, 518, 518])
  Text tokens shape:  torch.Size([8192])
  Clinical features:  torch.Size([45])
  View position:      PA
  Pseudo-note (first 150 chars):
    Patient is a 65 year old M. Chief complaint: chest pain. Vital signs: temperature 98.6°F, heart rate 85 bpm, respiratory rate 16 breaths/min, oxy...

[... val and test splits ...]

============================================================
Phase 3 Complete!
============================================================

============================================================
FINAL DATASET REPORT
============================================================

Overall Statistics:
  Total records:     17,639
  Valid records:     17,620
  Validation rate:   99.89%

Split Distribution:
  Train: 12,340 records (99.96% valid)
  Val  : 2,640 records (99.85% valid)
  Test : 2,640 records (99.81% valid)

Text Statistics:
  Train:
    Avg pseudo-note length:  245 chars
    Avg enhanced note length: 1025 chars
  Val:
    Avg pseudo-note length:  243 chars
    Avg enhanced note length: 1018 chars
  Test:
    Avg pseudo-note length:  247 chars
    Avg enhanced note length: 1031 chars

View Position Distribution:
  PA        : 10,500 (59.6%)
  AP        : 7,120 (40.4%)

============================================================
Datasets ready for model training!
============================================================
```

## Integration with Model Training

The final datasets are ready for PyTorch DataLoader:

```python
import torch
from torch.utils.data import Dataset, DataLoader

class MIMICMultiModalDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, weights_only=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        return {
            # Vision modality
            'image': record['image'],
            'attention_regions': record['attention_regions'],

            # Text modality
            'text_input_ids': record['text_input_ids'],
            'text_attention_mask': record['text_attention_mask'],

            # Clinical features
            'clinical_features': record['clinical_features'],

            # Labels
            'labels': record['labels'],

            # Metadata
            'subject_id': record['subject_id'],
            'study_id': record['study_id'],
            'dicom_id': record['dicom_id']
        }

# Load final dataset
train_dataset = MIMICMultiModalDataset('processed/phase1_output/train_final.pt')
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Use in training loop
for batch in train_loader:
    images = batch['image']  # [B, 3, 518, 518]
    text_ids = batch['text_input_ids']  # [B, 8192]
    text_mask = batch['text_attention_mask']  # [B, 8192]
    clinical = batch['clinical_features']  # [B, num_features]

    # Forward pass through multi-modal model
    # ...
```

## Error Handling

Phase 3 includes robust error handling:

1. **Missing Files**: Clear error messages if Phase 2 outputs not found
2. **Invalid Records**: Skips invalid records and logs issues
3. **Validation Failures**: Detailed reporting of validation failures
4. **GCS Errors**: Handles authentication and access issues gracefully

## Performance

### Processing Speed

- **Validation**: ~800-1000 records/second
- **Integration**: ~1500-2000 records/second
- **Full pipeline**: ~10-15 minutes for 100K records

### Memory Usage

- **Small samples**: <1 GB RAM
- **Full datasets**: 4-8 GB RAM (depends on dataset size)
- **Streaming**: Processes one split at a time to minimize memory

## Best Practices

1. **Test with Small Samples First**
   ```bash
   python src/phase3_integration.py \
     --input-path processed/phase1_output \
     --use-small-sample
   ```

2. **Check Validation Rates**
   - Target: >99% validation rate
   - If lower, investigate issues in Phase 1 or Phase 2

3. **Review Sample Records**
   - Phase 3 logs sample records for inspection
   - Verify data looks correct before training

4. **Keep Metadata**
   - `phase3_metadata.json` contains important quality metrics
   - Reference during model development and debugging

## Troubleshooting

### Low Validation Rate

**Problem**: Validation rate <95%

**Solutions**:
- Check Phase 2 logs for processing errors
- Verify Phase 1 image preprocessing completed successfully
- Review sample invalid records to identify patterns

### Missing Phase 2 Files

**Problem**: "Error loading enhanced split"

**Solutions**:
- Verify Phase 2 completed successfully
- Check file paths and GCS bucket configuration
- Ensure `--input-path` points to Phase 2 output directory

### Memory Issues

**Problem**: Out of memory during processing

**Solutions**:
- Use `--use-small-sample` for testing
- Process on machine with more RAM (16+ GB recommended)
- Close other applications to free memory

### GCS Access Errors

**Problem**: "AccessDeniedException" or "DefaultCredentialsError"

**Solutions**:
```bash
# Authenticate with GCS
gcloud auth application-default login

# Set project for requester pays
gcloud config set project YOUR_PROJECT_ID

# Verify authentication
gcloud auth list
```

## Next Steps

After Phase 3 completes successfully:

1. ✅ **Verify Output**: Check validation rates and sample records
2. ✅ **Review Metadata**: Examine `phase3_metadata.json` for quality metrics
3. ✅ **Test DataLoader**: Create PyTorch Dataset and verify data loading
4. ✅ **Begin Model Training**: Use final datasets with your multi-modal model

## Related Documentation

- [Phase 1: Data Preprocessing](../README.md#phase-1-data-preprocessing)
- [Phase 2: Enhanced Pseudo-Note Generation](PHASE2_ENHANCED_NOTES.md)
- [README: Main Documentation](../README.md)
- [GCS Setup Guide](GCS_SETUP.md)

## Summary

Phase 3 is the final preprocessing stage that:
- ✅ Validates data quality across all modalities
- ✅ Integrates vision, text, and clinical features
- ✅ Creates model-ready datasets with comprehensive validation
- ✅ Generates detailed quality reports and metadata
- ✅ Prepares data for efficient PyTorch DataLoader integration

The output of Phase 3 is production-ready datasets optimized for training multi-modal medical AI models.
