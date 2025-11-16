# CXR-PRO Integration Guide

## Overview

This document describes the integration of **CXR-PRO** (Chest X-Ray Prior Reference Omitted) into the MIMIC-IV-CXR-ED Anomaly Detection pipeline. CXR-PRO removes hallucinated references to prior radiology reports, improving the factual consistency and accuracy of radiology report generation and utilization.

## What is CXR-PRO?

**CXR-PRO** is an adaptation of the MIMIC-CXR dataset that automatically removes references to prior radiology reports using a fine-tuned BioBERT model called **GILBERT** (Generating In-text Labels of References to Priors with BioBERT).

### Key Features

- **374,139 radiology reports** with prior references removed
- **Token-level prior removal** using NER-based classification
- **Expert-validated test set** (2,188 reports) created by radiologists
- **Improves report quality** by eliminating hallucinated prior comparisons

### Why Remove Prior References?

Radiology reports often contain phrases like:
- "unchanged from prior"
- "stable compared to last exam"
- "new since previous study"

When training ML models on these reports, models learn to hallucinate references to non-existent priors. CXR-PRO solves this by:
1. Removing comparative references ("unchanged", "stable")
2. Rewriting ambiguous statements (e.g., "heart size is stable" → "heart size is abnormal")
3. Preserving current findings and observations

## Architecture Enhancement

### Before CXR-PRO (3 Modalities)
```
Vision (CXR Image) ──┐
Clinical Text (ED)  ─┼─→ Fusion → Classification
Clinical Features  ──┘
```

### After CXR-PRO (4 Modalities)
```
Vision (CXR Image)     ──┐
Clinical Text (ED)      ─┼─→ 4-Way Fusion → Classification
Radiology Text (CXR)    ─┤
Clinical Features       ──┘
```

### Benefits of 4-Modal Architecture

1. **Complementary Information**:
   - Clinical text: Patient presentation, symptoms, vitals
   - Radiology text: Expert radiological findings

2. **Cross-Modal Alignment**:
   - Attention learns to align visual findings with radiological descriptions
   - Relates clinical presentation to imaging observations

3. **Improved Accuracy**:
   - Additional signal from expert radiology impressions
   - Prior-free reports prevent confusion from temporal comparisons

## Implementation Components

### 1. CXR-PRO Integration (`src/cxr_pro_integration.py`)

Core module for working with CXR-PRO data:

**Classes:**
- `GILBERTModel`: Wraps HuggingFace's `rajpurkarlab/gilbert` model for prior removal
- `CXRProIntegrator`: Loads and processes CXR-PRO data

**Features:**
- Load pre-processed CXR-PRO data (if available)
- Apply GILBERT model to MIMIC-CXR reports (process from scratch)
- Extract IMPRESSION sections from radiology reports
- Track prior removal statistics

**Usage:**
```python
from cxr_pro_integration import CXRProIntegrator, CXRProConfig

config = CXRProConfig(
    cxr_pro_train_path="path/to/mimic_train_impressions.csv",
    cxr_pro_test_path="path/to/mimic_test_impressions.csv",
    device="cuda"
)

integrator = CXRProIntegrator(config)

# Load pre-processed CXR-PRO data
train_df = integrator.load_cxr_pro_preprocessed("train")

# Or process MIMIC-CXR reports with GILBERT
cleaned_df = integrator.process_reports(study_ids, report_paths)
```

### 2. Phase 1 Adapter (`src/phase1_cxr_pro_adapter.py`)

Integrates CXR-PRO impressions into existing Phase 1 preprocessed data:

**Workflow:**
1. Load Phase 1 `.pt` files (train/val/test)
2. Load CXR-PRO cleaned impressions
3. Match impressions to study_ids
4. Tokenize impressions (BiomedBERT)
5. Save enhanced `.pt` files with radiology context

**Output Format:**
```python
{
    # Existing Phase 1 fields
    'subject_id': int,
    'study_id': int,
    'dicom_id': str,
    'image': torch.Tensor,
    'clinical_features': torch.Tensor,
    'labels': Dict,

    # NEW: CXR-PRO radiology fields
    'radiology_impression_cleaned': str,
    'radiology_tokens': {
        'input_ids': torch.Tensor,
        'attention_mask': torch.Tensor
    },
    'has_radiology_report': bool,
    'prior_removal_stats': Dict
}
```

**Usage:**
```bash
python src/phase1_cxr_pro_adapter.py \
  --phase1-data-path processed/phase1_output \
  --cxr-pro-train path/to/mimic_train_impressions.csv \
  --cxr-pro-test path/to/mimic_test_impressions.csv \
  --output-path processed/phase1_with_radiology \
  --text-encoder microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --max-length 512
```

### 3. Enhanced Model (`src/model/enhanced_mdfnet_radiology.py`)

Extended Enhanced MDF-Net with 4 modalities:

**New Components:**
- `RadiologyEncoder`: BiomedBERT encoder for CXR-PRO impressions
- `QuadModalAttentionFusion`: 4-way cross-modal attention
- `EnhancedMDFNetWithRadiology`: Complete 4-modal architecture

**Architecture:**
```
INPUT (4 modalities):
├─ Vision: [B, 3, 518, 518] CXR images
├─ Clinical Text: [B, max_len] ED notes + RAG
├─ Radiology Text: [B, 512] CXR-PRO impressions (NEW!)
└─ Clinical Features: [B, 45] Vitals, demographics

ENCODERS:
├─ BiomedCLIP (vision): → [B, 512]
├─ Clinical ModernBERT (clinical text): → [B, 768]
├─ BiomedBERT (radiology text): → [B, 768] (NEW!)
└─ Dense layers (clinical features): → [B, 256]

FUSION:
└─ 4-way Cross-Modal Attention: → [B, 3072]

OUTPUT:
└─ Classification Head: → [B, 14] abnormality probabilities
```

**Usage:**
```python
from model.enhanced_mdfnet_radiology import EnhancedMDFNetWithRadiology

model = EnhancedMDFNetWithRadiology(
    num_classes=14,
    clinical_feature_dim=45,
    freeze_encoders=True,
    fusion_dim=768,
    num_heads=8
)

outputs = model(
    images=images,
    clinical_text_input_ids=clinical_tokens['input_ids'],
    clinical_text_attention_mask=clinical_tokens['attention_mask'],
    radiology_input_ids=radiology_tokens['input_ids'],
    radiology_attention_mask=radiology_tokens['attention_mask'],
    clinical_features=clinical_features,
    return_attention=True
)

logits = outputs['logits']  # [B, 14]
probs = outputs['probabilities']  # [B, 14]
attention = outputs['attention_weights']  # [B, num_heads, 4, 4]
```

**Model Parameters:**
- **Total**: ~327M parameters
  - Vision encoder: 87M (BiomedCLIP)
  - Clinical text encoder: 149M (Clinical ModernBERT)
  - Radiology encoder: 109M (BiomedBERT) ← NEW!
  - Clinical encoder: 0.5M
  - Fusion: 3M
  - Classification head: 1M

### 4. Data Quality Validation (`scripts/validate_cxr_pro_quality.py`)

Validates that prior references have been properly removed:

**Checks:**
- Remaining prior references (regex patterns)
- Incomplete cleaning (e.g., "stable" without object)
- Quality issues (empty, too short, PHI markers)
- Length statistics

**Patterns Detected:**
```python
PRIOR_REFERENCE_PATTERNS = [
    r'\b(unchanged|stable|improved|worsened)\s+(from|since)',
    r'\b(compared\s+to|similar\s+to)\s+(prior|previous)',
    r'\b(no\s+change|interval\s+change|new\s+since)',
    r'\b(prior\s+(study|exam|radiograph))',
    ...
]
```

**Usage:**
```bash
python scripts/validate_cxr_pro_quality.py \
  --data-path processed/phase1_with_radiology \
  --splits train val test \
  --output-report validation_report.json \
  --output-txt validation_report.txt
```

**Example Output:**
```
======================================================================
CXR-PRO Data Quality Validation Report - TRAIN Split
======================================================================

Total Impressions: 50000

PRIOR REFERENCES (should be 0):
  - With prior references: 23 (0.05%)

CLEANING QUALITY:
  - Incomplete cleaning: 145 (0.29%)
  - Quality issues: 89 (0.18%)
  - Empty impressions: 412 (0.82%)

LENGTH STATISTICS:
  - Average length: 187.3 chars
  - Median length: 165.0 chars
  - Distribution:
      0-50: 1203 (2.4%)
      50-100: 8945 (17.9%)
      100-200: 28432 (56.9%)
      200+: 11420 (22.8%)

OVERALL ASSESSMENT:
  ✓ PASS: Prior references successfully removed
```

## Quick Start Guide

### Option 1: Use Pre-Processed CXR-PRO Data (Recommended)

If you have CXR-PRO data from PhysioNet:

```bash
# Step 1: Download CXR-PRO from PhysioNet
# Navigate to: https://physionet.org/content/cxr-pro/1.0.0/
# Download:
#   - mimic_train_impressions.csv (371,951 reports)
#   - mimic_test_impressions.csv (2,188 expert-edited reports)

# Step 2: Integrate into Phase 1 data
python src/phase1_cxr_pro_adapter.py \
  --phase1-data-path processed/phase1_output \
  --cxr-pro-train /path/to/mimic_train_impressions.csv \
  --cxr-pro-test /path/to/mimic_test_impressions.csv \
  --output-path processed/phase1_with_radiology

# Step 3: Validate data quality
python scripts/validate_cxr_pro_quality.py \
  --data-path processed/phase1_with_radiology \
  --splits train val test

# Step 4: Update training config to use 4-modal model
# Edit configs/phase3_enhanced_rag.yaml:
#   model_class: enhanced_mdfnet_radiology
#   use_radiology: true

# Step 5: Train 4-modal model
python src/training/train_lightning.py \
  --config configs/phase3_enhanced_rag_radiology.yaml
```

### Option 2: Process MIMIC-CXR Reports with GILBERT

If you don't have CXR-PRO, apply GILBERT to your MIMIC-CXR reports:

```bash
# Step 1: Process MIMIC-CXR reports with GILBERT
python src/cxr_pro_integration.py \
  --mode preprocess \
  --mimic-cxr-path /path/to/mimic-cxr-jpg/2.1.0 \
  --output-path processed/cxr_pro_cleaned \
  --batch-size 16 \
  --device cuda

# Step 2: Follow Option 1 steps 2-5
```

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CXR-PRO INTEGRATION FLOW                      │
└─────────────────────────────────────────────────────────────────┘

MIMIC-CXR Reports                CXR-PRO Data
(with priors)                    (prior-free)
      │                                │
      │                                │
      ├─── Phase 1 Processing          │
      │    (images + clinical)         │
      │    │                           │
      │    └─→ train_data.pt           │
      │        val_data.pt             │
      │        test_data.pt            │
      │                                │
      │                                │
      └────────────┬───────────────────┘
                   │
                   ▼
         Phase 1 CXR-PRO Adapter
         (match + tokenize)
                   │
                   ▼
         train_data_with_radiology.pt
         val_data_with_radiology.pt
         test_data_with_radiology.pt
                   │
                   ▼
         Enhanced MDF-Net (4 modalities)
         ├─ Vision
         ├─ Clinical Text
         ├─ Radiology Text ← NEW!
         └─ Clinical Features
                   │
                   ▼
         Abnormality Predictions (14 classes)
```

## Expected Performance Improvements

Based on CXR-PRO paper results:

### Factual Consistency
- **Before CXR-PRO**: Models hallucinate prior references ~15-20% of the time
- **After CXR-PRO**: Prior references reduced to <1%

### Clinical Metrics (CXR-RePaiR model)
- **BLEU-4**: +2.3 points
- **RadGraph F1**: +3.1 points (factual accuracy)
- **Precision**: +4.2 points (fewer false positives)

### Multi-Modal Benefits
Our 4-modal architecture provides:
- **Additional signal**: Expert radiological findings complement clinical notes
- **Cross-modal alignment**: Attention learns visual-text correspondences
- **Reduced confusion**: Prior-free reports eliminate temporal ambiguity

## Troubleshooting

### Issue: "No matching impressions found"

**Cause**: Study IDs in Phase 1 data don't match CXR-PRO study_ids

**Solution:**
1. Check that CXR-PRO data is from MIMIC-CXR-JPG 2.1.0 (same version)
2. Verify study_id format (should be integers)
3. Check train/val/test split alignment

```bash
# Debug: Check study_id overlap
python -c "
import torch
import pandas as pd

# Load Phase 1 data
phase1 = torch.load('processed/phase1_output/train_data.pt')
phase1_ids = set(r['study_id'] for r in phase1)

# Load CXR-PRO
cxr_pro = pd.read_csv('mimic_train_impressions.csv')
cxr_pro_ids = set(cxr_pro['study_id'])

# Check overlap
print(f'Phase 1 study_ids: {len(phase1_ids)}')
print(f'CXR-PRO study_ids: {len(cxr_pro_ids)}')
print(f'Overlap: {len(phase1_ids & cxr_pro_ids)}')
"
```

### Issue: "GILBERT model fails to load"

**Cause**: Missing HuggingFace transformers or model not available

**Solution:**
```bash
# Install/update transformers
pip install --upgrade transformers

# Test GILBERT loading
python -c "
from transformers import AutoModelForTokenClassification
model = AutoModelForTokenClassification.from_pretrained('rajpurkarlab/gilbert')
print('GILBERT loaded successfully')
"
```

### Issue: "Too many empty impressions"

**Cause**: Aggressive prior removal or missing impressions

**Solution:**
1. Check CXR-PRO data quality with validation script
2. Verify IMPRESSION sections exist in reports
3. Consider using expert-validated test set for critical cases

### Issue: "Out of memory during tokenization"

**Cause**: Processing too many impressions at once

**Solution:**
```bash
# Reduce batch size or max length
python src/phase1_cxr_pro_adapter.py \
  --phase1-data-path processed/phase1_output \
  --cxr-pro-train mimic_train_impressions.csv \
  --cxr-pro-test mimic_test_impressions.csv \
  --output-path processed/phase1_with_radiology \
  --max-length 256  # Reduce from 512
```

## References

### CXR-PRO Paper
- **Title**: "Improving Radiology Report Generation Systems By Removing Hallucinated References to Non-existent Priors"
- **Authors**: Rajpurkar Lab, Stanford University
- **Link**: [arXiv preprint](https://arxiv.org/abs/2210.06340)

### CXR-PRO Dataset
- **PhysioNet**: https://physionet.org/content/cxr-pro/1.0.0/
- **Size**: 374,139 reports (371,951 train + 2,188 expert-edited test)

### GILBERT Model
- **HuggingFace**: `rajpurkarlab/gilbert`
- **Type**: Fine-tuned BioBERT for token classification (NER)
- **GitHub**: https://github.com/rajpurkarlab/CXR-ReDonE

### Related Work
- **MIMIC-CXR**: Johnson et al., 2019 - Original dataset with 377K reports
- **BioBERT**: Lee et al., 2019 - Biomedical language model
- **CXR-RePaiR**: Chen et al., 2021 - Radiology report generation model

## Citation

If you use CXR-PRO in your research, please cite:

```bibtex
@article{ramesh2022improving,
  title={Improving Radiology Report Generation Systems by Removing Hallucinated References to Non-existent Priors},
  author={Ramesh, Vignav and others},
  journal={arXiv preprint arXiv:2210.06340},
  year={2022}
}
```

And cite the original MIMIC-CXR dataset:

```bibtex
@article{johnson2019mimic,
  title={MIMIC-CXR, a de-identified publicly available database of chest radiographs with free-text reports},
  author={Johnson, Alistair EW and others},
  journal={Scientific data},
  volume={6},
  number={1},
  pages={1--8},
  year={2019}
}
```

## Contact & Support

- **Issues**: Open a GitHub issue
- **Questions**: Check existing documentation
- **CXR-PRO Data**: Apply for PhysioNet credentialing

---

**Last Updated**: 2025-11-15
**Status**: Implementation Complete | 4-Modal Architecture | Quality Validation | Documentation Complete
