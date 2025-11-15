# MIMIC Multi-Modal Dataset

HYPOTHESIS: Context-aware knowledge augmentation of clinical notes, when fused with visual features through cross-modal attention, will improve both the accuracy and interpretability of chest X-ray abnormality detection compared to models using raw clinical notes or images alone, with the improvement being most significant for rare conditions and complex multi-abnormality cases.

## Recent Improvements (November 2025)

**Phase 3: Enhanced RAG Training Pipeline**:
- **Official MIMIC-CXR Splits**: Implemented patient-level splits from `mimic-cxr-2.0.0-split.csv.gz` (377,110 studies)
- **Enhanced RAG Adapter**: Automatic conversion between Enhanced RAG and Standard data formats
- **Multi-Format Support**: Auto-detects and converts Enhanced RAG format with RAG-enhanced notes, attention segments, bounding boxes, and severity scores
- **BiomedCLIP Integration**: Medical-domain vision encoder (512-dim) loaded via open_clip
- **ModernBERT Support**: 8192-token context text encoder with cascading fallbacks (ModernBERT → BiomedBERT → BERT-base)
- **Bug Fixes**: Fixed 4 critical bugs in DataLoader, model loading, and dimension handling
- **Robust Testing**: Comprehensive test suite validates entire training pipeline
- **Production Ready**: 354M parameter model ready for training on Enhanced RAG data

**Previous Improvements (October 2025)**:
- **20-40x Faster Processing**: Batch downloading with parallel workers reduces processing time from 12 days to 7-15 hours
- **Fixed Critical Path Bug**: Corrected MIMIC-CXR image path construction (8-digit padding)
- **Smart Caching**: Eliminates duplicate downloads
- **Improved Temporal Matching**: Fixed StudyDate+StudyTime parsing (107,949+ matched records)
- **Memory-Efficient Streaming**: Process 100K+ records on 7.5GB RAM without OOM crashes

### Technical Documentation

**Phase 3 Training Pipeline**:
- **[Enhanced RAG Adapter](docs/ENHANCED_RAG_ADAPTER.md)**: Auto-detection and conversion between Enhanced RAG and Standard formats
- **[Bug Fixes Summary](BUG_FIXES_SUMMARY.md)**: All 4 critical bugs fixed (DataLoader, BiomedCLIP, ModernBERT, output dimensions)
- **[Troubleshooting Guide](TROUBLESHOOTING.md)**: Comprehensive troubleshooting for training pipeline
- **[Codebase Audit](CODEBASE_AUDIT.md)**: Complete audit report - production ready
- **[Official Splits Documentation](docs/OFFICIAL_SPLITS_FIX.md)**: Patient-level splits implementation

**Training & Evaluation**:
- **[Training Guide](docs/TRAINING_GUIDE.md)**: Fine-tuning Enhanced MDF-Net with multi-GPU support
- **[Evaluation Guide](docs/EVALUATION_GUIDE.md)**: Metrics, confusion matrices, and correlation analysis
- **[Phase 3 Integration Summary](ENHANCED_RAG_INTEGRATION_SUMMARY.md)**: Complete integration guide

**Data Processing**:
- **[OOM Solution Guide](docs/OOM_SOLUTION.md)**: Memory management and optimization
- **[Phase 2 Refactoring Summary](docs/PHASE2_REFACTORING_SUMMARY.md)**: Complete Phase 2 implementation

## Overview

This project implements Phase 1 data preprocessing for an Enhanced MDF-Net model that combines:
- **BiomedCLIP-CXR** vision encoder for chest X-ray analysis
- **Clinical ModernBERT** text encoder for clinical notes
- **RAG knowledge enhancement** for medical context
- **Cross-attention fusion** for multimodal integration

The pipeline links MIMIC-CXR chest X-rays with MIMIC-IV-ED emergency department data to create a unified dataset for medical abnormality detection.

## Key Features

- **Multi-Bucket GCS Support**: Seamlessly works with your data bucket + PhysioNet's public MIMIC-CXR bucket
- **Flexible File Format Support**: Handles both compressed (.csv.gz) and uncompressed (.csv) data files
- **Pseudo-Note Generation**: Converts structured clinical data into narrative text for LLM processing
- **Diagnosis Leakage Filtering**: Removes diagnosis information to prevent data leakage
- **Temporal Alignment**: Links chest X-rays to ED visits within 24-hour windows with accurate StudyDate+StudyTime matching
- **Optimized Image Processing**: Batch downloading with parallel workers (20-40x faster than sequential)
- **Smart Caching**: Eliminates duplicate downloads per record, saving bandwidth and time
- **Robust Error Handling**: Continues processing past failed images with detailed logging
- **Correct Path Construction**: Properly handles MIMIC-CXR's 8-digit padded directory structure
- **Local Testing**: Test preprocessing locally before cloud deployment
- **Automated GCP Deployment**: One-command deployment with auto-shutdown
- **Scalable**: Successfully processes 107,949+ matched multimodal records
- **Cloud-Native**: Optimized for Google Cloud Platform (Compute Engine, Cloud Storage)

## Data Pipeline Architecture

```
MIMIC-IV-ED (Your Bucket)          MIMIC-CXR (PhysioNet's Bucket)
└── Clinical Data                   └── Chest X-ray Images
    ├── ED Stays                        ├── 377K+ images
    ├── Triage Notes                    ├── Frontal views (AP/PA)
    ├── Vital Signs              →      └── DICOM metadata
    └── Medications

         ↓ Phase 1 Preprocessing ↓

    Multi-Modal Dataset (Your Bucket)
    ├── Preprocessed Images (518x518, BiomedCLIP format)
    ├── Pseudo-Notes (Clinical ModernBERT format)
    ├── RAG-Enhanced Context
    └── Train/Val/Test Splits
```

## Model Architecture

### Overview

The Enhanced MDF-Net (Multi-Modal Deep Fusion Network) combines three complementary modalities through cross-modal attention for chest X-ray abnormality detection.

### Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENHANCED MDF-NET ARCHITECTURE                 │
└─────────────────────────────────────────────────────────────────┘

INPUT MODALITIES:
┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│  Chest X-Ray    │  │  Enhanced Text   │  │  Clinical Features  │
│  (518×518×3)    │  │  (8192 tokens)   │  │  (Normalized Vector)│
└────────┬────────┘  └────────┬─────────┘  └──────────┬──────────┘
         │                    │                        │
         ▼                    ▼                        ▼

ENCODERS:
┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│  BiomedCLIP-CXR │  │ Clinical Modern  │  │   Dense Layers      │
│  Vision Encoder │  │  BERT Encoder    │  │  + Batch Norm       │
│                 │  │                  │  │  + Dropout          │
│  Pre-trained on │  │  8192 context    │  │                     │
│  medical images │  │  Specialized for │  │  Maps to latent     │
│                 │  │  clinical text   │  │  space              │
│  Output: [B,D_v]│  │  Output: [B,D_t] │  │  Output: [B,D_c]    │
└────────┬────────┘  └────────┬─────────┘  └──────────┬──────────┘
         │                    │                        │
         └────────────────────┴────────────────────────┘
                              │
                              ▼

FUSION LAYER:
┌──────────────────────────────────────────────────────────┐
│              Cross-Modal Attention Fusion                │
│                                                          │
│  Vision-Text Attention:                                 │
│  • Query: Vision features                               │
│  • Key/Value: Text features (RAG-enhanced)              │
│  • Captures visual-semantic alignment                   │
│                                                          │
│  Multi-Head Attention:                                  │
│  • Attends to all three modalities simultaneously       │
│  • Learns complementary feature interactions            │
│  • Residual connections + Layer Normalization           │
│                                                          │
│  Feature Concatenation:                                 │
│  • Fused = [Vision ⊕ Text ⊕ Clinical ⊕ Attention]      │
│                                                          │
│  Output: [B, D_fused]                                   │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼

CLASSIFICATION HEAD:
┌──────────────────────────────────────────────────────────┐
│  Dense Layer 1: [D_fused → 512]                         │
│  ├─ BatchNorm + ReLU + Dropout(0.3)                     │
│                                                          │
│  Dense Layer 2: [512 → 256]                             │
│  ├─ BatchNorm + ReLU + Dropout(0.2)                     │
│                                                          │
│  Output Layer: [256 → N_classes]                        │
│  ├─ Sigmoid activation (multi-label)                    │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼

OUTPUT:
┌──────────────────────────────────────────────────────────┐
│  Multi-Label Abnormality Predictions                     │
│  • Atelectasis, Cardiomegaly, Consolidation, Edema      │
│  • Enlarged Cardiomediastinum, Fracture, Lung Lesion    │
│  • Lung Opacity, Pleural Effusion, Pleural Other        │
│  • Pneumonia, Pneumothorax, Support Devices             │
│                                                          │
│  Shape: [B, 14] (probability per abnormality)           │
└──────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. **Vision Encoder: BiomedCLIP-CXR**
- **Architecture**: Vision Transformer (ViT) based
- **Pre-training**: Contrastive learning on 217K chest X-ray/report pairs
- **Input**: 518×518 RGB images (resized and normalized)
- **Output**: 768-dimensional vision embeddings
- **Advantages**:
  - Specialized for chest X-ray interpretation
  - Captures anatomical and pathological features
  - Better performance than ImageNet-pretrained models on medical images

#### 2. **Text Encoder: Clinical ModernBERT**
- **Architecture**: Transformer-based language model
- **Context Length**: 8192 tokens (4× longer than BERT)
- **Input**: RAG-enhanced pseudo-notes
  - Pseudo-notes: Generated from structured clinical data (vitals, labs, medications)
  - RAG Enhancement: Augmented with retrieved medical knowledge (FAISS)
- **Output**: 768-dimensional text embeddings
- **Advantages**:
  - Optimized for clinical terminology
  - Long context captures complete patient presentation
  - RAG provides relevant medical knowledge context

#### 3. **Clinical Features Encoder**
- **Input**: Normalized structured features (age, gender, vitals, acuity, pain score)
- **Architecture**:
  - Dense layers with batch normalization
  - Dropout for regularization
- **Output**: 256-dimensional clinical embeddings
- **Purpose**: Captures objective clinical measurements

#### 4. **Cross-Modal Attention Fusion**
- **Multi-Head Attention**: Learns interactions between modalities
  - Vision ↔ Text: Aligns visual findings with clinical descriptions
  - Vision ↔ Clinical: Relates imaging to vital signs
  - Text ↔ Clinical: Connects symptoms with measurements
- **Residual Connections**: Preserves individual modality information
- **Layer Normalization**: Stabilizes training
- **Output**: Unified multi-modal representation

#### 5. **Classification Head**
- **Multi-label Classification**: 14 CheXpert abnormality classes
- **Architecture**:
  - Two fully-connected layers with decreasing dimensions
  - Batch normalization and dropout for regularization
- **Output**: Sigmoid probabilities for each abnormality

### Model Parameters

| Component | Parameters | Description |
|-----------|------------|-------------|
| BiomedCLIP Vision | ~87M | Pre-trained ViT encoder (512-dim output) |
| ModernBERT / BiomedBERT Text | ~149M | Pre-trained text encoder (768-dim output, 8192 tokens) |
| Clinical Feature Encoder | ~0.5M | Dense layers (45 → 256 dimensions) |
| Cross-Modal Attention | ~2M | Multi-head attention fusion (768-dim) |
| Classification Head | ~0.4M | Dense layers (2304 → 14 classes) |
| **Total** | **~354M** | End-to-end trainable |

**Note**: Actual parameter count may vary based on encoder choice (ModernBERT vs BiomedBERT fallback).

### Training Strategy

1. **Stage 1: Feature Extraction**
   - Freeze BiomedCLIP and ModernBERT encoders
   - Train fusion layer and classification head
   - Learn optimal modality combination

2. **Stage 2: Fine-tuning**
   - Unfreeze all layers
   - Fine-tune with lower learning rate
   - Adapt pre-trained models to ED abnormality detection

## Loss Functions

### Primary Loss: Weighted Binary Cross-Entropy (BCE)

Multi-label abnormality detection uses weighted BCE to handle class imbalance:

```python
def weighted_bce_loss(predictions, targets, pos_weights):
    """
    Weighted Binary Cross-Entropy Loss for multi-label classification

    Args:
        predictions: [B, N_classes] - Sigmoid probabilities
        targets: [B, N_classes] - Binary ground truth
        pos_weights: [N_classes] - Weights for positive class per abnormality

    Returns:
        Scalar loss value
    """
    # BCE with positive class weighting
    loss = - (pos_weights * targets * log(predictions) +
              (1 - targets) * log(1 - predictions))

    return loss.mean()
```

**Positive Weights Calculation:**
```python
# For each abnormality class:
pos_weight[i] = n_negative_samples[i] / n_positive_samples[i]

# Example weights (based on CheXpert distribution):
{
    'No Finding': 1.0,           # Most common
    'Atelectasis': 2.8,
    'Cardiomegaly': 3.5,
    'Consolidation': 8.2,
    'Edema': 4.1,
    'Enlarged Cardiomediastinum': 12.3,
    'Fracture': 45.7,            # Rare
    'Lung Lesion': 67.4,         # Very rare
    'Lung Opacity': 1.9,
    'Pleural Effusion': 3.2,
    'Pleural Other': 89.3,       # Very rare
    'Pneumonia': 15.6,
    'Pneumothorax': 21.4,
    'Support Devices': 1.6
}
```

### Auxiliary Loss: Focal Loss (Optional)

For extremely imbalanced classes (Lung Lesion, Pleural Other), focal loss focuses training on hard examples:

```python
def focal_loss(predictions, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss for handling extreme class imbalance

    Args:
        predictions: [B, N_classes] - Sigmoid probabilities
        targets: [B, N_classes] - Binary ground truth
        alpha: Weighting factor for positive class
        gamma: Focusing parameter (higher = more focus on hard examples)

    Returns:
        Scalar loss value
    """
    bce = - (targets * log(predictions) + (1 - targets) * log(1 - predictions))

    # Compute focal weight
    p_t = predictions * targets + (1 - predictions) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma

    # Apply alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    loss = alpha_t * focal_weight * bce

    return loss.mean()
```

### Combined Loss Function

```python
def combined_loss(predictions, targets, pos_weights,
                 lambda_bce=0.7, lambda_focal=0.3):
    """
    Combined loss for robust multi-label classification

    Args:
        predictions: Model predictions
        targets: Ground truth labels
        pos_weights: Class weights for BCE
        lambda_bce: Weight for BCE loss
        lambda_focal: Weight for focal loss

    Returns:
        Combined loss value
    """
    bce = weighted_bce_loss(predictions, targets, pos_weights)
    focal = focal_loss(predictions, targets)

    return lambda_bce * bce + lambda_focal * focal
```

### Loss Components Summary

| Loss Component | Purpose | Weight |
|----------------|---------|--------|
| **Weighted BCE** | Handle class imbalance across all abnormalities | 0.7 |
| **Focal Loss** | Focus on hard examples and rare classes | 0.3 |

### Training Metrics

In addition to loss, we monitor:

**Classification Metrics:**
- **AUROC** (Area Under ROC Curve): Primary metric for each abnormality
- **AUPRC** (Area Under Precision-Recall Curve): For imbalanced classes
- **F1-Score**: Balance between precision and recall
- **Sensitivity/Specificity**: Clinical relevance metrics

**Calibration Metrics:**
- **Expected Calibration Error (ECE)**: Prediction reliability
- **Brier Score**: Probabilistic prediction accuracy

## Datasets Required

### What You Need to Upload to GCS

1. **MIMIC-IV 3.1** - Hospital data
   - `mimiciv/3.1/hosp/` - Admission records
   - `mimiciv/3.1/icu/` - ICU data
   - Size: ~5 GB
   - **Formats supported**: Both `.csv.gz` (compressed) and `.csv` (uncompressed)

2. **MIMIC-IV-ED 2.2** - Emergency Department data (separate dataset!)
   - `mimic-iv-ed/2.2/ed/` - ED stays, triage, vital signs
   - Size: ~2 GB
   - **Formats supported**: Both `.csv.gz` (compressed) and `.csv` (uncompressed)

3. **REFLACX** (Optional) - Eye-gaze annotations
   - `reflacx/` - Radiologist attention maps
   - Size: ~500 MB

### What's Already on GCS (No Upload Needed!)

4. **MIMIC-CXR-JPG 2.1.0** - Chest X-ray images
   - Available on PhysioNet's public bucket: `mimic-cxr-jpg-2.1.0.physionet.org`
   - Size: 500+ GB
   - **You don't need to copy this!** Access it directly from PhysioNet's bucket

## Quick Start

### 1. Prerequisites

```bash
# Install core dependencies
pip install -r requirements.txt

# Download spaCy language models (required for leakage filtering)
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz

# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Data Preprocessing with Official Splits

Process data using official MIMIC-CXR patient-level splits:

```bash
# Phase 1: Preprocess with official splits (RECOMMENDED)
python src/phase1_preprocess_streaming.py \
  --mimic-cxr-path /path/to/mimic-cxr-jpg/2.1.0 \
  --mimic-ed-path /path/to/mimic-iv-ed/2.2/ed \
  --output-dir /path/to/output \
  --use-official-splits  # Uses mimic-cxr-2.0.0-split.csv.gz

# Verify official splits loaded correctly
python scripts/verify_official_splits.py \
  /path/to/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz
```

### 3. Verify Data Loading and Format

Check that Enhanced RAG format data is detected correctly:

```bash
# Verify data loading with adapter
python scripts/verify_data_loading.py \
  --data-root /path/to/preprocessed_data

# Test Enhanced RAG adapter
python scripts/test_enhanced_rag_adapter.py \
  /path/to/train_chunk_000003.pt

# Analyze dataset statistics
python scripts/analyze_dataset.py \
  --data-root /path/to/preprocessed_data
```

### 4. Test Training Pipeline

Validate the entire training pipeline before full training:

```bash
# Test all components: data loading, model creation, forward pass, training step
python scripts/test_training_pipeline.py \
  --config configs/phase3_enhanced_rag.yaml

# Expected output: All 4 tests pass ✅
```

### 5. Start Training

Train the Enhanced MDF-Net model with your Enhanced RAG data:

```bash
# Full multi-modal training with Enhanced RAG adapter
python src/training/train_lightning.py \
  --config configs/phase3_enhanced_rag.yaml \
  --experiment-name "phase3_run1"

# Monitor training with TensorBoard
tensorboard --logdir tb_logs/

# Optional: Multi-GPU training
python src/training/train_lightning.py \
  --config configs/phase3_enhanced_rag.yaml \
  --gpus 2

# Optional: Override config settings
python src/training/train_lightning.py \
  --config configs/phase3_enhanced_rag.yaml \
  --batch-size 16 \
  --max-epochs 30 \
  --lr 1e-4
```

### 6. Google Cloud Setup (Optional)

**First: Authenticate with Google Cloud**

```bash
# Install gcloud CLI if not already installed
# See: https://cloud.google.com/sdk/docs/install

# Authenticate with your Google account
gcloud auth login

# Set up Application Default Credentials (ADC) for Python SDK
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID

# Verify authentication
gcloud auth list
```

**Upload your data and run preprocessing:**

```bash
# Upload MIMIC-IV and MIMIC-IV-ED to your bucket
# Note: The pipeline supports both .csv.gz (compressed) and .csv (uncompressed) files
gsutil -m cp -r ~/MIMIC_Data/physionet.org/files/mimiciv \
  gs://bergermimiciv/

gsutil -m cp -r ~/MIMIC_Data/physionet.org/files/mimic-iv-ed \
  gs://bergermimiciv/

# OPTION A: Automated Deployment (Recommended)
# Automatically creates VM, installs dependencies, runs full pipeline, auto-shuts down
bash scripts/deploy_gcp.sh YOUR_PROJECT_ID bergermimiciv

# OPTION B: Manual Deployment
# Create VM and run pipeline manually
gcloud compute instances create mimic-preprocessing \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --boot-disk-size=200GB \
  --scopes=cloud-platform

# SSH and run full pipeline (preprocessing + leakage filtering)
gcloud compute ssh mimic-preprocessing --zone=us-central1-a
python src/run_full_pipeline.py \
  --gcs-bucket bergermimiciv \
  --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
  --gcs-project-id YOUR_PROJECT_ID \
  --mimic-iv-path mimiciv/3.1 \
  --mimic-ed-path mimic-iv-ed/2.2 \
  --output-path processed/phase1_final \
  --aggressive-filtering \
  --batch-size 100 \
  --num-workers 4

# Performance tuning options:
# --batch-size: Number of images to download in parallel (default: 100)
# --num-workers: Number of parallel download threads (default: 4)
# Increase on more powerful VMs: --batch-size 200 --num-workers 8

# See docs/GCP_DEPLOYMENT.md and DEPLOYMENT_QUICKSTART.md for complete deployment guides
```

## Project Structure

```
.
├── src/
│   ├── phase1_preprocess.py             # Main preprocessing pipeline
│   ├── phase1_preprocess_streaming.py   # Memory-efficient streaming version
│   ├── phase1_stay_identification.py    # ED stay linking
│   ├── phase2_enhanced_notes.py         # Pseudo-note generation + RAG enhancement
│   ├── phase3_integration.py            # Multi-modal data integration
│   ├── run_full_pipeline.py             # Orchestrates preprocessing + leakage filtering
│   ├── apply_leakage_filter.py          # Diagnosis leakage filtering
│   ├── leakage_filt_util.py             # Leakage filtering utilities
│   └── test_phase1_local.py             # Local testing script
├── scripts/
│   ├── deploy_gcp.sh                    # Automated GCP VM deployment
│   └── vm_startup.sh                    # VM initialization script
├── docs/
│   ├── DATAFLOW_SETUP.md                # Google Cloud Dataflow setup
│   ├── DEPLOYMENT_QUICKSTART.md         # Quick deployment reference
│   ├── LOCAL_TESTING.md                 # Local testing guide
│   ├── OOM_SOLUTION.md                  # Out-of-memory solutions
│   ├── PHASE2_ENHANCED_NOTES.md         # Phase 2 documentation
│   ├── PHASE2_REFACTORING_SUMMARY.md    # Phase 2 refactoring details
│   ├── PHASE3_INTEGRATION.md            # Phase 3 integration guide
│   └── QUICK_START.md                   # Quick start guide
└── README.md                            # This file
```

## How It Works

### Phase 1: Data Preprocessing

1. **Load Metadata**
   - MIMIC-CXR: Chest X-ray metadata (from PhysioNet's bucket)
   - MIMIC-IV-ED: ED stays, triage, vital signs (from your bucket)

2. **Join Multimodal Data**
   - Match chest X-rays to ED visits using temporal alignment (±24 hours)
   - Link patient demographics, vital signs, medications

3. **Image Preprocessing**
   - Resize to 518x518 (BiomedCLIP format)
   - Normalize with ImageNet statistics
   - Extract attention regions with edge detection

4. **Extract Clinical Features**
   - Extract structured clinical features (vitals, demographics)
   - Store as tensors for later narrative generation

5. **Create Splits**
   - Train (70%), Val (15%), Test (15%)
   - Save as .pt files (torch format) to output bucket

### Phase 2: Enhanced Pseudo-Note Generation and RAG Integration

1. **Load Phase 1 Outputs**
   - Read train/val/test splits (.pt files)
   - Load preprocessed images and clinical features

2. **Generate Pseudo-Notes**
   - Convert structured clinical data → narrative text
   - Example: `{age: 65, HR: 85}` → `"Patient is a 65 year old M. Vitals: HR: 85bpm..."`
   - Expand medical abbreviations (HTN → hypertension)

3. **RAG Enhancement**
   - Query medical knowledge base with FAISS
   - Retrieve relevant medical knowledge (top-k documents)
   - Augment pseudo-notes with medical context

4. **Text Tokenization**
   - Tokenize enhanced notes with ModernBERT (8192 token context)
   - Generate input_ids and attention_mask tensors

5. **Save Enhanced Data**
   - Add pseudo_note, enhanced_note, enhanced_text_tokens to records
   - Save as *_enhanced.pt files ready for model training

**See [docs/PHASE2_ENHANCED_NOTES.md](docs/PHASE2_ENHANCED_NOTES.md) for detailed Phase 2 documentation**

### Phase 3: Multi-Modal Integration and Final Dataset Preparation

1. **Load Phase 2 Enhanced Outputs**
   - Read *_enhanced.pt files from Phase 2
   - Load all modalities: images, enhanced text, clinical features

2. **Data Quality Validation**
   - Validate all required fields are present
   - Check image tensor shapes (3, 518, 518)
   - Verify enhanced text tokens have input_ids and attention_mask
   - Validate clinical features tensors

3. **Multi-Modal Integration**
   - Combine vision modality (BiomedCLIP input)
   - Integrate text modality (Clinical ModernBERT input)
   - Align clinical features (structured data)
   - Create unified model-ready format

4. **Generate Statistics**
   - Calculate validation rates per split
   - Compute text length statistics
   - Analyze view position distribution
   - Generate comprehensive quality metrics

5. **Save Final Datasets**
   - Save as *_final.pt files ready for model training
   - Generate detailed metadata and quality reports
   - Create comprehensive dataset documentation

**Usage:**
```bash
# Run Phase 3 integration
python src/phase3_integration.py \
  --input-path processed/phase1_output \
  --gcs-bucket bergermimiciv \
  --gcs-project-id YOUR_PROJECT_ID

# Or test with small samples
python src/phase3_integration.py \
  --input-path processed/phase1_output \
  --use-small-sample
```

### Expected Output

After running Phase 1, you'll find these files in your output directory:

```
processed/phase1_output/
├── train_data.pt           # Training set (70% of data)
├── val_data.pt             # Validation set (15% of data)
├── test_data.pt            # Test set (15% of data)
├── train_small.pt          # Small training sample (100 records, optional)
├── val_small.pt            # Small validation sample (100 records, optional)
├── test_small.pt           # Small test sample (100 records, optional)
└── metadata.json           # Dataset metadata and configuration
```

After running Phase 2, enhanced files are added:

```
processed/phase1_output/
├── train_data.pt                  # Original Phase 1 output
├── train_data_enhanced.pt         # Phase 2: With pseudo-notes + RAG
├── val_data.pt                    # Original Phase 1 output
├── val_data_enhanced.pt           # Phase 2: With pseudo-notes + RAG
├── test_data.pt                   # Original Phase 1 output
├── test_data_enhanced.pt          # Phase 2: With pseudo-notes + RAG
└── phase2_metadata.json           # Phase 2 processing metadata
```

After running Phase 3, final integrated files are created:

```
processed/phase1_output/
├── train_data.pt                  # Original Phase 1 output
├── train_data_enhanced.pt         # Phase 2: With pseudo-notes + RAG
├── train_final.pt                 # Phase 3: Final model-ready dataset
├── val_data.pt                    # Original Phase 1 output
├── val_data_enhanced.pt           # Phase 2: With pseudo-notes + RAG
├── val_final.pt                   # Phase 3: Final model-ready dataset
├── test_data.pt                   # Original Phase 1 output
├── test_data_enhanced.pt          # Phase 2: With pseudo-notes + RAG
├── test_final.pt                  # Phase 3: Final model-ready dataset
├── phase2_metadata.json           # Phase 2 processing metadata
└── phase3_metadata.json           # Phase 3 integration metadata and quality report
```

**Note:** Small sample files (`*_small.pt`) are only created when using the `--create-small-samples` flag in Phase 1. These are perfect for quickly testing Phase 2 and Phase 3 without loading the full dataset.

**Phase 1 record structure** (.pt files):
```python
{
    'subject_id': int,                  # Patient identifier
    'study_id': int,                    # Imaging study identifier
    'dicom_id': str,                    # DICOM image ID
    'image': torch.Tensor,              # Preprocessed image (518x518x3)
    'attention_regions': Dict,          # Attention maps from edge detection
    'text_tokens': Dict,                # Original text tokens (chief complaint)
    'clinical_features': torch.Tensor,  # Structured clinical features
    'retrieved_knowledge': List[str],   # Pre-retrieved knowledge (Phase 1)
    'labels': Dict                      # View position, etc.
}
```

**Phase 2 enhanced record structure** (*_enhanced.pt files):
```python
{
    # All Phase 1 fields (preserved) PLUS:
    'pseudo_note': str,                 # Generated narrative clinical note
    'enhanced_note': str,               # RAG-enhanced note with medical context
    'enhanced_text_tokens': {           # Tokenized for Clinical ModernBERT
        'input_ids': torch.Tensor,
        'attention_mask': torch.Tensor
    },
    'phase2_processed': True            # Processing flag
}
```

**Phase 3 final record structure** (*_final.pt files):
```python
{
    # Identifiers
    'subject_id': int,                  # Patient identifier
    'study_id': int,                    # Imaging study identifier
    'dicom_id': str,                    # DICOM image ID

    # Vision modality (BiomedCLIP input)
    'image': torch.Tensor,              # [3, 518, 518]
    'attention_regions': Dict,          # Attention maps

    # Text modality (Clinical ModernBERT input)
    'text_input_ids': torch.Tensor,     # Tokenized enhanced note
    'text_attention_mask': torch.Tensor,# Attention mask
    'pseudo_note': str,                 # Raw pseudo-note (for analysis)
    'enhanced_note': str,               # Raw enhanced note (for analysis)

    # Clinical features (structured data)
    'clinical_features': torch.Tensor,  # Normalized clinical features

    # Labels and metadata
    'labels': Dict,                     # All labels
    'view_position': str,               # CXR view position
    'retrieved_knowledge': List[str],   # RAG knowledge

    # Processing flags
    'phase1_processed': True,
    'phase2_processed': True,
    'phase3_integrated': True           # Final integration flag
}
```

**metadata.json contains:**
```json
{
    "config": {...},                    # Full preprocessing configuration
    "n_train": 12345,                  # Number of training samples
    "n_val": 2647,                     # Number of validation samples
    "n_test": 2647,                    # Number of test samples
    "total_records": 17639,            # Total records processed
    "stratified": true,                # Whether splits are stratified by class
    "small_samples_created": true,     # Whether small samples were created
    "small_sample_size": 100           # Size of small sample datasets
}
```

## Advanced Usage

### Resume from Intermediate Batches

If batch processing is complete but split creation failed (e.g., OOM error), you can skip directly to combining batches:

```bash
# Resume from intermediate batches (memory-efficient streaming mode)
python src/phase1_preprocess.py \
  --gcs-bucket bergermimiciv \
  --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
  --gcs-project-id YOUR_PROJECT_ID \
  --output-path processed/phase1_with_path_fixes_raw \
  --skip-to-combine \
  --create-small-samples \
  --small-sample-size 100
```

**Key arguments:**
- `--skip-to-combine`: Skip batch processing, load intermediate batches and create splits
- `--create-small-samples`: Generate small sample datasets for Phase 2 testing
- `--small-sample-size N`: Number of records in small samples (default: 100)

### Memory-Efficient Streaming Mode

The pipeline now uses streaming processing for split creation, allowing it to handle 100K+ records on machines with as little as 7.5GB RAM:

- **Counts records** without loading full data
- **Extracts labels** for stratified splitting (streaming)
- **Writes splits** in chunks (1000 records at a time)
- **Combines chunks** into final files with automatic cleanup

This means you can run the full pipeline on budget-friendly VM instances (n4-standard-2, n4-standard-4) without OOM issues.

### Stratified Splitting

The pipeline automatically creates stratified splits to ensure even distribution of classes across train/val/test:

- Uses disease labels if available
- Falls back to subject_id-based pseudo-random stratification
- Maintains 70/15/15 split ratios within each class group
- Logs split statistics for verification

### Small Sample Datasets

Create small versions of train/val/test for rapid Phase 2 development:

```bash
python src/phase1_preprocess.py \
  --gcs-bucket bergermimiciv \
  --output-path processed/phase1_preprocess \
  --create-small-samples \
  --small-sample-size 200  # 200 records per split
```

**Use cases:**
- Test Phase 2 model architecture quickly
- Debug data loading pipelines
- Verify training loop works end-to-end
- Rapid iteration during development

### Phase 1b: Diagnosis Leakage Filtering (Optional)

To prevent data leakage, you can filter the pseudo-notes to remove diagnosis-related information:

```bash
# Apply leakage filtering to preprocessed data
python src/apply_leakage_filter.py \
  --gcs-bucket bergermimiciv \
  --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
  --gcs-project-id YOUR_PROJECT_ID \
  --input-path processed/phase1_preprocess \
  --output-path processed/phase1_filtered \
  --aggressive
```

**What it does:**
- Loads CheXpert labels for each X-ray
- Removes diagnosis-related terms from pseudo-notes using regex patterns
- Filters radiological language ("chest x-ray shows...")
- Removes diagnostic medications and lab values
- Validates that diagnosis information is removed

**Note:** The leakage filter works via regex pattern matching (no additional dependencies required). Installing spaCy models (see Prerequisites above) enables optional entity extraction features, but is not required for core functionality.

**Output:** Filtered train/val/test splits in `processed/phase1_filtered/` with:
- `enhanced_note` → diagnosis information removed
- `positive_findings` → CheXpert labels preserved for training
- `filter_stats` → metrics on what was filtered

## Configuration

### Local Mode

```python
config = DataConfig()
config.use_gcs = False
config.mimic_cxr_path = "~/MIMIC_Data/physionet.org/files/mimic-cxr-jpg/2.1.0"
config.mimic_iv_path = "~/MIMIC_Data/physionet.org/files/mimiciv/3.1"
config.mimic_ed_path = "~/MIMIC_Data/physionet.org/files/mimic-iv-ed/2.2"
```

### GCS Mode

```python
config = DataConfig()
config.use_gcs = True
config.gcs_bucket = "bergermimiciv"  # Your bucket
config.gcs_cxr_bucket = "mimic-cxr-jpg-2.1.0.physionet.org"  # PhysioNet's bucket
config.output_gcs_bucket = "bergermimiciv"  # Output bucket
config.mimic_cxr_path = "physionet.org/files/mimic-cxr-jpg/2.1.0"
config.mimic_iv_path = "physionet.org/files/mimiciv/3.1"
config.mimic_ed_path = "physionet.org/files/mimic-iv-ed/2.2"
```

## Important Notes

### MIMIC-IV vs MIMIC-IV-ED

**MIMIC-IV-ED is a separate dataset!**
- MIMIC-IV: Hospital admissions, ICU stays
- MIMIC-IV-ED: Emergency department visits
- They must be downloaded separately from PhysioNet
- ED data lives in `mimic-iv-ed/2.2/ed/`, NOT inside `mimiciv/3.1/`

### Multi-Bucket Support

The pipeline automatically routes data to the correct bucket:
- **Bucket** (`bergermimiciv`): MIMIC-IV, MIMIC-IV-ED, metadata, outputs
- **PhysioNet's bucket** (`mimic-cxr-jpg-2.1.0.physionet.org`): MIMIC-CXR images

This saves 500+ GB of storage and hours of upload time!

## Cost Estimates (Google Cloud)

### Storage
- Your bucket: 10-20 GB → $0.20-0.40/month
- PhysioNet's bucket: 500+ GB → **FREE** (you're reading, not storing)

### Compute (Preprocessing)
- n1-standard-8 + T4 GPU: ~$0.73/hour
- Full preprocessing: ~8-12 hours = $6-9

**Total**: <$10 for complete preprocessing!

### Cost Optimization
- Use preemptible VMs (70% cheaper)
- Stop VM when not in use
- Use same region for bucket and VM (no egress fees)

## Testing

### Unit Tests

```bash
# Test data loading
python src/test_phase1_local.py --num-samples 5

# Test with images
python src/test_phase1_local.py --num-samples 10 --test-images --num-images 3
```

### Integration Tests

```bash
# Run on small subset on GCS
python src/phase1_preprocess.py \
  --gcs-bucket bergermimiciv \
  --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
  --gcs-project-id YOUR_PROJECT_ID \
  --image-size 224 \
  --max-text-length 512
```

## Troubleshooting

### "DefaultCredentialsError: Your default credentials were not found"

**Problem:** Google Cloud authentication not configured.

**Solution:**
```bash
# Set up Application Default Credentials
gcloud auth application-default login

# Set your project (required for requester pays)
gcloud config set project YOUR_PROJECT_ID

# Verify
gcloud auth list
```

### "BadRequestException: Bucket is a requester pays bucket but no user project provided"

**Problem:** PhysioNet bucket requires your project ID for billing.

**Solution:** Always include `--gcs-project-id` when using the PhysioNet bucket:
```bash
python src/phase1_preprocess.py \
  --gcs-bucket bergermimiciv \
  --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
  --gcs-project-id YOUR_PROJECT_ID
```

### "No ED stays available" or "Could not find edstays"
- Check that MIMIC-IV-ED is in `mimic-iv-ed/2.2/ed/`, not `mimiciv/3.1/ed/`
- Verify `mimic_ed_path` is set correctly
- The pipeline supports both `.csv.gz` and `.csv` file formats - no need to compress/decompress

### "AccessDeniedException" on MIMIC-CXR bucket
- Ensure you have PhysioNet MIMIC-CXR access
- Authenticate: `gcloud auth application-default login`

### "Out of memory"
- Reduce `--image-size` (default: 518 → try 224)
- Reduce `--max-text-length` (default: 8192 → try 512)
- Use larger VM instance

See [LOCAL_TESTING.md](LOCAL_TESTING.md) for more troubleshooting tips.

## Documentation

- **[DEPLOYMENT_QUICKSTART.md](DEPLOYMENT_QUICKSTART.md)** - Quick deployment reference
- **[LOCAL_TESTING.md](LOCAL_TESTING.md)** - Test locally before cloud deployment
## Requirements

### Data Access
- PhysioNet credentialing for MIMIC datasets
- MIMIC-IV 3.1 access
- MIMIC-IV-ED 2.2 access
- MIMIC-CXR-JPG 2.1.0 access

### Python Dependencies
```
# Core ML libraries
pandas>=2.0.0
numpy>=1.24.0,<2.0  # NumPy 2.x not yet fully supported
torch>=2.0.0
torchvision>=0.15.0
pytorch-lightning>=2.0.0

# Model encoders
transformers>=4.30.0  # For text encoders (ModernBERT, BiomedBERT)
open-clip-torch>=2.20.0  # For BiomedCLIP vision encoder
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0

# Image processing
pillow>=10.0.0
opencv-python-headless>=4.8.0

# Cloud and utilities
google-cloud-storage>=2.10.0  # For GCS support
tqdm>=4.65.0
```

### System Requirements (Local Testing)
- 8+ GB RAM
- 20+ GB disk space (for small samples)
- Python 3.9-3.11

### System Requirements (Cloud)
- Compute Engine: n1-standard-4 (4 vCPUs, 15 GB RAM) - minimum recommended
- GPU: NVIDIA Tesla T4 (optional, for faster preprocessing)
- Storage: 200 GB boot disk

## Acknowledgments

This project builds upon:
- **MDF-Net**: Multi-modal Deep Fusion Network for medical diagnosis
- **BiomedCLIP**: Biomedical vision-language model
- **ModernBERT**: Fast, efficient language model
- **MIMIC Datasets**: Open-source medical datasets from MIT LCP

## License

This project follows the MIMIC data usage agreements. MIMIC data is available under the PhysioNet Credentialed Health Data License.

## Citation

If you use this preprocessing pipeline, please cite:

```bibtex
@misc{mimic-iv-cxr-ed-preprocessing,
  title={MIMIC-IV-CXR-ED Preprocessing Pipeline for Multi-Modal Anomaly Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/cam-berger/MIMIC-IV-CXR-ED-Anomaly-Detection}
}
```

And cite the original MIMIC datasets:
- [MIMIC-IV](https://physionet.org/content/mimiciv/)
- [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/)
- [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/)

## Contact

For questions or issues:
- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review troubleshooting in [LOCAL_TESTING.md](LOCAL_TESTING.md)

---

**Last Updated**: 2025-11-15
**Status**: Production Ready | Official MIMIC-CXR Splits (377K studies) | Enhanced RAG Adapter | BiomedCLIP + ModernBERT | 354M Parameters | Multi-Format Support | Comprehensive Testing | 4 Critical Bugs Fixed | Training Pipeline Validated
