# MIMIC Multi-Modal Dataset

HYPOTHESIS: Context-aware knowledge augmentation of clinical notes, when fused with visual features through cross-modal attention, will improve both the accuracy and interpretability of chest X-ray abnormality detection compared to models using raw clinical notes or images alone, with the improvement being most significant for rare conditions and complex multi-abnormality cases.

## Recent Improvements (October 2025)

**Major Performance & Reliability Enhancements**:
- **20-40x Faster Processing**: Batch downloading with parallel workers reduces processing time from 12 days to 7-15 hours
- **Fixed Critical Path Bug**: Corrected MIMIC-CXR image path construction (8-digit padding) - was causing 100% image lookup failures
- **Smart Caching**: Eliminates duplicate downloads (previously downloading each image twice per record)
- **Robust Error Handling**: Pipeline continues past failed images with detailed diagnostics
- **Improved Temporal Matching**: Fixed StudyDate+StudyTime parsing, increased matches from 0 to 107,949 records
- **Detailed Statistics Logging**: Real-time visibility into joining process (successful matches, missing images, temporal mismatches)
- **Memory-Efficient Streaming**: Process 100K+ records on 7.5GB RAM without OOM crashes
- **Stratified Splitting**: Evenly distribute classes across train/val/test splits
- **Resume Capability**: Skip completed batch processing and jump directly to dataset creation
- **Small Sample Datasets**: Automatically generate small test versions for rapid Phase 2 development

### Technical Documentation
- **[OPUS Fixes Verification](OPUS_FIXES_VERIFICATION.md)**: Complete verification of all 5 critical issues identified and fixed
- **[OOM Fix Summary](SESSION_OOM_FIX_SUMMARY.md)**: Detailed analysis and solution for memory management issues
- All fixes verified and deployed to production (VM: mimic-preprocessing-20251026-092532)

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

## Architecture

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

### 2. Local Testing (Recommended First)

Test on your laptop before cloud deployment:

```bash
# Run local tests with small sample
python src/test_phase1_local.py \
  --mimic-path ~/Documents/Portfolio/MIMIC_Data/physionet.org/files \
  --num-samples 10

# See LOCAL_TESTING.md for detailed guide
```

### 3. Google Cloud Setup

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
│   ├── GCP_DEPLOYMENT.md                # Complete GCP deployment guide
│   ├── GCS_SETUP.md                     # Google Cloud setup guide
│   ├── ARCHITECTURE.md                  # System architecture
│   ├── IMAGE_DOWNLOAD_GUIDE.md          # MIMIC-CXR download guide
│   └── PHASE2_ENHANCED_NOTES.md         # Phase 2 documentation
├── LOCAL_TESTING.md                     # Local testing guide
├── DEPLOYMENT_QUICKSTART.md             # Quick deployment reference
├── PHASE2_REFACTORING_SUMMARY.md        # Phase 2 refactoring details
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

**Note:** Small sample files (`*_small.pt`) are only created when using the `--create-small-samples` flag in Phase 1. These are perfect for quickly testing Phase 2 without loading the full dataset.

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
pandas>=2.0.0
numpy>=1.24.0,<2.0  # NumPy 2.x not yet fully supported
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0
pillow>=10.0.0
opencv-python-headless>=4.8.0
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

**Last Updated**: 2025-10-28
**Status**: Phase 1 Complete | Memory-Efficient Streaming | Stratified Splitting | Resume Capability | Small Sample Datasets | 107,949+ Records Matched | Runs on 7.5GB RAM
