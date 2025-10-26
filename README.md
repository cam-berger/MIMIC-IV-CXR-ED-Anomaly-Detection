# MIMIC Multi-Modal Dataset 
HYPOTHESIS: Context-aware knowledge augmentation of clinical notes, when fused with visual features through cross-modal attention, will improve both the accuracy and interpretability of chest X-ray abnormality detection compared to models using raw clinical notes or images alone, with the improvement being most significant for rare conditions and complex multi-abnormality cases.
>>>>>>> main

## Overview

This project implements Phase 1 data preprocessing for an Enhanced MDF-Net model that combines:
- **BiomedCLIP-CXR** vision encoder for chest X-ray analysis
- **Clinical ModernBERT** text encoder for clinical notes
- **RAG knowledge enhancement** for medical context
- **Cross-attention fusion** for multimodal integration

The pipeline links MIMIC-CXR chest X-rays with MIMIC-IV-ED emergency department data to create a unified dataset for medical abnormality detection.

## Key Features

- ✅ **Multi-Bucket GCS Support**: Seamlessly works with your data bucket + PhysioNet's public MIMIC-CXR bucket
- ✅ **Pseudo-Note Generation**: Converts structured clinical data into narrative text for LLM processing
- ✅ **Temporal Alignment**: Links chest X-rays to ED visits within 24-hour windows
- ✅ **Local Testing**: Test preprocessing locally before cloud deployment
- ✅ **Scalable**: Handles 377K+ chest X-rays and 425K+ ED visits
- ✅ **Cloud-Native**: Optimized for Google Cloud Platform (Compute Engine, Cloud Storage)

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

2. **MIMIC-IV-ED 2.2** - Emergency Department data (separate dataset!)
   - `mimic-iv-ed/2.2/ed/` - ED stays, triage, vital signs
   - Size: ~2 GB

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

# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Optional: Install spaCy for enhanced leakage filtering (not required)
# pip install spacy>=3.5.0 scispacy>=0.5.0
# python -m spacy download en_core_web_sm
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz
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
gsutil -m cp -r ~/MIMIC_Data/physionet.org/files/mimiciv \
  gs://bergermimiciv/physionet.org/files/

gsutil -m cp -r ~/MIMIC_Data/physionet.org/files/mimic-iv-ed \
  gs://bergermimiciv/physionet.org/files/

# OPTION A: Automated Deployment (Recommended)
# Automatically creates VM, installs dependencies, runs full pipeline, auto-shuts down
bash scripts/deploy_gcp.sh YOUR_PROJECT_ID bergermimiciv

# OPTION B: Manual Deployment
# Create VM and run pipeline manually
gcloud compute instances create mimic-preprocessing \
  --zone=us-central1-a \
  --machine-type=n1-highmem-8 \
  --boot-disk-size=200GB \
  --scopes=cloud-platform

# SSH and run full pipeline (preprocessing + leakage filtering)
gcloud compute ssh mimic-preprocessing --zone=us-central1-a
python src/run_full_pipeline.py \
  --gcs-bucket bergermimiciv \
  --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
  --gcs-project-id YOUR_PROJECT_ID \
  --mimic-iv-path physionet.org/files/mimiciv/3.1 \
  --mimic-ed-path physionet.org/files/mimic-iv-ed/2.2 \
  --output-path processed/phase1_final \
  --aggressive-filtering

# See docs/GCP_DEPLOYMENT.md for complete deployment guide
```

## Project Structure

```
.
├── src/
│   ├── phase1_preprocess.py        # Main preprocessing pipeline
│   ├── phase1_stay_identification.py  # ED stay linking
│   └── test_phase1_local.py        # Local testing script
├── notebooks/
│   └── phase1_preprocess.ipynb     # Interactive notebook
├── docs/
│   ├── GCS_SETUP.md                # Google Cloud setup guide
│   ├── ARCHITECTURE.md             # System architecture
│   └── IMAGE_DOWNLOAD_GUIDE.md     # MIMIC-CXR download guide
├── LOCAL_TESTING.md                # Local testing guide
├── PSEUDO_NOTES_EXPLAINED.md       # Pseudo-note generation explained
└── README.md                       # This file
```

## How It Works

### Phase 1: Data Preprocessing

1. **Load Metadata**
   - MIMIC-CXR: Chest X-ray metadata (from PhysioNet's bucket)
   - MIMIC-IV-ED: ED stays, triage, vital signs (from your bucket)

2. **Join Multimodal Data**
   - Match chest X-rays to ED visits using temporal alignment (±24 hours)
   - Link patient demographics, vital signs, medications

3. **Create Pseudo-Notes**
   - Convert structured data → narrative text
   - Example: `{age: 65, HR: 85}` → `"Patient is a 65 year old M. Vitals: HR: 85bpm..."`

4. **Image Preprocessing**
   - Resize to 518x518 (BiomedCLIP format)
   - Normalize with ImageNet statistics
   - Extract attention regions with edge detection

5. **Text Preprocessing**
   - Tokenize with ModernBERT (8192 token context)
   - Expand medical abbreviations
   - Extract medical entities for RAG

6. **RAG Enhancement**
   - Query medical knowledge base
   - Append relevant context to pseudo-notes

7. **Create Splits**
   - Train (70%), Val (15%), Test (15%)
   - Save as pickle files to output bucket

### Expected Output

After running the preprocessing pipeline, you'll find these files in your output directory (`processed/phase1_preprocess/` or `gs://bergermimiciv/processed/phase1_preprocess/`):

```
processed/phase1_preprocess/
├── train_data.pkl          # Training set (70% of data)
├── val_data.pkl            # Validation set (15% of data)
├── test_data.pkl           # Test set (15% of data)
└── metadata.json           # Dataset metadata and configuration
```

**Each record in the pickle files contains:**
```python
{
    'subject_id': int,              # Patient identifier
    'study_id': int,                # Imaging study identifier
    'image_tensor': torch.Tensor,   # Preprocessed image (518x518x3)
    'attention_mask': np.array,     # Attention regions from edge detection
    'text_input_ids': List[int],    # Tokenized text (ModernBERT)
    'text_attention_mask': List[int],  # Text attention mask
    'enhanced_note': str,           # RAG-enhanced pseudo-note
    'attention_segments': Dict,     # Cross-attention preparation
    'clinical_data': Dict,          # Original clinical features
    'labels': Dict                  # Disease labels, bboxes, severity scores
}
```

**metadata.json contains:**
```json
{
    "config": {...},              # Full preprocessing configuration
    "n_train": 12345,            # Number of training samples
    "n_val": 2647,               # Number of validation samples
    "n_test": 2647,              # Number of test samples
    "total_records": 17639       # Total records processed
}
```

See [PSEUDO_NOTES_EXPLAINED.md](PSEUDO_NOTES_EXPLAINED.md) for detailed explanation of pseudo-note generation.

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
- **Your bucket** (`bergermimiciv`): MIMIC-IV, MIMIC-IV-ED, metadata, outputs
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

### "No ED stays available"
- Check that MIMIC-IV-ED is in `mimic-iv-ed/2.2/ed/`, not `mimiciv/3.1/ed/`
- Verify `mimic_ed_path` is set correctly

### "AccessDeniedException" on MIMIC-CXR bucket
- Ensure you have PhysioNet MIMIC-CXR access
- Authenticate: `gcloud auth application-default login`

### "Out of memory"
- Reduce `--image-size` (default: 518 → try 224)
- Reduce `--max-text-length` (default: 8192 → try 512)
- Use larger VM instance

See [LOCAL_TESTING.md](LOCAL_TESTING.md) for more troubleshooting tips.

## Documentation

- **[LOCAL_TESTING.md](LOCAL_TESTING.md)** - Test locally before cloud deployment
- **[docs/GCS_SETUP.md](docs/GCS_SETUP.md)** - Complete Google Cloud setup guide
- **[PSEUDO_NOTES_EXPLAINED.md](PSEUDO_NOTES_EXPLAINED.md)** - How pseudo-notes work
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture details

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
- Compute Engine: n1-standard-8 (8 vCPUs, 30 GB RAM)
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

**Last Updated**: 2025-10-22
**Status**: Phase 1 Complete ✅ | Multi-Bucket GCS Support ✅ | Local Testing ✅
