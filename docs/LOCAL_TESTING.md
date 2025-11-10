# Local Testing Guide for Phase 1 Preprocessing

This guide helps you test Phase 1 preprocessing functionality on your resource-limited laptop before deploying to Google Cloud Platform.

## Prerequisites

### What You Have
- MIMIC-IV dataset (CSV files) - Hospital data in `mimiciv/3.1/`
- MIMIC-IV-ED dataset (CSV files) - Emergency Department data in `mimic-iv-ed/2.2/` (separate dataset!)
- MIMIC-CXR-JPG metadata (without images)

### What You Need to Install

```bash
# Activate your conda environment
conda activate ml_env  # Or your preferred environment

# Install required packages (if not already installed)
pip install pandas numpy tqdm pillow opencv-python-headless
pip install torch torchvision  # If not already installed
pip install transformers sentence-transformers faiss-cpu
```

## Testing Options

### Option 1: Quick Metadata-Only Test (Recommended First)

This tests data loading and joining **without any images**. Perfect for validating your setup.

```bash
# Use the lightweight stay identification script
python src/phase1_stay_identification.py \
  --local \
  --local-path ~/Documents/Portfolio/MIMIC_Data/physionet.org/files \
  --num-subjects 5 \
  --num-records 20
```

**What this does:**
- Loads MIMIC-CXR metadata (CSV only)
- Loads MIMIC-IV-ED data (CSV only)
- Links records based on subject IDs and timestamps
- Creates sample linked records
- **Estimated time:** 1-2 minutes
- **Memory usage:** < 1 GB

### Option 2: Comprehensive Test Without Images

Test all preprocessing components except actual image loading.

```bash
python src/test_phase1_local.py \
  --mimic-path ~/Documents/Portfolio/MIMIC_Data/physionet.org/files \
  --num-samples 10
```

**What this tests:**
1. Metadata loading (MIMIC-CXR, MIMIC-IV-ED)
2. Pseudo-note creation from clinical data
3. Text preprocessing and tokenization
4. Image preprocessing pipeline setup (without loading images)

**Estimated time:** 2-5 minutes (includes downloading ModernBERT tokenizer on first run)
**Memory usage:** < 2 GB

### Option 3: Full Test With Sample Images

If you want to test with actual images, first download a few sample images:

#### Download Sample Images

```bash
# Create the directory structure for one sample patient
# Example: subject_id=10000032, study_id=50414267

# The directory structure should be:
# mimic-cxr-jpg/2.1.0/files/p10/p10000032/s50414267/

# You can download a few .jpg files from PhysioNet for this study
# Place them in the appropriate directory
```

Then run:

```bash
python src/test_phase1_local.py \
  --mimic-path ~/Documents/Portfolio/MIMIC_Data/physionet.org/files \
  --num-samples 10 \
  --test-images \
  --num-images 3
```

**What this tests:**
1. Everything from Option 2, plus:
2. Actual image loading and preprocessing
3. Image tensor creation with BiomedCLIP transforms

**Estimated time:** 3-10 minutes
**Memory usage:** 2-4 GB (depending on number of images)

## Understanding Test Output

### Successful Test Output

```
==================================================================
TEST 1: Loading Metadata (No Images)
==================================================================
INFO - Loaded 377,110 CXR metadata records
Loaded CXR metadata
Loaded 6 ED tables
  - edstays: 425,087 records
  - triage: 425,087 records
  ...

==================================================================
TEST SUMMARY
==================================================================
data_loading........................ PASS
pseudo_notes........................ PASS
text_preprocessing.................. PASS
image_setup......................... PASS

Passed: 4/4

All tests passed! Your Phase 1 setup is working correctly.
```

### Common Issues

#### Issue 1: "File not found" or "No ED stays available" errors

**Problem:** Paths to MIMIC datasets are incorrect, or MIMIC-IV-ED is in a separate directory

**Solution:**
```bash
# Check your actual MIMIC data structure
ls ~/Documents/Portfolio/MIMIC_Data/physionet.org/files/

# You should see:
# - mimiciv/3.1/              (MIMIC-IV hospital data)
# - mimic-iv-ed/2.2/          (MIMIC-IV-ED emergency dept data - SEPARATE!)
# - mimic-cxr-jpg/2.1.0/      (MIMIC-CXR chest X-rays)
# - reflacx/                   (optional eye-gaze annotations)

# IMPORTANT: MIMIC-IV-ED is a separate dataset from MIMIC-IV!
# The code now handles this correctly by using a separate mimic_ed_path

# Verify ED data is in the right place:
ls ~/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-iv-ed/2.2/ed/
# Should show: edstays.csv, triage.csv, vitalsign.csv, etc.
```

#### Issue 2: "ModuleNotFoundError"

**Problem:** Missing Python packages

**Solution:**
```bash
# Install missing package
pip install <package-name>

# Or install all requirements
pip install -r requirements.txt  # If you have one
```

#### Issue 3: "Out of memory"

**Problem:** Processing too many records

**Solution:**
```bash
# Reduce the number of samples
python test_phase1_local.py \
  --num-samples 5  # Reduce from 10 to 5

# Or use the even lighter script
python src/phase1_stay_identification.py \
  --local \
  --num-subjects 3 \
  --num-records 10
```

#### Issue 4: Image files not found

**Problem:** You haven't downloaded any images (this is OK!)

**Solution:**
- Skip the `--test-images` flag for now
- Or download a few sample images from PhysioNet
- The metadata-only tests will still validate your preprocessing logic

## What Gets Tested

### Data Loading & Joining
- MIMIC-CXR metadata loading (frontal views filtering)
- MIMIC-IV-ED data loading (edstays, triage, vitalsign tables)
- Timestamp-based joining (within 24-hour windows)
- Subject ID matching across datasets

### Clinical Text Processing
- Pseudo-note creation from structured data
- Medical abbreviation expansion
- Text cleaning and normalization
- Tokenization with ModernBERT
- Medical entity extraction

### Image Processing (if images available)
- Image loading and conversion
- Resizing to target dimensions
- Normalization with BiomedCLIP parameters
- Tensor creation for model input

## Resource Requirements by Test Type

| Test Type | Time | RAM | Disk I/O | Downloads |
|-----------|------|-----|----------|-----------|
| Option 1 (Metadata only) | 1-2 min | < 1 GB | Low | None |
| Option 2 (No images) | 2-5 min | < 2 GB | Low | ModernBERT tokenizer (~500 MB, first run only) |
| Option 3 (With 3 images) | 3-10 min | 2-4 GB | Medium | ModernBERT + images |
| Option 3 (With 50 images) | 10-30 min | 4-8 GB | High | ModernBERT + images |

## Next Steps After Successful Testing

1. **Review test output** - Check the `test_output/` directory for sample results

2. **Test with more samples** - Gradually increase `--num-samples` to see how your laptop handles it:
   ```bash
   python test_phase1_local.py --num-samples 50
   python test_phase1_local.py --num-samples 100
   ```

3. **Download sample images** - If you want to test image processing:
   - Select 5-10 studies from the test output
   - Download corresponding .jpg files from PhysioNet
   - Place in correct directory structure

4. **Run full preprocessing on subset** - Once everything works:
   ```bash
   python src/phase1_preprocess.py \
     --mimic-cxr-path ~/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-cxr-jpg/2.1.0 \
     --mimic-iv-path ~/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimiciv/3.1 \
     --output-path ./preprocessed_output \
     --image-size 224 \
     --max-text-length 512
   ```

5. **Deploy to GCP** - Once local testing is successful, follow the [DEPLOYMENT_QUICKSTART.md](DEPLOYMENT_QUICKSTART.md) or [docs/GCP_DEPLOYMENT.md](docs/GCP_DEPLOYMENT.md) guide

## Monitoring Resource Usage

### Monitor Memory Usage (Linux/Mac)

```bash
# In another terminal, watch memory usage
watch -n 1 'ps aux | grep python | grep phase1'

# Or use htop
htop
```

### Monitor Disk I/O

```bash
# Check how much data is being read
iostat -x 2  # Update every 2 seconds
```

### Kill If Needed

```bash
# If test hangs or uses too much memory
pkill -f phase1
```

## FAQ

**Q: Do I need to download all MIMIC-CXR images?**
A: No! You can test the full preprocessing logic with just metadata. Images are only needed for the final image preprocessing step.

**Q: How much data should I test with locally?**
A: Start with 10-20 samples. If your laptop handles it well, try 100-500. For full preprocessing, use AWS.

**Q: Can I test without REFLACX?**
A: Yes! REFLACX is optional. The scripts will work without it.

**Q: What if I want to test specific disease labels?**
A: Look at the CXR labels in the metadata and filter for specific findings:
```python
# In your own test script
metadata = pd.read_csv('.../mimic-cxr-2.0.0-chexpert.csv.gz')
pneumonia_cases = metadata[metadata['Pneumonia'] == 1.0]
```

**Q: How do I know my preprocessing is correct?**
A: The test scripts validate:
- Data shapes and types are correct
- No crashes during processing
- Output files are created
- Text tokenization produces expected shapes
- Images are loaded and transformed correctly

## Troubleshooting

### Detailed Logs

Enable more detailed logging:
```python
# Add to test script
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Test Individual Components

You can import and test components individually:
```python
from src.phase1_preprocess import TextPreprocessor, DataConfig

config = DataConfig()
config.max_text_length = 512

text_proc = TextPreprocessor(config)
sample_text = "Patient with chest pain"
result = text_proc.preprocess_text(sample_text)
print(result['input_ids'].shape)
```

## Contact

If you encounter issues not covered here, check:
1. Error messages in console output
2. Paths are correctly set
3. All dependencies are installed
4. MIMIC data files are not corrupted
