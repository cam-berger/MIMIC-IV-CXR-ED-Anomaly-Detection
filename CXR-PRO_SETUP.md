# CXR-PRO Dataset Setup Guide

This project has been configured to use the **CXR-PRO** dataset instead of MIMIC-CXR.

## What is CXR-PRO?

CXR-PRO is an enhanced version of MIMIC-CXR where references to non-existent prior radiology reports have been removed using machine learning. This improves the quality of radiology report generation systems.

**Key differences from MIMIC-CXR:**
- Uses HDF5 format (`cxr.h5`) for images instead of individual JPG files
- Provides train/test split CSV files with impressions
- Omits hallucinated references to priors
- Contains 371,951 training reports and 2,188 test reports

## Setup Steps

### 1. Download CXR-PRO from PhysioNet

1. Visit https://physionet.org/content/cxr-pro/1.0.0/
2. Ensure you have credentialed access (sign the data use agreement)
3. Download the following files:
   - `cxr.h5` (chest radiographs in HDF5 format)
   - `mimic_train_impressions.csv` (371,951 reports)
   - `mimic_test_impressions.csv` (2,188 reports)

### 2. Upload to S3

Run the upload script:

```bash
./scripts/upload_cxr_pro.sh
```

This will:
- Prompt you for the local download path
- Verify files exist
- Upload to `s3://bergermimiciv/cxr-pro/1.0.0/`

**Manual upload (alternative):**

```bash
# Upload HDF5 file
aws s3 cp ~/Downloads/cxr-pro/cxr.h5 \
  s3://bergermimiciv/cxr-pro/1.0.0/cxr.h5

# Upload CSV files
aws s3 cp ~/Downloads/cxr-pro/mimic_train_impressions.csv \
  s3://bergermimiciv/cxr-pro/1.0.0/mimic_train_impressions.csv

aws s3 cp ~/Downloads/cxr-pro/mimic_test_impressions.csv \
  s3://bergermimiciv/cxr-pro/1.0.0/mimic_test_impressions.csv
```

### 3. Verify Upload

```bash
aws s3 ls s3://bergermimiciv/cxr-pro/1.0.0/
```

Expected output:
```
cxr.h5
mimic_test_impressions.csv
mimic_train_impressions.csv
```

## Configuration Changes

The following files have been updated to use CXR-PRO:

### `config/aws_config.yaml`
- Updated `data_paths.mimic_cxr` section
- Added custom bucket override: `bergermimiciv`
- Changed paths to point to CXR-PRO files

### `src/phase1_stay_identification.py`
- Added support for custom bucket override
- Added logic to handle non-compressed CSV files
- Auto-detects CXR-PRO format

## Data Structure

### CXR-PRO CSV Format

The `mimic_train_impressions.csv` and `mimic_test_impressions.csv` files contain:

**Key columns:**
- `dicom_id` - DICOM identifier for the image
- `study_id` - Study identifier
- `subject_id` - Patient identifier (links to MIMIC-IV)
- Impression text (with prior references removed)

### HDF5 File Structure

The `cxr.h5` file contains the actual chest X-ray images in HDF5 format. You may need additional code to extract images from HDF5 if needed.

## Running the Pipeline with CXR-PRO

Everything works the same way, but now uses CXR-PRO data:

```bash
# Run Phase 1 (Stay Identification)
python scripts/run_local.py --mode local --phase 1

# Run Phase 2 (Clinical Extraction)
python scripts/run_local.py --mode local --phase 2

# Run Phase 3 (Integration)
python scripts/run_local.py --mode local --phase 3
```

## Data Sources Summary

| Dataset | Source | Location |
|---------|--------|----------|
| **CXR-PRO** | Your S3 bucket | `s3://bergermimiciv/cxr-pro/1.0.0/` |
| **MIMIC-ED** | Your S3 bucket | `s3://bergermimiciv/mimic-iv-ed/2.2/` |
| **MIMIC-IV** | AWS Open Data | `s3://physionet-open/files/mimiciv/2.2/hosp/` |
| **MIMIC-ICU** | AWS Open Data | `s3://physionet-open/files/mimiciv/2.2/icu/` |

## Troubleshooting

### Issue: "No module named h5py"

If you need to read the HDF5 file directly:

```bash
pip install h5py
```

### Issue: Missing dicom_id column

CXR-PRO uses `dicom_id`, `study_id`, and `subject_id`. Ensure your code references these exact column names.

### Issue: Upload takes too long

The `cxr.h5` file is large. Consider:
- Using AWS S3 multipart upload for faster transfers
- Uploading from an EC2 instance in the same region for faster speeds
- Checking your internet connection

## References

- CXR-PRO Paper: "Improving Radiology Report Generation Systems By Removing Hallucinated References to Non-existent Priors"
- Dataset: https://physionet.org/content/cxr-pro/1.0.0/
- MIMIC-IV: https://physionet.org/content/mimiciv/2.2/
