# MIMIC-CXR Download Guide

This guide explains how to download MIMIC-CXR data directly to your S3 bucket without filling up your local disk.

## Prerequisites

1. **PhysioNet Account with Credentialed Access**
   - Create account at https://physionet.org/register/
   - Complete CITI training (Data or Specimens Only Research)
   - Request credentialed access to MIMIC-CXR-JPG at https://physionet.org/content/mimic-cxr-jpg/

2. **AWS Account and S3 Bucket**
   - S3 bucket created: `bergermimiciv`
   - AWS CLI configured with credentials

## Option 1: Stream to S3 (Best - No Local Storage)

This Python script streams files directly from PhysioNet to S3 without using local disk space.

### Install Dependencies

```bash
pip install requests boto3 tqdm
```

### Run the Script

```bash
python scripts/stream_mimic_to_s3.py
```

The script will:
1. Prompt for your PhysioNet username and password
2. Test your credentials before downloading
3. Stream metadata files directly to S3 (~15MB total)

**Alternative:** Set credentials as environment variables (more secure):
```bash
export PHYSIONET_USERNAME='your_username'
export PHYSIONET_PASSWORD='your_password'
python scripts/stream_mimic_to_s3.py
```

## Option 2: Bash Script with Local Temp Storage

Downloads to temp directory, uploads to S3, then deletes local copy.

```bash
./scripts/download_mimic_cxr_direct_to_s3.sh
```

This script will:
1. Download metadata files first
2. Ask if you want the full dataset (reports, etc.)
3. Upload everything to S3
4. Clean up local files

## Option 3: Manual wget + S3 Sync

If you prefer to do it manually:

```bash
# Download to local directory
wget -r -N -c -np \
  --user camberger21 \
  --ask-password \
  https://physionet.org/files/mimic-cxr/2.1.0/

# Sync to S3
aws s3 sync files/mimic-cxr/2.1.0/ \
  s3://bergermimiciv/mimic-cxr/2.1.0/ \
  --exclude "index.html*"

# Clean up
rm -rf files/
```

## What Gets Downloaded

The scripts download the **essential metadata files**:

- `cxr-record-list.csv.gz` (~14MB) - Main metadata with DICOM tags, StudyDate, StudyTime, ViewPosition
- `cxr-study-list.csv.gz` (~2MB) - Study-level metadata

**Optional additional files** (if you run full download):
- Radiology reports (mimic-cxr-reports.zip)
- Provider lists
- Complete directory structure

## About Image Files

⚠️ **IMPORTANT:** The `mimic-cxr` dataset includes both metadata AND images!

### Image Files Details:
- **Size:** ~500 GB
- **Count:** ~377,110 DICOM/JPG files
- **Location:** `files/p10/`, `files/p11/`, ... `files/p19/` subdirectories
- **Organization:** Nested by patient ID

### Do You Need Images?

**Most analysis tasks DON'T need images!** Images are only required for:
- ✅ Computer vision / image analysis
- ✅ Deep learning models on chest X-rays
- ✅ Visual inspection of radiographs

**You DON'T need images for:**
- ❌ Phase 1: Linking CXRs to ED stays (metadata only)
- ❌ Phase 2: Clinical data extraction (metadata only)
- ❌ Most statistical analysis

### Option A: Metadata Only (Recommended)

```bash
python scripts/stream_mimic_to_s3.py
```

Downloads:
- `cxr-record-list.csv.gz` (~14 MB)
- `cxr-study-list.csv.gz` (~2 MB)
- Optional: `mimic-cxr-reports.zip` (~135 MB)

**Total:** ~17 MB (or ~152 MB with reports)

### Option B: Download Images Later (If Needed)

If you discover you need images after all:

```bash
./scripts/download_mimic_images.sh
```

This script will:
1. Warn you about the size (500 GB)
2. Download images with wget
3. Upload to S3 in batches
4. Clean up local copies to save disk space

**Time estimate:** Several hours to days depending on connection speed

## Verify Upload

Check that files were uploaded correctly:

```bash
aws s3 ls s3://bergermimiciv/mimic-cxr/2.1.0/
```

Expected output:
```
2024-XX-XX XX:XX:XX   14234567 cxr-record-list.csv.gz
2024-XX-XX XX:XX:XX    2234567 cxr-study-list.csv.gz
```

## Next Steps

Once metadata is uploaded, you can run the preprocessing pipeline:

```bash
# Phase 1: Link CXRs to ED stays
python -m src.phase1_stay_identification

# Phase 2: Extract clinical data
python -m src.phase2_clinical_extraction

# Phase 3: Integrate data
python -m src.phase3_integration
```

## Troubleshooting

### Authentication Errors (401 - Unauthorized)
**Symptoms:** "Authentication failed" or "Invalid username or password"

**Solutions:**
- Double-check your PhysioNet username (usually email or account name)
- Verify password is correct (try logging in at physionet.org first)
- Make sure your account is active and email is verified
- Clear any cached credentials

### Access Denied (403 - Forbidden)
**Symptoms:** "Access denied" or "You need credentialed access"

**This means you don't have permission to access MIMIC-CXR. To fix:**

1. **Complete CITI Training:**
   - Go to https://physionet.org/settings/training/
   - Complete "Data or Specimens Only Research" course
   - Upload your completion certificate

2. **Sign Data Use Agreement (DUA):**
   - Visit https://physionet.org/content/mimic-cxr/2.1.0/
   - Click "Request Access"
   - Read and sign the Data Use Agreement
   - Wait for approval (usually 1-2 business days)

3. **Verify Access Granted:**
   - Log in to PhysioNet
   - Go to https://physionet.org/content/mimic-cxr/2.1.0/
   - You should see "Download" buttons (not "Request Access")

### AWS Errors
- Verify AWS credentials: `aws sts get-caller-identity`
- Check bucket exists: `aws s3 ls s3://bergermimiciv`
- Verify permissions to write to bucket

### File Not Found (404)
- Version might have changed - check https://physionet.org/content/mimic-cxr/
- Update version number in script if needed

## Storage Costs

Estimated S3 costs (us-west-2 Standard):
- **Metadata only**: ~300MB = ~$0.007/month
- **Metadata + All images**: ~500GB = ~$11.50/month

Consider using S3 Intelligent-Tiering or Glacier for cost savings on image storage.
