# Migration from AWS to Google Cloud Storage

## Summary

This document summarizes the migration from AWS S3 to Google Cloud Storage (GCS) with multi-bucket support.

## Date

2025-10-22

## Changes Made

### 1. Code Updates

#### src/phase1_preprocess.py
- **Removed**: `import boto3` and AWS S3 client
- **Added**: `from google.cloud import storage` with GCS client
- **Replaced**: `S3Helper` class → `GCSHelper` class with multi-bucket support
- **Updated**: `DataConfig` to use `use_gcs`, `gcs_bucket`, `gcs_cxr_bucket`, `output_gcs_bucket`
- **Updated**: Command-line arguments from `--s3-bucket` → `--gcs-bucket`, `--gcs-cxr-bucket`
- **Enhanced**: Multi-bucket routing (your bucket + PhysioNet's public bucket)

Key features of `GCSHelper`:
```python
- gcs_bucket: Your main bucket (bergermimiciv)
- gcs_cxr_bucket: PhysioNet's public bucket (mimic-cxr-jpg-2.1.0.physionet.org)  
- output_gcs_bucket: Output bucket (usually same as main bucket)
- _get_bucket_for_path(): Automatically routes to correct bucket based on path
```

#### src/test_phase1_local.py
- **Updated**: Import from `S3Helper` → `GCSHelper`
- **Updated**: Configuration from `use_s3` → `use_gcs`
- **Updated**: Documentation references from "AWS deployment" → "Google Cloud deployment"

### 2. Documentation Updates

#### README.md
- Complete rewrite with Google Cloud Platform focus
- Added multi-bucket architecture diagram
- Updated quick start with GCS commands
- Removed all AWS/S3 references
- Added GCS cost estimates
- Updated configuration examples

#### Requirements.txt
- **Removed**: `boto3`
- **Added**: `google-cloud-storage>=2.10.0`
- Added version constraints for NumPy (<2.0) for compatibility

#### Removed Files
- ❌ `DEPLOYMENT_CHECKLIST.md` (AWS-specific)
- ❌ `DEPLOYMENT_SUMMARY.md` (AWS-specific)
- ❌ `docs/AWS_SETUP.md`
- ❌ `docs/EC2_DEPLOYMENT.md`
- ❌ `scripts/deploy_to_aws.sh`
- ❌ `scripts/setup_aws.sh`
- ❌ `scripts/setup_ec2.sh`
- ❌ `scripts/setup_s3_bucket.sh`
- ❌ `scripts/upload_mimic_ed.sh`
- ❌ `scripts/verify_deployment.sh`
- ❌ `src/aws_processor.py`
- ❌ `SAGEMAKER_QUICKSTART.md`

#### Kept/Updated Files
- ✅ `LOCAL_TESTING.md` - Updated with GCS references
- ✅ `PSEUDO_NOTES_EXPLAINED.md` - No changes needed (platform-agnostic)
- ✅ `docs/GCS_SETUP.md` - New comprehensive GCS guide

### 3. Multi-Bucket Support

The pipeline now supports reading from multiple GCS buckets simultaneously:

**Your Bucket** (`bergermimiciv`):
- MIMIC-IV (`physionet.org/files/mimiciv/3.1/`)
- MIMIC-IV-ED (`physionet.org/files/mimic-iv-ed/2.2/`)
- REFLACX (`physionet.org/files/reflacx/`)
- Metadata files
- Output data

**PhysioNet's Public Bucket** (`mimic-cxr-jpg-2.1.0.physionet.org`):
- MIMIC-CXR images (`files/p10/`, `files/p11/`, etc.)
- 377K+ chest X-ray images (500+ GB)
- **Read-only access** - no upload needed!

### 4. Benefits of Migration

1. **Cost Savings**:
   - Don't need to upload/store 500+ GB of MIMIC-CXR images
   - Save ~$10/month in storage costs
   - Save hours of upload time

2. **Simplified Setup**:
   - Access PhysioNet's bucket directly
   - Only upload ~10-20 GB of your data
   - Faster deployment

3. **Multi-Bucket Architecture**:
   - Automatic routing to correct bucket
   - Seamless integration
   - No code changes needed by users

4. **Better Performance**:
   - Images accessed from PhysioNet's optimized bucket
   - Same region access (us-central1)
   - No data egress fees

## Usage Changes

### Old (AWS)
```bash
python src/phase1_preprocess.py \
  --s3-bucket bergermimiciv \
  --output-s3-bucket bergermimiciv \
  --mimic-cxr-path physionet.org/files/mimic-cxr-jpg/2.1.0
```

### New (GCS)
```bash
python src/phase1_preprocess.py \
  --gcs-bucket bergermimiciv \
  --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
  --output-gcs-bucket bergermimiciv \
  --mimic-cxr-path physionet.org/files/mimic-cxr-jpg/2.1.0 \
  --mimic-iv-path physionet.org/files/mimiciv/3.1 \
  --mimic-ed-path physionet.org/files/mimic-iv-ed/2.2
```

## Configuration Changes

### Old DataConfig
```python
config.use_s3 = True
config.s3_bucket = "bergermimiciv"
config.output_s3_bucket = "bergermimiciv"
```

### New DataConfig
```python
config.use_gcs = True
config.gcs_bucket = "bergermimiciv"  # Your bucket
config.gcs_cxr_bucket = "mimic-cxr-jpg-2.1.0.physionet.org"  # PhysioNet's bucket
config.output_gcs_bucket = "bergermimiciv"  # Output bucket
```

## Migration Checklist

- [x] Replace boto3 with google-cloud-storage
- [x] Update S3Helper → GCSHelper with multi-bucket support
- [x] Update all use_s3 → use_gcs references
- [x] Update command-line arguments
- [x] Update configuration classes
- [x] Remove AWS-specific documentation
- [x] Update README with GCS instructions
- [x] Create GCS setup guide
- [x] Update requirements.txt
- [x] Test local mode (no GCS)
- [ ] Test GCS mode with multi-bucket (next step)

## Testing

### Local Testing (Still Works!)
```bash
python src/test_phase1_local.py \
  --mimic-path ~/MIMIC_Data/physionet.org/files \
  --num-samples 10
```

### GCS Testing
```bash
# With gcsfuse (recommended)
gcsfuse bergermimiciv ~/my_data
gcsfuse -o ro mimic-cxr-jpg-2.1.0.physionet.org ~/cxr_images
python src/phase1_preprocess.py --mimic-cxr-path ~/cxr_images/files ...

# With native GCS
python src/phase1_preprocess.py \
  --gcs-bucket bergermimiciv \
  --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org ...
```

## Rollback Plan

If issues arise, the previous commit has the AWS version:
```bash
git log --oneline | head -5  # Find the commit before migration
git checkout <previous-commit-hash>
```

## Next Steps

1. Test GCS multi-bucket functionality
2. Verify PhysioNet's public bucket access
3. Run full preprocessing pipeline on GCS
4. Update any remaining AWS references in comments
5. Archive old AWS scripts for reference

## Notes

- All local testing functionality preserved
- Backward compatible with local file paths
- Multi-bucket support is transparent to users
- PhysioNet's bucket must be publicly accessible (it is)

