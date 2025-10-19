# MIMIC-CXR Image Download Guide

## For Deep Learning / Computer Vision Projects

If you're doing deep learning on X-ray images (like anomaly detection), you need the **MIMIC-CXR-JPG** dataset.

---

## Quick Start

### Step 1: Download Metadata First

```bash
python scripts/stream_mimic_to_s3.py
```

This downloads the metadata files you need for Phase 1 and 2 (~17 MB).

### Step 2: Download Images

```bash
python scripts/download_cxr_jpg_images.py
```

This will:
1. Prompt for your PhysioNet credentials
2. Download the MIMIC-CXR-JPG metadata to get the image list
3. Ask how many images you want to download:
   - **Option 1:** ALL 377,110 images (~500 GB)
   - **Option 2:** Sample of 1,000 images for testing (~1.3 GB)
   - **Option 3:** Custom amount
4. Download images in parallel (4 workers)
5. Upload directly to S3 without storing locally

---

## Dataset Structure

### MIMIC-CXR-JPG Contains:

**Metadata Files:**
- `mimic-cxr-2.0.0-metadata.csv.gz` - Image details (view, dimensions, etc.)
- `mimic-cxr-2.0.0-split.csv.gz` - Train/validation/test split
- `mimic-cxr-2.0.0-chexpert.csv.gz` - Labels (14 diagnoses)
- `mimic-cxr-2.0.0-negbio.csv.gz` - Alternative labels

**Image Files:**
- 377,110 JPG images (~500 GB total)
- Organized: `files/p10/p10000032/s50414267/[dicom_id].jpg`
- Average size: ~1.3 MB per image

---

## Download Options Explained

### Option 1: Full Dataset (Recommended for Production)

```bash
python scripts/download_cxr_jpg_images.py
# Choose option 1
```

**Downloads:** All 377,110 images
**Size:** ~500 GB
**Time:** 10-20 hours (depends on connection)
**S3 Cost:** ~$11.50/month

**Use when:**
- Training production models
- Need complete dataset
- Have good internet connection

### Option 2: Sample for Testing (Recommended for Development)

```bash
python scripts/download_cxr_jpg_images.py
# Choose option 2
```

**Downloads:** 1,000 random images
**Size:** ~1.3 GB
**Time:** ~30 minutes
**S3 Cost:** ~$0.03/month

**Use when:**
- Testing your pipeline
- Developing/debugging code
- Prototyping models
- Limited bandwidth

### Option 3: Custom Amount

```bash
python scripts/download_cxr_jpg_images.py
# Choose option 3
# Enter your desired number
```

**Use when:**
- You know exactly how many images you need
- Training on a subset (e.g., only frontal views)
- Budget constraints

---

## Recommended Workflow

### For Development:

1. **Download sample (1,000 images):**
   ```bash
   python scripts/download_cxr_jpg_images.py
   # Choose option 2
   ```

2. **Develop your pipeline** using the sample

3. **Test end-to-end** with sample data

4. **When ready, download full dataset:**
   ```bash
   python scripts/download_cxr_jpg_images.py
   # Choose option 1
   ```

### For Production:

1. **Download metadata:**
   ```bash
   python scripts/stream_mimic_to_s3.py
   ```

2. **Download ALL images:**
   ```bash
   python scripts/download_cxr_jpg_images.py
   # Choose option 1
   ```

3. **Run your pipeline:**
   ```bash
   python -m src.phase1_stay_identification
   python -m src.phase2_clinical_extraction
   python -m src.phase3_integration
   ```

---

## What Gets Downloaded to S3

### After Running `stream_mimic_to_s3.py`:
```
s3://bergermimiciv/
└── mimic-cxr/2.0.0/
    ├── cxr-record-list.csv.gz       # Metadata with dates/times
    └── cxr-study-list.csv.gz        # Study-level metadata
```

### After Running `download_cxr_jpg_images.py`:
```
s3://bergermimiciv/
└── mimic-cxr-jpg/2.1.0/
    ├── mimic-cxr-2.0.0-metadata.csv.gz   # Image metadata
    ├── mimic-cxr-2.0.0-split.csv.gz      # Train/val/test
    ├── mimic-cxr-2.0.0-chexpert.csv.gz   # Labels
    ├── mimic-cxr-2.0.0-negbio.csv.gz     # Alternative labels
    └── files/
        ├── p10/
        │   ├── p10000032/
        │   │   └── s50414267/
        │   │       └── 02aa804e...4014.jpg
        │   └── ...
        ├── p11/
        └── ... (p12-p19)
```

---

## Troubleshooting

### "Authentication failed"
- Check your PhysioNet credentials
- Make sure you signed the DUA for MIMIC-CXR-JPG (not just MIMIC-CXR)
- Visit: https://physionet.org/content/mimic-cxr-jpg/2.1.0/

### "Access denied (403)"
- Sign the Data Use Agreement for MIMIC-CXR-JPG specifically
- Each dataset requires its own DUA signature

### Download is slow
- **Use sample for development** (option 2)
- Full download takes 10-20 hours - this is normal
- The script uses 4 parallel workers for speed
- Can resume if interrupted (images already in S3 are skipped)

### Out of disk space
- The script streams directly to S3 without local storage
- Only downloads metadata file to memory first
- If you see disk usage, it's temporary

### Failed images
- Some images may fail due to network issues
- Script retries 3 times per image
- Failed images are saved to `failed_images.txt`
- You can re-run to retry failed images

---

## Cost Estimates

### S3 Storage (us-west-2):
- **Sample (1,000 images):** ~1.3 GB = ~$0.03/month
- **Full dataset:** ~500 GB = ~$11.50/month

### Data Transfer:
- **Into S3:** Free
- **Out of S3:** $0.09/GB (only when accessing)

### Recommended:
- Use S3 Intelligent-Tiering for automatic cost optimization
- Access images from EC2 in same region (free transfer)
- Delete sample after testing to save costs

---

## Advanced: Resume Failed Downloads

If download is interrupted:

```bash
# Re-run the same command
python scripts/download_cxr_jpg_images.py
```

The script will:
- Check S3 for existing images
- Skip already downloaded images
- Only download missing ones

---

## Next Steps

After downloading images:

1. **Verify upload:**
   ```bash
   aws s3 ls s3://bergermimiciv/mimic-cxr-jpg/2.1.0/files/ --recursive | head -20
   ```

2. **Check image count:**
   ```bash
   aws s3 ls s3://bergermimiciv/mimic-cxr-jpg/2.1.0/files/ --recursive | wc -l
   ```

3. **Start your deep learning pipeline!**

---

## References

- MIMIC-CXR-JPG: https://physionet.org/content/mimic-cxr-jpg/2.1.0/
- Paper: Johnson et al., "MIMIC-CXR-JPG Database v2.0.0" (2019)
- Labels: CheXpert (14 observations) and NegBio annotations included
