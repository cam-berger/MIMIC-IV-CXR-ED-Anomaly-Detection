# Quick Start: MIMIC-CXR-JPG Download

## Download Chest X-Ray Images + Clinical Data for Deep Learning

This project uses **MIMIC-CXR-JPG 2.1.0** which includes:
- ✅ 377,110 chest X-ray images (JPG format)
- ✅ CheXpert labels (14 diagnoses)
- ✅ Metadata with dates/times for linking to ED stays
- ✅ Train/validation/test splits

---

## One Script Downloads Everything

```bash
python scripts/download_mimic_cxr_to_s3.py
```

**What it does:**
1. Downloads MIMIC-CXR-JPG metadata (~15 MB)
2. Asks if you want to download images
3. Streams images **directly from PhysioNet to S3** (no local storage needed)
4. Shows progress bars and stats

**You'll be asked:**
```
Download Options:
  1. Download ALL images (377,110 images, ~500 GB)
  2. Download a SAMPLE for testing (1,000 images, ~1.3 GB)
  3. Download CUSTOM amount
  4. Cancel
```

### ⭐ Recommended: Start with Sample

**For Development/Testing:**
- Choose **Option 2** (1,000 images)
- Time: ~30 minutes
- Cost: ~$0.03/month in S3
- Perfect for testing your pipeline

**For Production/Training:**
- Choose **Option 1** (all 377,110 images)
- Time: ~10-20 hours
- Cost: ~$11.50/month in S3
- Full dataset for final model training

---

## What Gets Downloaded to S3

```
s3://bergermimiciv/mimic-cxr-jpg/2.1.0/
├── mimic-cxr-2.0.0-metadata.csv.gz    # Image metadata (dates, times, views)
├── mimic-cxr-2.0.0-split.csv.gz       # Train/val/test split
├── mimic-cxr-2.0.0-chexpert.csv.gz    # Labels (14 diagnoses)
├── mimic-cxr-2.0.0-negbio.csv.gz      # Alternative labels
└── files/
    ├── p10/p10000032/s50414267/[dicom_id].jpg
    ├── p11/...
    └── ... (p12-p19)
```

---

## ✅ Key Features

### Streams Directly to S3
- **No local disk space needed**
- Downloads to memory → immediately uploads to S3
- Works on any machine, even with limited storage

### Included Labels for Deep Learning
- **CheXpert:** 14 diagnostic categories (Cardiomegaly, Edema, Consolidation, etc.)
- **Train/Val/Test Split:** Pre-defined splits ready to use
- **NegBio:** Alternative labeling approach

### Resume-able
If interrupted, just re-run:
```bash
python scripts/download_mimic_cxr_to_s3.py
```
Already-downloaded images are skipped automatically.

---

## Verify Upload

### Check metadata:
```bash
aws s3 ls s3://bergermimiciv/mimic-cxr-jpg/2.1.0/
```

### Check images:
```bash
aws s3 ls s3://bergermimiciv/mimic-cxr-jpg/2.1.0/files/ --recursive | head -20
```

### Count images:
```bash
aws s3 ls s3://bergermimiciv/mimic-cxr-jpg/2.1.0/files/ --recursive | wc -l
```

---

## Next Steps: Run Your Pipeline

### Phase 1: Link CXRs to ED Stays
```bash
python -m src.phase1_stay_identification
```
Links chest X-rays to emergency department visits using timestamps.

### Phase 2: Extract Clinical Data
```bash
python -m src.phase2_clinical_extraction
```
Extracts labs, vitals, medications, diagnoses for each ED stay.

### Phase 3: Integrate Everything
```bash
python -m src.phase3_integration
```
Combines images + clinical data into final multimodal dataset.

---

## Your Multimodal Dataset

After running the pipeline, you'll have:
- ✅ **377,110 chest X-ray images** (or your sample size)
- ✅ **Clinical data** (labs, vitals, medications) for each image
- ✅ **ED stay information** (arrival time, diagnoses, outcomes)
- ✅ **CheXpert labels** for anomaly detection
- ✅ **Train/val/test splits** ready for modeling

**Perfect for multimodal deep learning anomaly detection!** 🎯

---

## Requirements

### PhysioNet Access
1. Complete CITI training: https://physionet.org/settings/training/
2. Request credentialing: https://physionet.org/settings/credentialing/
3. Sign DUA for MIMIC-CXR-JPG: https://physionet.org/content/mimic-cxr-jpg/2.1.0/

### AWS Setup
- S3 bucket: `bergermimiciv` (already configured)
- AWS CLI configured with credentials

---

## Troubleshooting

### "Authentication failed"
Sign the Data Use Agreement for **MIMIC-CXR-JPG** (not just MIMIC-CXR):
https://physionet.org/content/mimic-cxr-jpg/2.1.0/

### Download is slow
- Normal for 500 GB dataset
- Use sample (Option 2) for testing
- Full download takes 10-20 hours

### Failed images
- Script retries 3 times automatically
- Failed images saved to `failed_images.txt`
- Re-run to retry failed images

---

## Cost Summary

| Item | Size | Monthly Cost (S3 Standard) |
|------|------|----------------------------|
| Metadata | ~15 MB | ~$0.0003 |
| Sample (1,000 images) | ~1.3 GB | ~$0.03 |
| Full Dataset | ~500 GB | ~$11.50 |

**Tip:** Use S3 Intelligent-Tiering to automatically reduce costs for infrequently accessed data.

---

## More Information

- **Detailed Guide:** [docs/IMAGE_DOWNLOAD_GUIDE.md](docs/IMAGE_DOWNLOAD_GUIDE.md)
- **MIMIC-CXR-JPG Dataset:** https://physionet.org/content/mimic-cxr-jpg/2.1.0/
- **Paper:** Johnson et al., "MIMIC-CXR-JPG Database v2.0.0" (2019)
