# Dependency Updates Summary

**Date**: 2025-10-18
**Purpose**: Update dependencies for phase1_stay_identification.py and leakage_filt_util.py

## Files Updated

### 1. [requirements.txt](requirements.txt)
**Changes**:
- ✓ Fixed typo: "dependenciesR" → "dependencies"
- ✓ Added `requests>=2.31.0` (for download_images_from_filelist.py)
- ✓ Added NLP libraries for leakage filtering:
  - `spacy>=3.5.0` - Core NLP framework
  - `scispacy>=0.5.3` - Medical/scientific text processing
  - `transformers>=4.30.0` - Hugging Face transformers (BioBERT)
  - `torch>=2.0.0` - PyTorch backend for transformers

**Why needed**:
- `requests`: Used by [scripts/download_images_from_filelist.py](scripts/download_images_from_filelist.py) to download images from PhysioNet
- `spacy` + `scispacy`: Used by [src/leakage_filt_util.py](src/leakage_filt_util.py) for medical entity recognition
- `transformers` + `torch`: Used for BioBERT semantic similarity to detect diagnosis leakage

### 2. [docker/Dockerfile](docker/Dockerfile)
**Changes**:
- ✓ Added system dependencies: `build-essential`, `git`
- ✓ Pre-download SciSpacy model: `en_core_sci_md`
- ✓ Pre-cache BioBERT model: `dmis-lab/biobert-v1.1`

**Why needed**:
- `build-essential`: Required to compile C extensions in spaCy and other NLP libraries
- `git`: Required by some pip packages
- Pre-downloading models: Avoids runtime downloads on AWS Batch instances

### 3. [docker/update_aws_image.sh](docker/update_aws_image.sh) ✨ NEW
**Purpose**: Automated script to rebuild and push Docker image to AWS ECR

**Usage**:
```bash
./docker/update_aws_image.sh
```

### 4. [docker/UPDATE_AWS_IMAGE.md](docker/UPDATE_AWS_IMAGE.md) ✨ NEW
**Purpose**: Complete guide for updating AWS Docker images

**Covers**:
- Step-by-step manual instructions
- Automated script usage
- Troubleshooting common issues
- Cost implications (~$0.40/month for larger image)
- Verification steps

## Dependency Analysis by File

### phase1_stay_identification.py
**Current dependencies**:
```python
import pandas as pd           # ✓ Already in requirements.txt
from datetime import timedelta # ✓ Built-in
from typing import Optional    # ✓ Built-in
from loguru import logger      # ✓ Already in requirements.txt
from tqdm import tqdm          # ✓ Already in requirements.txt
```
**Status**: ✓ All dependencies already satisfied

### leakage_filt_util.py
**Current dependencies**:
```python
import re                      # ✓ Built-in
import pandas as pd            # ✓ Already in requirements.txt
import numpy as np             # ✓ Already in requirements.txt
from typing import ...         # ✓ Built-in
from datetime import ...       # ✓ Built-in
import spacy                   # ✅ ADDED
from transformers import ...   # ✅ ADDED
import torch                   # ✅ ADDED
```
**Status**: ✓ Added missing NLP dependencies

### scripts/download_images_from_filelist.py
**Current dependencies**:
```python
import os, sys, time           # ✓ Built-in
import requests                # ✅ ADDED
import boto3                   # ✓ Already in requirements.txt
from pathlib import Path       # ✓ Built-in
from tqdm import tqdm          # ✓ Already in requirements.txt
import getpass                 # ✓ Built-in
from concurrent.futures import # ✓ Built-in
```
**Status**: ✓ Added requests library

## Image Size Impact

| Component | Before | After | Increase |
|-----------|--------|-------|----------|
| Base Image | ~200 MB | ~200 MB | 0 MB |
| Python Packages | ~300 MB | ~500 MB | +200 MB |
| NLP Models | 0 MB | ~3 GB | +3 GB |
| **Total** | **~500 MB** | **~3.7 GB** | **+3.2 GB** |

**Note**: The BioBERT model (dmis-lab/biobert-v1.1) is ~1.5 GB and SciSpacy model is ~50 MB.

## Cost Impact

### ECR Storage
- Before: ~$0.05/month (500 MB @ $0.10/GB/month)
- After: ~$0.37/month (3.7 GB @ $0.10/GB/month)
- **Increase**: ~$0.32/month

### Performance Impact
- **First pull**: ~5-10 minutes (downloading 3.7 GB)
- **Subsequent pulls**: ~30 seconds (cached layers)
- **Memory usage**: +2 GB RAM when NLP models loaded

## Installation Steps

### Local Development
```bash
# Update dependencies
pip install -r requirements.txt

# Install SciSpacy medical model
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz

# Test imports
python -c "import spacy, transformers, torch, requests; print('✓ All imports successful')"
```

### AWS Deployment
```bash
# Option 1: Use automated script
./docker/update_aws_image.sh

# Option 2: Manual (see docker/UPDATE_AWS_IMAGE.md)
cd /home/dev/Documents/Portfolio/MIMIC/MIMIC-IV-CXR-ED-Anomaly-Detection
docker build -t mimic-preprocessor:latest -f docker/Dockerfile .
docker tag mimic-preprocessor:latest 851454408197.dkr.ecr.us-west-2.amazonaws.com/mimic-preprocessor:latest
docker push 851454408197.dkr.ecr.us-west-2.amazonaws.com/mimic-preprocessor:latest
```

## Verification

### Test Local Installation
```bash
# Test phase1_stay_identification.py
python -m src.phase1_stay_identification --help

# Test leakage filtering
python src/leakage_filt_util.py
```

### Test Docker Image
```bash
# Build and test locally
docker build -t mimic-preprocessor:latest -f docker/Dockerfile .

# Test dependencies
docker run --rm mimic-preprocessor:latest \
    python -c "import spacy, transformers, torch, requests; print('✓ All dependencies loaded')"

# Test NLP models
docker run --rm mimic-preprocessor:latest \
    python -c "import spacy; nlp = spacy.load('en_core_sci_md'); print('✓ SciSpacy model loaded')"
```

### Test AWS ECR Image
```bash
# Pull from ECR
docker pull 851454408197.dkr.ecr.us-west-2.amazonaws.com/mimic-preprocessor:latest

# Run test
docker run --rm \
    851454408197.dkr.ecr.us-west-2.amazonaws.com/mimic-preprocessor:latest \
    python -c "from src.leakage_filt_util import DiagnosisLeakageFilter; \
               f = DiagnosisLeakageFilter(use_nlp_model=False); \
               print('✓ Leakage filter initialized')"
```

## Rollback Plan

If issues occur with new dependencies:

### Local Rollback
```bash
git checkout HEAD~1 requirements.txt
pip install -r requirements.txt
```

### AWS Rollback
```bash
# Use previous image version
aws batch register-job-definition \
    --job-definition-name mimic-preprocess-job \
    --type container \
    --container-properties '{
        "image": "851454408197.dkr.ecr.us-west-2.amazonaws.com/mimic-preprocessor:v1.0"
    }'
```

## Known Issues & Solutions

### Issue 1: PyTorch installation timeout
**Solution**: Pre-download PyTorch wheel
```bash
pip download torch==2.0.0 -d /tmp/wheels
docker build --build-arg WHEELS_DIR=/tmp/wheels -f docker/Dockerfile .
```

### Issue 2: SciSpacy model download fails
**Solution**: Download manually and copy to Docker image
```bash
wget https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz
# Update Dockerfile to COPY and install from local file
```

### Issue 3: Out of memory during Docker build
**Solution**: Increase Docker memory limit
```bash
# Docker Desktop: Settings → Resources → Memory → 8 GB
# Or use multi-stage build to reduce memory usage
```

## Next Steps

1. ✅ Dependencies updated in requirements.txt
2. ✅ Dockerfile updated with NLP support
3. ✅ Update scripts created
4. ⏳ Test local installation
5. ⏳ Build and push Docker image to ECR
6. ⏳ Update AWS Batch job definitions
7. ⏳ Run test job on AWS Batch

## References

- [requirements.txt](requirements.txt) - Python dependencies
- [docker/Dockerfile](docker/Dockerfile) - Docker image definition
- [docker/update_aws_image.sh](docker/update_aws_image.sh) - Automated update script
- [docker/UPDATE_AWS_IMAGE.md](docker/UPDATE_AWS_IMAGE.md) - Complete AWS update guide
- [src/leakage_filt_util.py](src/leakage_filt_util.py) - Diagnosis leakage filtering utilities
- [src/phase1_stay_identification.py](src/phase1_stay_identification.py) - Phase 1 processing
- [scripts/download_images_from_filelist.py](scripts/download_images_from_filelist.py) - Image download script
