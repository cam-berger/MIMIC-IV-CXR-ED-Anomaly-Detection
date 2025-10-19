# Updating AWS Docker Image for MIMIC Pipeline

This guide explains how to update your Docker image in AWS ECR after dependency changes.

## Summary of Changes

### Updated Dependencies in requirements.txt:
1. **Fixed typo**: "dependenciesR" → "dependencies"
2. **Added requests**: For download_images_from_filelist.py script
3. **Added NLP libraries**: For leakage_filt_util.py (diagnosis leakage filtering)
   - `spacy>=3.5.0` - Core NLP library
   - `scispacy>=0.5.3` - Medical/scientific text processing
   - `transformers>=4.30.0` - BioBERT model support
   - `torch>=2.0.0` - PyTorch for transformer models

### Updated Dockerfile:
1. **Added build tools**: git, build-essential (for compiling NLP libraries)
2. **Pre-download SciSpacy model**: en_core_sci_md (medical entity recognition)
3. **Pre-cache BioBERT**: dmis-lab/biobert-v1.1 (semantic similarity for leakage detection)

## Step-by-Step AWS Image Update

### 1. Build Docker Image Locally (Optional - Test First)

```bash
cd /home/dev/Documents/Portfolio/MIMIC/MIMIC-IV-CXR-ED-Anomaly-Detection

# Build and test locally
docker build -t mimic-preprocessor:latest -f docker/Dockerfile .

# Test the image
docker run --rm mimic-preprocessor:latest python -c "import spacy, transformers, torch, requests; print('All imports successful')"
```

### 2. Authenticate Docker with AWS ECR

```bash
# Get your ECR login credentials
aws ecr get-login-password --region us-west-2 --profile default | \
    docker login --username AWS --password-stdin 851454408197.dkr.ecr.us-west-2.amazonaws.com
```

### 3. Build Image for AWS

```bash
# Build with ECR tag
docker build -t mimic-preprocessor:latest -f docker/Dockerfile .

# Tag for ECR
docker tag mimic-preprocessor:latest \
    851454408197.dkr.ecr.us-west-2.amazonaws.com/mimic-preprocessor:latest

# Also create a versioned tag (recommended)
docker tag mimic-preprocessor:latest \
    851454408197.dkr.ecr.us-west-2.amazonaws.com/mimic-preprocessor:v1.1-nlp
```

### 4. Push to ECR

```bash
# Push latest tag
docker push 851454408197.dkr.ecr.us-west-2.amazonaws.com/mimic-preprocessor:latest

# Push versioned tag
docker push 851454408197.dkr.ecr.us-west-2.amazonaws.com/mimic-preprocessor:v1.1-nlp
```

### 5. Update AWS Batch Job Definition (if using AWS Batch)

```bash
# Register new job definition revision with updated image
aws batch register-job-definition \
    --job-definition-name mimic-preprocess-job \
    --type container \
    --container-properties '{
        "image": "851454408197.dkr.ecr.us-west-2.amazonaws.com/mimic-preprocessor:latest",
        "vcpus": 4,
        "memory": 8192,
        "jobRoleArn": "arn:aws:iam::851454408197:role/BatchJobRole",
        "executionRoleArn": "arn:aws:iam::851454408197:role/ecsTaskExecutionRole"
    }' \
    --region us-west-2 \
    --profile default
```

## Complete Update Script

Save this as `docker/update_aws_image.sh`:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "Updating MIMIC Preprocessor Docker Image"
echo "=========================================="

# Configuration
AWS_REGION="us-west-2"
AWS_PROFILE="default"
AWS_ACCOUNT_ID="851454408197"
ECR_REPO="mimic-preprocessor"
VERSION_TAG="v1.1-nlp"

# Get to project root
cd "$(dirname "$0")/.."

echo ""
echo "Step 1: Building Docker image..."
docker build -t ${ECR_REPO}:latest -f docker/Dockerfile .

echo ""
echo "Step 2: Authenticating with ECR..."
aws ecr get-login-password --region ${AWS_REGION} --profile ${AWS_PROFILE} | \
    docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo ""
echo "Step 3: Tagging images..."
docker tag ${ECR_REPO}:latest \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest

docker tag ${ECR_REPO}:latest \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${VERSION_TAG}

echo ""
echo "Step 4: Pushing to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${VERSION_TAG}

echo ""
echo "=========================================="
echo "✓ Image updated successfully!"
echo "=========================================="
echo ""
echo "Images pushed:"
echo "  • ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"
echo "  • ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${VERSION_TAG}"
echo ""
echo "To verify:"
echo "  aws ecr describe-images --repository-name ${ECR_REPO} --region ${AWS_REGION} --profile ${AWS_PROFILE}"
echo ""
```

## Quick Start

**Option 1: Use the automated script**
```bash
chmod +x docker/update_aws_image.sh
./docker/update_aws_image.sh
```

**Option 2: Manual steps**
Follow steps 2-4 above.

## Verify Update

```bash
# Check ECR for new image
aws ecr describe-images \
    --repository-name mimic-preprocessor \
    --region us-west-2 \
    --profile default \
    --query 'sort_by(imageDetails,& imagePushedAt)[-5:]' \
    --output table

# Pull and test the image
docker pull 851454408197.dkr.ecr.us-west-2.amazonaws.com/mimic-preprocessor:latest

docker run --rm \
    851454408197.dkr.ecr.us-west-2.amazonaws.com/mimic-preprocessor:latest \
    python -c "import spacy, transformers, torch, requests; print('✓ All dependencies loaded')"
```

## Troubleshooting

### Build fails during NLP model download
**Issue**: Network timeout when downloading BioBERT or SciSpacy models

**Solution**: Increase Docker build timeout and retry
```bash
docker build --network=host -t mimic-preprocessor:latest -f docker/Dockerfile .
```

### "No space left on device"
**Issue**: Large model files fill up Docker storage

**Solution**: Prune old Docker images
```bash
docker system prune -a
```

### ECR authentication fails
**Issue**: AWS credentials expired or incorrect

**Solution**: Refresh credentials
```bash
aws sso login --profile default
# OR
aws configure --profile default
```

## Image Size Considerations

The updated image will be significantly larger due to NLP models:
- **Previous size**: ~500 MB
- **Updated size**: ~3-4 GB (BioBERT + SciSpacy models)

**Implications**:
1. Longer first-time pull on AWS Batch/ECS instances (~5-10 minutes)
2. Higher ECR storage costs (~$0.40/month for 4GB)
3. Subsequent pulls use cached layers (faster)

**Optimization** (if needed):
- Use multi-stage builds to reduce final image size
- Consider pulling models at runtime instead of baking into image
- Use AWS EFS to share model cache across instances

## Cost Estimate

- **ECR Storage**: ~$0.10/GB/month = $0.40/month
- **Data Transfer**: Negligible (within same region)
- **Total**: ~$0.40/month additional cost

## Next Steps After Update

1. ✓ Update local requirements: `pip install -r requirements.txt`
2. ✓ Test phase1_stay_identification.py locally
3. ✓ Test leakage_filt_util.py filtering
4. Push image to ECR (follow this guide)
5. Update AWS Batch job definitions
6. Run test job on AWS Batch
7. Monitor logs for any missing dependencies

## Support Files

- [requirements.txt](../requirements.txt) - Updated Python dependencies
- [Dockerfile](Dockerfile) - Updated Docker image definition
- [aws_config.yaml](../config/aws_config.yaml) - AWS configuration
