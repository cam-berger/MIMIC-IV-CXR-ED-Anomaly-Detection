# SageMaker Quick Start Guide
## Your Setup: S3 Bucket "bergermimiciv"

Since your data is already in S3, you can start immediately with SageMaker!

---

## Option 1: SageMaker Notebook Instance (Recommended - Interactive)

### Step 1: Create IAM Role (One-time setup)

```bash
# Check if role already exists
aws iam get-role --role-name SageMaker-MIMIC-Role 2>/dev/null || \
aws iam create-role \
    --role-name SageMaker-MIMIC-Role \
    --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "sagemaker.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }'

# Attach policies
aws iam attach-role-policy \
    --role-name SageMaker-MIMIC-Role \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
    --role-name SageMaker-MIMIC-Role \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Step 2: Create SageMaker Notebook Instance

**Via AWS Console (Easiest):**

1. Go to [SageMaker Console](https://console.aws.amazon.com/sagemaker/)
2. Click **Notebook instances** → **Create notebook instance**
3. Settings:
   - **Name**: `mimic-preprocessing`
   - **Instance type**: `ml.p3.2xlarge` (GPU) or `ml.m5.2xlarge` (CPU, cheaper)
   - **Volume size**: 100 GB
   - **IAM role**: Select "SageMaker-MIMIC-Role" (or create new with S3 access)
   - **GitHub repository** (optional): Add your GitHub repo
4. Click **Create notebook instance**
5. Wait 5-10 minutes for "InService" status

**Via AWS CLI:**

```bash
aws sagemaker create-notebook-instance \
    --notebook-instance-name mimic-preprocessing \
    --instance-type ml.p3.2xlarge \
    --role-arn arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):role/SageMaker-MIMIC-Role \
    --volume-size-in-gb 100 \
    --default-code-repository https://github.com/cam-berger/MIMIC-IV-CXR-ED-Anomaly-Detection.git
```

### Step 3: Open Jupyter and Setup

1. Click **Open Jupyter** (or **Open JupyterLab**)
2. Open a terminal in Jupyter
3. Install dependencies:

```bash
# Activate conda environment
conda activate pytorch_p310

# Install required packages
pip install opencv-python-headless faiss-cpu sentence-transformers transformers
```

### Step 4: Upload and Configure Notebook

1. Upload `notebooks/phase1_preprocess.ipynb` (if not using Git)
2. Open the notebook
3. **Update Configuration Cell (#2)** with your S3 bucket:

```python
config = DataConfig(
    # S3 paths - UPDATE THESE TO MATCH YOUR BUCKET STRUCTURE
    mimic_cxr_path="files/mimic-cxr-jpg/2.1.0",  # Adjust to your path
    mimic_iv_path="files/mimiciv/3.1",           # Adjust to your path
    reflacx_path="files/reflacx",                # Adjust to your path
    output_path="processed/phase1_preprocess",

    # Enable S3 mode
    use_s3=True,
    s3_bucket="bergermimiciv",           # ✓ Your bucket name
    output_s3_bucket="bergermimiciv",    # ✓ Output to same bucket

    # Processing settings
    image_size=518,
    max_text_length=8192,
    top_k_retrieval=5,

    # Data splits
    train_split=0.7,
    val_split=0.15,
    test_split=0.15
)
```

### Step 5: Verify Your S3 Structure

In the notebook, add a cell to check your S3 structure:

```python
import boto3

s3 = boto3.client('s3')

# List top-level folders in your bucket
response = s3.list_objects_v2(Bucket='bergermimiciv', Delimiter='/')
print("Top-level folders in bergermimiciv:")
for prefix in response.get('CommonPrefixes', []):
    print(f"  📁 {prefix['Prefix']}")

# Update the paths in config based on what you see!
```

### Step 6: Run the Notebook

Run all cells! The notebook will:
1. ✓ Load data from S3 bucket "bergermimiciv"
2. ✓ Process images and clinical text
3. ✓ Create train/val/test splits
4. ✓ Save results back to S3 bucket "bergermimiciv"

### Step 7: Monitor Progress

Watch the progress bars in the notebook. For the full MIMIC-CXR dataset:
- **ml.p3.2xlarge**: ~8-12 hours
- **ml.p3.8xlarge**: ~2-4 hours

### Step 8: Stop Instance When Done

**IMPORTANT:** Stop the instance to avoid charges!

```bash
aws sagemaker stop-notebook-instance --notebook-instance-name mimic-preprocessing
```

Or in AWS Console: Select instance → **Actions** → **Stop**

---

## Option 2: SageMaker Processing Job (Batch, Automated)

For automated batch processing without manual intervention:

### Quick Launch Script

```python
# Create file: launch_my_job.py
import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

# Initialize
session = sagemaker.Session()
role = sagemaker.get_execution_role()  # Or specify your role ARN

# Create processor
processor = ScriptProcessor(
    role=role,
    image_uri='763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.0.0-gpu-py310',
    command=['python3'],
    instance_count=1,
    instance_type='ml.p3.2xlarge',
    volume_size_in_gb=100,
    max_runtime_in_seconds=43200,  # 12 hours
    base_job_name='mimic-preprocess'
)

# Run processing job
processor.run(
    code='sagemaker/sagemaker_processing_job.py',
    source_dir='.',
    inputs=[
        ProcessingInput(
            source='s3://bergermimiciv/',
            destination='/opt/ml/processing/input/data'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output',
            destination='s3://bergermimiciv/processed/phase1_preprocess'
        )
    ],
    arguments=[
        '--s3-bucket', 'bergermimiciv',
        '--output-s3-bucket', 'bergermimiciv',
        '--mimic-cxr-path', 'YOUR_PATH_HERE',  # Update based on your structure
        '--mimic-iv-path', 'YOUR_PATH_HERE',   # Update based on your structure
    ],
    wait=False
)

print(f"Job launched: {processor.latest_job.job_name}")
```

Then run:
```bash
python launch_my_job.py
```

---

## Check Your S3 Bucket Structure

First, let's see what's in your bucket:

```bash
# List contents
aws s3 ls s3://bergermimiciv/ --recursive | head -20

# Or just top-level folders
aws s3 ls s3://bergermimiciv/
```

**Then update the paths in the notebook config accordingly!**

Common structures:
- `s3://bergermimiciv/mimic-cxr-jpg/2.1.0/...`
- `s3://bergermimiciv/physionet.org/files/mimic-cxr-jpg/2.1.0/...`
- `s3://bergermimiciv/files/mimic-cxr-jpg/2.1.0/...`

---

## Estimated Costs

### Notebook Instance (Interactive)
- **ml.m5.2xlarge**: $0.54/hour (8 cores, 32GB RAM)
- **ml.p3.2xlarge**: $3.82/hour (8 cores, 61GB RAM, 1 GPU)

**Full preprocessing run**: ~$30-45 for ml.p3.2xlarge (10-12 hours)

### Processing Job
Same pricing + you can use **Spot Instances** for 70% savings!

### Storage (S3)
- Minimal cost (~$0.023/GB/month)
- Your data + processed output

---

## Troubleshooting

### Issue: Can't find files in S3
**Solution:** Run this in notebook to explore:
```python
import boto3
s3 = boto3.client('s3')
response = s3.list_objects_v2(Bucket='bergermimiciv', MaxKeys=10)
for obj in response.get('Contents', []):
    print(obj['Key'])
```

### Issue: Out of memory
**Solution:** Use larger instance type:
- Upgrade to `ml.m5.4xlarge` or `ml.p3.8xlarge`

### Issue: Permission denied
**Solution:** Check IAM role has S3 access to "bergermimiciv"

---

## Next Steps

After preprocessing completes:

1. **Verify output:**
```bash
aws s3 ls s3://bergermimiciv/processed/phase1_preprocess/
```

2. **Download metadata:**
```bash
aws s3 cp s3://bergermimiciv/processed/phase1_preprocess/metadata.json ./
cat metadata.json
```

3. **Proceed to model training** with the processed data!

---

## Need Help?

1. Check CloudWatch Logs: SageMaker Console → Notebook Instance → View Logs
2. Check S3 bucket permissions
3. Verify IAM role has both SageMaker and S3 access

**Remember to STOP your notebook instance when not in use to avoid charges!**
