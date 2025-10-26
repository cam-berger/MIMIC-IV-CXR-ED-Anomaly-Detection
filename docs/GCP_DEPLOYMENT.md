# Google Cloud Platform Deployment Guide

Complete guide for deploying the MIMIC preprocessing pipeline on GCP to process the full dataset.

## Overview

This deployment:
- Creates a GCP Compute Engine VM (8 vCPUs, 52GB RAM)
- Automatically installs dependencies
- Runs the full preprocessing pipeline (Phase 1 + Leakage Filtering)
- Processes all available MIMIC-IV + MIMIC-CXR data
- Auto-shuts down when complete to save costs
- Saves results to Google Cloud Storage

## Prerequisites

### 1. Data Preparation

Upload your MIMIC data to GCS:

```bash
# Upload MIMIC-IV
gsutil -m cp -r ~/MIMIC_Data/physionet.org/files/mimiciv \
  gs://bergermimiciv/physionet.org/files/

# Upload MIMIC-IV-ED
gsutil -m cp -r ~/MIMIC_Data/physionet.org/files/mimic-iv-ed \
  gs://bergermimiciv/physionet.org/files/

# MIMIC-CXR-JPG: Already on PhysioNet's public bucket (no upload needed!)
```

### 2. Code Preparation

**Option A: Use Git Repository (Recommended)**

Update `scripts/deploy_gcp.sh`:
```bash
GIT_REPO="https://github.com/YOUR_USERNAME/MIMIC-IV-CXR-ED-Anomaly-Detection.git"
```

**Option B: Upload Code to GCS**

```bash
# Package code
tar -czf src.tar.gz src/ requirements.txt

# Upload to GCS
gsutil cp src.tar.gz gs://bergermimiciv/code/
```

### 3. GCP Setup

```bash
# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable compute.googleapis.com
gcloud services enable storage.googleapis.com
```

## Deployment Options

### Option 1: Automated Deployment (Recommended)

Use the deployment script to automatically create and configure the VM:

```bash
bash scripts/deploy_gcp.sh YOUR_PROJECT_ID bergermimiciv
```

The script will:
1. Create a VM named `mimic-preprocessing-YYYYMMDD-HHMMSS`
2. Upload startup script
3. Start the VM
4. Show monitoring commands

### Option 2: Manual Deployment

If you prefer manual control:

```bash
# 1. Create VM
gcloud compute instances create mimic-preprocessing \
  --project=YOUR_PROJECT_ID \
  --zone=us-central1-a \
  --machine-type=n1-highmem-8 \
  --boot-disk-size=200GB \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --scopes=cloud-platform

# 2. SSH into VM
gcloud compute ssh mimic-preprocessing --zone=us-central1-a

# 3. On the VM, run setup manually
sudo apt-get update
sudo apt-get install -y python3-pip git

# Clone repository
git clone YOUR_REPO_URL
cd MIMIC-IV-CXR-ED-Anomaly-Detection

# Install dependencies
pip3 install -r requirements.txt

# Run pipeline
python3 src/run_full_pipeline.py \
  --gcs-bucket bergermimiciv \
  --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
  --gcs-project-id YOUR_PROJECT_ID \
  --mimic-iv-path physionet.org/files/mimiciv/3.1 \
  --mimic-ed-path physionet.org/files/mimic-iv-ed/2.2 \
  --output-path processed/phase1_final \
  --aggressive-filtering
```

## Monitoring the Pipeline

### Check VM Status

```bash
# List running VMs
gcloud compute instances list

# View serial port output (startup logs)
gcloud compute instances get-serial-port-output mimic-preprocessing-YYYYMMDD-HHMMSS --zone=us-central1-a

# SSH into VM
gcloud compute ssh mimic-preprocessing-YYYYMMDD-HHMMSS --zone=us-central1-a
```

### Monitor Pipeline Progress

```bash
# SSH into VM and monitor logs
gcloud compute ssh mimic-preprocessing-YYYYMMDD-HHMMSS --zone=us-central1-a

# On the VM:
tail -f /var/log/mimic-preprocessing.log

# Or view startup script output
sudo journalctl -u google-startup-scripts.service -f
```

### Check Output in GCS

```bash
# List output files
gsutil ls -lh gs://bergermimiciv/processed/phase1_final/

# View metadata
gsutil cat gs://bergermimiciv/processed/phase1_final/filtering_metadata.json

# Download logs
gsutil ls gs://bergermimiciv/logs/
gsutil cp gs://bergermimiciv/logs/preprocessing-*.log ./
```

## Auto-Shutdown Behavior

The VM is configured to **automatically shut down** when the pipeline completes to save costs.

### Prevent Auto-Shutdown

If you need to keep the VM running (e.g., for debugging):

```bash
# SSH into VM
gcloud compute ssh mimic-preprocessing-YYYYMMDD-HHMMSS --zone=us-central1-a

# Create flag file to prevent shutdown
sudo touch /tmp/no-shutdown
```

### Manual Shutdown

```bash
# Stop VM (can be restarted)
gcloud compute instances stop mimic-preprocessing-YYYYMMDD-HHMMSS --zone=us-central1-a

# Delete VM (permanent)
gcloud compute instances delete mimic-preprocessing-YYYYMMDD-HHMMSS --zone=us-central1-a
```

## Cost Estimation

### VM Costs (us-central1)

| Machine Type | vCPUs | RAM | Cost/hour | Est. Total* |
|--------------|-------|-----|-----------|-------------|
| n1-standard-8 | 8 | 30GB | $0.38 | ~$15-30 |
| n1-highmem-8 | 8 | 52GB | $0.47 | ~$19-38 |
| n1-standard-16 | 16 | 60GB | $0.76 | ~$30-61 |

*Estimated for 40-80 hour runtime on full MIMIC dataset

### Storage Costs

- GCS Standard Storage: $0.020/GB/month
- Estimated output size: ~50-100GB
- Monthly cost: $1-2

### Requester Pays (PhysioNet Bucket)

- Egress from `mimic-cxr-jpg-2.1.0.physionet.org`: ~$0.12/GB
- Estimated transfer: ~10-50GB (metadata + sampled images)
- Total: $1-6

**Total Estimated Cost: $20-50 for full pipeline run**

## Troubleshooting

### Pipeline Fails

```bash
# Check logs on VM
gcloud compute ssh mimic-preprocessing-YYYYMMDD-HHMMSS --zone=us-central1-a
tail -200 /var/log/mimic-preprocessing.log

# Check error logs in GCS
gsutil ls gs://bergermimiciv/logs/
gsutil cat gs://bergermimiciv/logs/preprocessing-ERROR-*.log
```

### Out of Memory

If you encounter OOM errors:

1. **Increase VM RAM:**
   ```bash
   # Edit deploy_gcp.sh
   MACHINE_TYPE="n1-highmem-16"  # 16 vCPUs, 104GB RAM
   ```

2. **Reduce resource requirements:**
   ```bash
   # In vm_startup.sh, modify pipeline arguments:
   --image-size 224 \       # Instead of 518
   --max-text-length 512    # Instead of 8192
   ```

### Authentication Errors

```bash
# On the VM, check service account
gcloud auth list

# Ensure VM has correct scopes
gcloud compute instances describe mimic-preprocessing-YYYYMMDD-HHMMSS --zone=us-central1-a --format="value(serviceAccounts[0].scopes)"
```

### Slow Performance

1. **Use faster disk:**
   ```bash
   # In deploy_gcp.sh
   --boot-disk-type=pd-ssd  # Instead of pd-standard
   ```

2. **Add GPU (optional):**
   ```bash
   # Uncomment in deploy_gcp.sh
   GPU_TYPE="nvidia-tesla-t4"
   GPU_COUNT="1"
   ```

## Advanced Configuration

### Using GPU Acceleration

Edit `scripts/deploy_gcp.sh`:

```bash
# Uncomment these lines:
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT="1"
```

Then the VM will have GPU support for faster image processing.

### Custom Pipeline Parameters

Edit `scripts/vm_startup.sh` to modify pipeline arguments:

```bash
python src/run_full_pipeline.py \
    --gcs-bucket "$BUCKET_NAME" \
    --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
    --gcs-project-id "$PROJECT_ID" \
    --mimic-iv-path physionet.org/files/mimiciv/3.1 \
    --mimic-ed-path physionet.org/files/mimic-iv-ed/2.2 \
    --output-path processed/phase1_final \
    --aggressive-filtering \
    --image-size 518 \              # Modify this
    --max-text-length 8192 \        # Modify this
    --use-nlp-model                 # Add for BioBERT filtering
```

### Running Only One Stage

Skip preprocessing (run filtering only):
```bash
python src/run_full_pipeline.py \
    --skip-preprocessing \
    --gcs-bucket bergermimiciv \
    ...
```

Skip filtering (run preprocessing only):
```bash
python src/run_full_pipeline.py \
    --skip-filtering \
    --gcs-bucket bergermimiciv \
    ...
```

## Next Steps

After preprocessing completes:

1. **Verify Output:**
   ```bash
   gsutil ls -lh gs://bergermimiciv/processed/phase1_final/
   gsutil cat gs://bergermimiciv/processed/phase1_final/filtering_metadata.json
   ```

2. **Download for Local Development (Optional):**
   ```bash
   gsutil -m cp -r gs://bergermimiciv/processed/phase1_final/ ./data/
   ```

3. **Proceed to Phase 2:** Model training (see `docs/PHASE2_TRAINING.md`)

## Support

For issues or questions:
- Check logs: `/var/log/mimic-preprocessing.log`
- Review GCS logs: `gs://bergermimiciv/logs/`
- Open GitHub issue with log excerpts
