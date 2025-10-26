# Quick Deployment Guide: Run Full Pipeline on GCP Compute Engine

Complete step-by-step guide to deploy and run the full preprocessing pipeline on Google Cloud.

---

## Prerequisites Checklist

Before you start, ensure you have:

- ✅ Google Cloud Project created (`mimic-cxr-pred`)
- ✅ Billing enabled on the project
- ✅ MIMIC data uploaded to GCS bucket (`bergermimiciv`)
- ✅ `gcloud` CLI installed and authenticated
- ✅ Code pushed to GitHub repository

---

## Step 1: Enable Required APIs (One-time setup)

```bash
# Set your project
gcloud config set project mimic-cxr-pred

# Enable Compute Engine API
gcloud services enable compute.googleapis.com --project=mimic-cxr-pred

# Enable Cloud Storage API
gcloud services enable storage.googleapis.com --project=mimic-cxr-pred

# Verify APIs are enabled
gcloud services list --enabled --project=mimic-cxr-pred | grep -E 'compute|storage'
```

**Expected output:**
```
compute.googleapis.com          Compute Engine API
storage.googleapis.com          Cloud Storage JSON API
```

---

## Step 2: Delete Any Existing VMs (If needed)

```bash
# Check for existing VMs
gcloud compute instances list --project=mimic-cxr-pred

# If you see old VMs, delete them to free up quota
gcloud compute instances delete mimic-preprocessing-YYYYMMDD-HHMMSS \
  --zone=us-central1-a \
  --project=mimic-cxr-pred \
  --quiet
```

---

## Step 3: Deploy with Automated Script (Recommended)

### Option A: Fully Automated Deployment

```bash
# Navigate to project directory
cd ~/Documents/Portfolio/MIMIC/MIMIC-IV-CXR-ED-Anomaly-Detection

# Run deployment script (creates VM, installs everything, auto-runs pipeline)
bash scripts/deploy_gcp.sh mimic-cxr-pred bergermimiciv
```

**What happens:**
1. ✅ Creates VM with 4 vCPUs, 15GB RAM
2. ✅ Uploads startup script
3. ✅ VM automatically installs dependencies
4. ✅ VM clones code from GitHub
5. ✅ VM runs full pipeline automatically
6. ✅ VM auto-shuts down when complete

**Monitor the deployment:**

The script will output commands like:

```
Monitor startup progress:
  gcloud compute ssh mimic-preprocessing-20251025-HHMMSS --zone=us-central1-a --command='tail -f /var/log/syslog | grep startup-script'

SSH into VM:
  gcloud compute ssh mimic-preprocessing-20251025-HHMMSS --zone=us-central1-a
```

---

### Option B: Manual Deployment (More Control)

If you want full control over each step:

#### Step 3.1: Create the VM

```bash
gcloud compute instances create mimic-preprocessing \
  --project=mimic-cxr-pred \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-standard \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --scopes=cloud-platform
```

**Expected output:**
```
Created [https://www.googleapis.com/compute/v1/projects/mimic-cxr-pred/zones/us-central1-a/instances/mimic-preprocessing].
NAME                  ZONE           MACHINE_TYPE    PREEMPTIBLE  INTERNAL_IP  EXTERNAL_IP    STATUS
mimic-preprocessing  us-central1-a  n1-standard-4                10.128.0.2   35.xxx.xxx.xxx RUNNING
```

#### Step 3.2: SSH into the VM

```bash
gcloud compute ssh mimic-preprocessing --zone=us-central1-a --project=mimic-cxr-pred
```

#### Step 3.3: Setup Environment on the VM

Once SSH'd into the VM, run these commands:

```bash
# Update system
sudo apt-get update
sudo apt-get install -y python3-pip git wget

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda3
rm ~/miniconda.sh
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh

# Create conda environment
conda create -n mimic_env python=3.11 -y
conda activate mimic_env

# Clone repository
cd ~
git clone https://github.com/cam-berger/MIMIC-IV-CXR-ED-Anomaly-Detection.git
cd MIMIC-IV-CXR-ED-Anomaly-Detection

# Install dependencies
pip install -r requirements.txt

# Authenticate with GCP (should already be authenticated via VM service account)
gcloud config set project mimic-cxr-pred
```

#### Step 3.4: Run the Full Pipeline

```bash
# Still on the VM, in the repository directory
python src/run_full_pipeline.py \
  --gcs-bucket bergermimiciv \
  --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
  --gcs-project-id mimic-cxr-pred \
  --mimic-iv-path physionet.org/files/mimiciv/3.1 \
  --mimic-ed-path physionet.org/files/mimic-iv-ed/2.2 \
  --output-path processed/phase1_final \
  --aggressive-filtering \
  --image-size 518 \
  --max-text-length 8192
```

**Expected output:**
```
================================================================================
MIMIC-IV-CXR-ED FULL PREPROCESSING PIPELINE
================================================================================
GCS Bucket: bergermimiciv
CXR Bucket: mimic-cxr-jpg-2.1.0.physionet.org
Project ID: mimic-cxr-pred
Output Path: gs://bergermimiciv/processed/phase1_final
================================================================================
================================================================================
PHASE 1: DATA PREPROCESSING
================================================================================
...
```

---

## Step 4: Monitor Pipeline Progress

### View Real-time Logs

From your local machine:

```bash
# SSH and monitor logs
gcloud compute ssh mimic-preprocessing --zone=us-central1-a --project=mimic-cxr-pred

# On the VM:
tail -f /var/log/mimic-preprocessing.log
# OR
journalctl -u google-startup-scripts.service -f
```

### Check Output in GCS

From your local machine:

```bash
# List output files
gsutil ls gs://bergermimiciv/processed/phase1_final/

# View processing logs
gsutil ls gs://bergermimiciv/logs/

# Download and view metadata
gsutil cat gs://bergermimiciv/processed/phase1_final/filtering_metadata.json
```

---

## Step 5: Verify Completion

### Check Pipeline Status

```bash
# SSH into VM
gcloud compute ssh mimic-preprocessing --zone=us-central1-a --project=mimic-cxr-pred

# Check if pipeline is still running
ps aux | grep python

# View end of log
tail -100 /var/log/mimic-preprocessing.log
```

### Verify Output Files

```bash
# From local machine
gsutil ls -lh gs://bergermimiciv/processed/phase1_final/

# Expected output:
# train_data.pkl
# val_data.pkl
# test_data.pkl
# filtering_metadata.json
```

### Check Metadata

```bash
gsutil cat gs://bergermimiciv/processed/phase1_final/filtering_metadata.json
```

Expected JSON:
```json
{
  "filtering_config": {
    "aggressive": true,
    "use_nlp_model": false
  },
  "split_statistics": {
    "train": {...},
    "val": {...},
    "test": {...}
  },
  "total_records": 12345,
  "total_with_findings": 8901
}
```

---

## Step 6: Cleanup (Important - Save Costs!)

### Shutdown the VM

If using **automated deployment** (Option A), the VM will auto-shutdown when complete.

If using **manual deployment** (Option B):

```bash
# Stop VM (can be restarted later)
gcloud compute instances stop mimic-preprocessing \
  --zone=us-central1-a \
  --project=mimic-cxr-pred

# OR Delete VM permanently
gcloud compute instances delete mimic-preprocessing \
  --zone=us-central1-a \
  --project=mimic-cxr-pred
```

---

## Troubleshooting

### Pipeline Fails

```bash
# SSH into VM
gcloud compute ssh mimic-preprocessing --zone=us-central1-a --project=mimic-cxr-pred

# Check error logs
tail -200 /var/log/mimic-preprocessing.log

# Check Python errors
journalctl -u google-startup-scripts.service | grep -A 20 "Error"
```

### Out of Memory

If you get OOM errors with `n1-standard-4` (15GB RAM):

**Quick fix:** Reduce resource requirements:

```bash
python src/run_full_pipeline.py \
  --gcs-bucket bergermimiciv \
  --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
  --gcs-project-id mimic-cxr-pred \
  --mimic-iv-path physionet.org/files/mimiciv/3.1 \
  --mimic-ed-path physionet.org/files/mimic-iv-ed/2.2 \
  --output-path processed/phase1_final \
  --aggressive-filtering \
  --image-size 224 \      # Reduced from 518
  --max-text-length 512   # Reduced from 8192
```

**Better fix:** Request quota increase for larger VM or delete old VMs to use `n1-highmem-8`.

### Authentication Errors

```bash
# On the VM, verify authentication
gcloud auth list

# Should show the VM's service account
# If not, set up manually:
gcloud auth application-default login
```

### Quota Exceeded

```bash
# List all VMs
gcloud compute instances list --project=mimic-cxr-pred

# Delete old/unused VMs
gcloud compute instances delete OLD_VM_NAME \
  --zone=us-central1-a \
  --project=mimic-cxr-pred
```

---

## Expected Timeline

- **VM Creation:** 2-3 minutes
- **Environment Setup:** 5-10 minutes
- **Pipeline Execution:** 40-80 hours (for full MIMIC dataset)
- **Total:** ~2-4 days for complete run

---

## Cost Estimate

- **VM (`n1-standard-4`):** $0.19/hour × 80 hours = **~$15**
- **Storage:** ~50GB × $0.02/GB/month = **$1/month**
- **PhysioNet Egress:** ~20GB × $0.12/GB = **$2.40**

**Total: ~$18-20 for full run**

---

## Quick Reference Commands

```bash
# Deploy automated
bash scripts/deploy_gcp.sh mimic-cxr-pred bergermimiciv

# SSH into VM
gcloud compute ssh mimic-preprocessing --zone=us-central1-a

# Monitor logs
tail -f /var/log/mimic-preprocessing.log

# Check output
gsutil ls gs://bergermimiciv/processed/phase1_final/

# Stop VM
gcloud compute instances stop mimic-preprocessing --zone=us-central1-a

# Delete VM
gcloud compute instances delete mimic-preprocessing --zone=us-central1-a
```

---

## Next Steps

After pipeline completes successfully:

1. **Verify output:**
   ```bash
   gsutil ls -lh gs://bergermimiciv/processed/phase1_final/
   gsutil cat gs://bergermimiciv/processed/phase1_final/filtering_metadata.json
   ```

2. **Download for local development (optional):**
   ```bash
   gsutil -m cp -r gs://bergermimiciv/processed/phase1_final/ ./data/
   ```

3. **Proceed to Phase 2:** Model training

---

## Support

- **Full Documentation:** [docs/GCP_DEPLOYMENT.md](docs/GCP_DEPLOYMENT.md)
- **Local Testing:** [LOCAL_TESTING.md](LOCAL_TESTING.md)
- **GitHub Issues:** https://github.com/cam-berger/MIMIC-IV-CXR-ED-Anomaly-Detection/issues
