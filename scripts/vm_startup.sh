#!/bin/bash
#
# GCP VM Startup Script for MIMIC Preprocessing Pipeline
#
# This script runs automatically when the VM starts.
# It sets up the environment and runs the full preprocessing pipeline.
#

set -e  # Exit on error

# Get metadata
PROJECT_ID=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/project-id)
BUCKET_NAME=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket-name)
GIT_REPO=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/git-repo)

# Logging
LOG_FILE="/var/log/mimic-preprocessing.log"
exec 1> >(tee -a "$LOG_FILE")
exec 2>&1

echo "========================================"
echo "MIMIC Preprocessing VM Startup"
echo "========================================"
echo "Time: $(date)"
echo "Project: $PROJECT_ID"
echo "Bucket: $BUCKET_NAME"
echo "========================================"

# Update system
echo "[1/8] Updating system packages..."
apt-get update
apt-get install -y python3-pip python3-dev git wget curl

# Install Miniconda
echo "[2/8] Installing Miniconda..."
if [ ! -d "/opt/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda3
    rm /tmp/miniconda.sh
fi

export PATH="/opt/miniconda3/bin:$PATH"
source /opt/miniconda3/etc/profile.d/conda.sh

# Create conda environment
echo "[3/8] Creating conda environment..."
conda create -n mimic_env python=3.11 -y
conda activate mimic_env

# Clone repository
echo "[4/8] Cloning repository..."
cd /home
if [ -d "MIMIC-IV-CXR-ED-Anomaly-Detection" ]; then
    cd MIMIC-IV-CXR-ED-Anomaly-Detection
    git pull
else
    # If no git repo provided, create a minimal setup
    mkdir -p MIMIC-IV-CXR-ED-Anomaly-Detection
    cd MIMIC-IV-CXR-ED-Anomaly-Detection

    # Download code from GCS bucket if available
    if gsutil -q stat "gs://$BUCKET_NAME/code/src.tar.gz"; then
        echo "Downloading code from GCS..."
        gsutil cp "gs://$BUCKET_NAME/code/src.tar.gz" .
        tar -xzf src.tar.gz
    else
        echo "ERROR: No code repository found!"
        echo "Please either:"
        echo "  1. Set GIT_REPO in deploy script, or"
        echo "  2. Upload code to gs://$BUCKET_NAME/code/src.tar.gz"
        exit 1
    fi
fi

# Install dependencies
echo "[5/8] Installing Python dependencies..."
pip install -r requirements.txt

# Authenticate with GCP (VM has service account with proper permissions)
echo "[6/8] Setting up GCP authentication..."
gcloud config set project "$PROJECT_ID"

# Run preprocessing pipeline
echo "[7/8] Running full preprocessing pipeline..."
echo "This may take several hours..."

python src/run_full_pipeline.py \
    --gcs-bucket "$BUCKET_NAME" \
    --gcs-cxr-bucket mimic-cxr-jpg-2.1.0.physionet.org \
    --gcs-project-id "$PROJECT_ID" \
    --mimic-iv-path physionet.org/files/mimiciv/3.1 \
    --mimic-ed-path physionet.org/files/mimic-iv-ed/2.2 \
    --output-path processed/phase1_final \
    --aggressive-filtering \
    --image-size 518 \
    --max-text-length 8192

PIPELINE_STATUS=$?

if [ $PIPELINE_STATUS -eq 0 ]; then
    echo "========================================"
    echo "PIPELINE COMPLETED SUCCESSFULLY!"
    echo "========================================"
    echo "Output location: gs://$BUCKET_NAME/processed/phase1_final/"
    echo "Time: $(date)"

    # Upload logs to GCS
    gsutil cp "$LOG_FILE" "gs://$BUCKET_NAME/logs/preprocessing-$(date +%Y%m%d-%H%M%S).log"

    # Shutdown VM to save costs (unless /tmp/no-shutdown exists)
    echo "[8/8] Pipeline complete. Checking for auto-shutdown..."
    if [ ! -f "/tmp/no-shutdown" ]; then
        echo "Auto-shutdown enabled. VM will shutdown in 60 seconds..."
        echo "To prevent shutdown: sudo touch /tmp/no-shutdown"
        sleep 60
        shutdown -h now
    else
        echo "Auto-shutdown disabled (/tmp/no-shutdown exists)"
        echo "VM will remain running. Remember to shut it down manually!"
    fi
else
    echo "========================================"
    echo "PIPELINE FAILED!"
    echo "========================================"
    echo "Check logs: $LOG_FILE"
    echo "Time: $(date)"

    # Upload error logs
    gsutil cp "$LOG_FILE" "gs://$BUCKET_NAME/logs/preprocessing-ERROR-$(date +%Y%m%d-%H%M%S).log"

    echo "VM will NOT auto-shutdown due to errors"
    echo "SSH in to debug: gcloud compute ssh $(hostname) --zone=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)"
fi

echo "Startup script complete"
