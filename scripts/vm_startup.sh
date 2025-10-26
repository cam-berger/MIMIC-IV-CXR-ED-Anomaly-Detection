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
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3-pip python3-dev python3-venv git wget curl build-essential

# Create Python virtual environment (simpler than conda, no TOS issues)
echo "[2/8] Creating Python virtual environment..."
python3 -m venv /opt/mimic_env

# Activate virtual environment
source /opt/mimic_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Clone repository
echo "[3/8] Cloning repository..."
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
echo "[4/8] Installing Python dependencies..."
pip install -r requirements.txt

# Download spaCy models
echo "[5/8] Downloading spaCy language models..."
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_md-0.5.1.tar.gz

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
