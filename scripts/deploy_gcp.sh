#!/bin/bash
#
# Deploy MIMIC preprocessing pipeline to Google Cloud Platform
#
# This script:
# 1. Creates a GCP Compute Engine VM with GPU (optional)
# 2. Sets up the environment
# 3. Runs the full preprocessing pipeline
# 4. Shuts down the VM when complete
#
# Usage:
#   bash scripts/deploy_gcp.sh YOUR_PROJECT_ID YOUR_BUCKET_NAME [TEST_MODE]
#
# Arguments:
#   YOUR_PROJECT_ID  - GCP project ID (required)
#   YOUR_BUCKET_NAME - GCS bucket name (default: bergermimiciv)
#   TEST_MODE        - Optional: "test" to run with only 10 batches
#
# Examples:
#   # Test run (10 batches, ~10-30 minutes, <$1)
#   bash scripts/deploy_gcp.sh mimic-cxr-pred bergermimiciv test
#
#   # Full run (all batches, ~2-4 days, ~$18)
#   bash scripts/deploy_gcp.sh mimic-cxr-pred bergermimiciv
#

set -e  # Exit on error

# Configuration
PROJECT_ID="${1:-}"
BUCKET_NAME="${2:-bergermimiciv}"
TEST_MODE="${3:-}"
VM_NAME="mimic-preprocessing-$(date +%Y%m%d-%H%M%S)"
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"  # 4 vCPUs, 15GB RAM
BOOT_DISK_SIZE="200GB"
GIT_REPO="https://github.com/cam-berger/MIMIC-IV-CXR-ED-Anomaly-Detection.git"  # Update this!

# Test mode configuration
if [ "$TEST_MODE" == "test" ]; then
    MAX_BATCHES="10"
    OUTPUT_PATH="processed/phase1_test"
    VM_NAME="mimic-preprocessing-test-$(date +%Y%m%d-%H%M%S)"
else
    MAX_BATCHES=""
    OUTPUT_PATH="processed/phase1_final"
fi

# Optional: GPU configuration (uncomment to enable)
# GPU_TYPE="nvidia-tesla-t4"
# GPU_COUNT="1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if project ID provided
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: Project ID required${NC}"
    echo "Usage: bash scripts/deploy_gcp.sh YOUR_PROJECT_ID [BUCKET_NAME]"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MIMIC Preprocessing GCP Deployment${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Project ID: $PROJECT_ID"
echo "Bucket: $BUCKET_NAME"
echo "VM Name: $VM_NAME"
echo "Zone: $ZONE"
echo "Machine Type: $MACHINE_TYPE"
if [ "$TEST_MODE" == "test" ]; then
    echo -e "${YELLOW}Mode: TEST (10 batches only)${NC}"
    echo "Max Batches: $MAX_BATCHES"
else
    echo -e "${GREEN}Mode: FULL PRODUCTION RUN${NC}"
fi
echo "Output Path: $OUTPUT_PATH"
echo -e "${GREEN}========================================${NC}"

# Set project
echo -e "${YELLOW}Setting GCP project...${NC}"
gcloud config set project "$PROJECT_ID"

# Create VM
echo -e "${YELLOW}Creating Compute Engine VM...${NC}"

CREATE_CMD="gcloud compute instances create $VM_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=pd-standard \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --scopes=cloud-platform \
    --metadata-from-file=startup-script=scripts/vm_startup.sh \
    --metadata=bucket-name=$BUCKET_NAME,project-id=$PROJECT_ID,git-repo=$GIT_REPO,max-batches=$MAX_BATCHES,output-path=$OUTPUT_PATH"

# Add GPU if configured
if [ -n "$GPU_TYPE" ]; then
    CREATE_CMD="$CREATE_CMD --accelerator=type=$GPU_TYPE,count=$GPU_COUNT --maintenance-policy=TERMINATE"
fi

eval $CREATE_CMD

echo -e "${GREEN}VM created: $VM_NAME${NC}"

# Wait for VM to be ready
echo -e "${YELLOW}Waiting for VM to start...${NC}"
sleep 30

# Show how to monitor
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}VM is being provisioned!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
if [ "$TEST_MODE" == "test" ]; then
    echo -e "${YELLOW}TEST MODE: Processing only 10 batches${NC}"
    echo -e "${YELLOW}Estimated time: 10-30 minutes${NC}"
    echo -e "${YELLOW}Estimated cost: <$1${NC}"
    echo ""
fi
echo "Monitor startup progress:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -f /var/log/syslog | grep startup-script'"
echo ""
echo "SSH into VM:"
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE"
echo ""
echo "View serial port output:"
echo "  gcloud compute instances get-serial-port-output $VM_NAME --zone=$ZONE"
echo ""
echo "The VM will automatically:"
echo "  1. Install dependencies"
echo "  2. Clone repository"
if [ "$TEST_MODE" == "test" ]; then
    echo "  3. Run preprocessing with 10 batches (TEST MODE)"
else
    echo "  3. Run full preprocessing pipeline (all batches)"
fi
echo "  4. Shut down when complete (to save costs)"
echo ""
echo -e "${YELLOW}IMPORTANT: The VM will auto-shutdown when pipeline completes!${NC}"
echo -e "${YELLOW}To prevent auto-shutdown, SSH in and: sudo touch /tmp/no-shutdown${NC}"
echo ""
echo "Check pipeline output in GCS:"
echo "  gsutil ls gs://$BUCKET_NAME/$OUTPUT_PATH/"
echo ""
if [ "$TEST_MODE" == "test" ]; then
    echo -e "${YELLOW}After test succeeds, run full pipeline:${NC}"
    echo "  bash scripts/deploy_gcp.sh $PROJECT_ID $BUCKET_NAME"
    echo ""
fi
echo -e "${GREEN}========================================${NC}"
