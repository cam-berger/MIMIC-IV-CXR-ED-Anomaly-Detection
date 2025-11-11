#!/bin/bash
# Quick-start script for running Dataflow split creation
# This script provides a convenient way to run the Dataflow pipeline with common configurations

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==================================================================${NC}"
echo -e "${GREEN}  Phase 1 Dataflow Split Creation - Quick Start${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo ""

# Check if config file exists
CONFIG_FILE="dataflow_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: $CONFIG_FILE not found!${NC}"
    echo -e "${YELLOW}Please copy dataflow_config.json.example to dataflow_config.json and fill in your values.${NC}"
    echo ""
    echo "Run:"
    echo "  cp dataflow_config.json.example dataflow_config.json"
    echo "  nano dataflow_config.json  # Edit with your settings"
    exit 1
fi

# Parse config file (requires jq)
if ! command -v jq &> /dev/null; then
    echo -e "${RED}Error: jq is not installed. Please install it:${NC}"
    echo "  Ubuntu/Debian: sudo apt-get install jq"
    echo "  macOS: brew install jq"
    exit 1
fi

PROJECT_ID=$(jq -r '.project_id' $CONFIG_FILE)
GCS_BUCKET=$(jq -r '.gcs_bucket' $CONFIG_FILE)
BATCH_PREFIX=$(jq -r '.batch_files_prefix' $CONFIG_FILE)
OUTPUT_PREFIX=$(jq -r '.output_prefix' $CONFIG_FILE)
RUNNER=$(jq -r '.dataflow.runner' $CONFIG_FILE)
REGION=$(jq -r '.dataflow.region' $CONFIG_FILE)
MAX_WORKERS=$(jq -r '.dataflow.max_num_workers' $CONFIG_FILE)
MACHINE_TYPE=$(jq -r '.dataflow.machine_type' $CONFIG_FILE)
CHUNK_SIZE=$(jq -r '.output_settings.chunk_size' $CONFIG_FILE)

# Validate required fields
if [ "$PROJECT_ID" == "YOUR_PROJECT_ID" ] || [ "$PROJECT_ID" == "null" ]; then
    echo -e "${RED}Error: Please set project_id in $CONFIG_FILE${NC}"
    exit 1
fi

if [ "$GCS_BUCKET" == "YOUR_BUCKET_NAME" ] || [ "$GCS_BUCKET" == "null" ]; then
    echo -e "${RED}Error: Please set gcs_bucket in $CONFIG_FILE${NC}"
    exit 1
fi

echo "Configuration:"
echo "  Project ID:     $PROJECT_ID"
echo "  GCS Bucket:     gs://$GCS_BUCKET"
echo "  Batch Prefix:   $BATCH_PREFIX"
echo "  Output Prefix:  $OUTPUT_PREFIX"
echo "  Runner:         $RUNNER"
echo "  Region:         $REGION"
echo "  Max Workers:    $MAX_WORKERS"
echo "  Machine Type:   $MACHINE_TYPE"
echo "  Chunk Size:     $CHUNK_SIZE"
echo ""

# Mode selection
echo "Select mode:"
echo "  1) Compute stratification indices only (first-time setup)"
echo "  2) Run full pipeline (split creation)"
echo "  3) Test locally with DirectRunner (small subset)"
echo ""
read -p "Enter choice [1-3]: " MODE_CHOICE

case $MODE_CHOICE in
    1)
        echo -e "${GREEN}Mode: Computing stratification indices...${NC}"
        MODE="--compute_indices_only"
        ;;
    2)
        echo -e "${GREEN}Mode: Running full pipeline...${NC}"
        MODE=""
        ;;
    3)
        echo -e "${YELLOW}Mode: Local test with DirectRunner${NC}"
        echo -e "${YELLOW}Note: This will process ALL batches locally (not just a subset)${NC}"
        echo -e "${YELLOW}For true testing, manually copy a few batches to a test prefix first.${NC}"
        read -p "Continue? (y/n): " CONFIRM
        if [ "$CONFIRM" != "y" ]; then
            echo "Aborted."
            exit 0
        fi
        RUNNER="DirectRunner"
        MODE=""
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Checking prerequisites...${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI not found. Please install:${NC}"
    echo "  https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo -e "${YELLOW}Not authenticated. Running gcloud auth login...${NC}"
    gcloud auth login
fi

# Set project
gcloud config set project $PROJECT_ID

# Check if Dataflow API is enabled (only for DataflowRunner)
if [ "$RUNNER" == "DataflowRunner" ]; then
    echo "Checking if Dataflow API is enabled..."
    if ! gcloud services list --enabled --filter="name:dataflow.googleapis.com" --format="value(name)" | grep -q "dataflow"; then
        echo -e "${YELLOW}Dataflow API not enabled. Enabling now...${NC}"
        gcloud services enable dataflow.googleapis.com
        echo -e "${GREEN}Dataflow API enabled.${NC}"
    else
        echo -e "${GREEN}Dataflow API already enabled.${NC}"
    fi
fi

# Check if Python dependencies are installed
echo "Checking Python dependencies..."
if ! python3 -c "import apache_beam" 2>/dev/null; then
    echo -e "${YELLOW}Apache Beam not installed. Installing requirements...${NC}"
    pip install -r requirements_dataflow.txt
fi

echo ""
echo -e "${GREEN}Starting pipeline...${NC}"
echo ""

# Build command
CMD="python3 src/phase1_dataflow_split.py \
    --project_id $PROJECT_ID \
    --gcs_bucket $GCS_BUCKET \
    --batch_files_prefix $BATCH_PREFIX \
    --output_prefix $OUTPUT_PREFIX \
    --region $REGION \
    --runner $RUNNER \
    --max_num_workers $MAX_WORKERS \
    --machine_type $MACHINE_TYPE \
    --chunk_size $CHUNK_SIZE \
    $MODE"

echo "Running command:"
echo "$CMD"
echo ""

# Execute
eval $CMD

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}==================================================================${NC}"
    echo -e "${GREEN}  Pipeline completed successfully!${NC}"
    echo -e "${GREEN}==================================================================${NC}"
    echo ""

    if [ "$MODE" == "--compute_indices_only" ]; then
        echo "Next steps:"
        echo "  1. Verify indices: gsutil cat gs://$GCS_BUCKET/${OUTPUT_PREFIX}stratification_indices.json | jq"
        echo "  2. Run full pipeline: ./run_dataflow_split.sh (choose option 2)"
    else
        echo "Output location: gs://$GCS_BUCKET/$OUTPUT_PREFIX"
        echo ""
        echo "Verify output:"
        echo "  gsutil ls gs://$GCS_BUCKET/${OUTPUT_PREFIX}*_chunk_*.pt | head -10"
        echo ""
        echo "Count chunks:"
        echo "  echo \"Train:\" \$(gsutil ls gs://$GCS_BUCKET/${OUTPUT_PREFIX}train_chunk_*.pt | wc -l)"
        echo "  echo \"Val:\" \$(gsutil ls gs://$GCS_BUCKET/${OUTPUT_PREFIX}val_chunk_*.pt | wc -l)"
        echo "  echo \"Test:\" \$(gsutil ls gs://$GCS_BUCKET/${OUTPUT_PREFIX}test_chunk_*.pt | wc -l)"
    fi

    if [ "$RUNNER" == "DataflowRunner" ]; then
        echo ""
        echo "View job in Cloud Console:"
        echo "  https://console.cloud.google.com/dataflow/jobs?project=$PROJECT_ID"
    fi
else
    echo -e "${RED}==================================================================${NC}"
    echo -e "${RED}  Pipeline failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}==================================================================${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check logs above for error messages"
    echo "  2. Verify GCS paths: gsutil ls gs://$GCS_BUCKET/$BATCH_PREFIX*.pt"
    echo "  3. Check permissions: gcloud projects get-iam-policy $PROJECT_ID"
    echo "  4. See docs/DATAFLOW_SETUP.md for detailed troubleshooting"
    exit $EXIT_CODE
fi
