#!/bin/bash

# Upload MIMIC-ED dataset to S3

set -e

# Use the actual user's home directory
ACTUAL_USER_HOME="/home/dev"
export PATH="$ACTUAL_USER_HOME/.local/bin:$PATH"

# Set AWS credentials path
export AWS_CONFIG_FILE="$ACTUAL_USER_HOME/.aws/config"
export AWS_SHARED_CREDENTIALS_FILE="$ACTUAL_USER_HOME/.aws/credentials"

# Function to run AWS CLI using ml_env conda environment (Python 3.11)
aws_cli() {
    /home/dev/miniconda3/envs/ml_env/bin/python -m awscli "$@"
}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Upload MIMIC-ED Dataset to S3${NC}"
echo -e "${GREEN}========================================${NC}"

# Load environment
source .env

# Validate
if [ -z "$OUTPUT_BUCKET" ] || [ -z "$AWS_REGION" ]; then
    echo -e "${RED}Error: OUTPUT_BUCKET or AWS_REGION not set${NC}"
    exit 1
fi

# Get local path from user
read -p "Enter the local path where you have MIMIC-ED files (e.g., ~/Downloads/mimic-iv-ed): " LOCAL_PATH

# Expand path
LOCAL_PATH="${LOCAL_PATH/#\~/$HOME}"

# Validate path exists
if [ ! -d "$LOCAL_PATH" ]; then
    echo -e "${RED}Error: Directory $LOCAL_PATH does not exist${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Checking for MIMIC-ED files in $LOCAL_PATH...${NC}"

# Expected MIMIC-ED files
EXPECTED_FILES=(
    "edstays.csv.gz"
    "diagnosis.csv.gz"
    "pyxis.csv.gz"
    "triage.csv.gz"
    "vitalsign.csv.gz"
    "medrecon.csv.gz"
)

# Check for files
FILES_FOUND=0
MISSING_FILES=()

for file in "${EXPECTED_FILES[@]}"; do
    if [ -f "$LOCAL_PATH/$file" ]; then
        echo -e "${GREEN}✓ Found $file${NC}"
        FILES_FOUND=$((FILES_FOUND + 1))
    else
        echo -e "${YELLOW}✗ Missing $file${NC}"
        MISSING_FILES+=("$file")
    fi
done

if [ $FILES_FOUND -eq 0 ]; then
    echo -e "${RED}Error: No MIMIC-ED files found in $LOCAL_PATH${NC}"
    echo -e "${YELLOW}Expected files:${NC}"
    for file in "${EXPECTED_FILES[@]}"; do
        echo -e "  - $file"
    done
    exit 1
fi

echo -e "\n${YELLOW}Found $FILES_FOUND file(s)${NC}"

if [ ${#MISSING_FILES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Note: ${#MISSING_FILES[@]} file(s) are missing (will be skipped)${NC}"
fi

# Define S3 destination
S3_DEST="s3://${OUTPUT_BUCKET}/mimic-iv-ed/2.2/"

echo -e "\n${YELLOW}Uploading to: $S3_DEST${NC}"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled"
    exit 0
fi

# Upload files
echo -e "\n${YELLOW}Uploading MIMIC-ED files...${NC}"

for file in "${EXPECTED_FILES[@]}"; do
    if [ -f "$LOCAL_PATH/$file" ]; then
        echo -e "\n${YELLOW}Uploading $file...${NC}"
        aws_cli s3 cp "$LOCAL_PATH/$file" "${S3_DEST}${file}" --region "$AWS_REGION"
        echo -e "${GREEN}✓ $file uploaded${NC}"
    fi
done

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Upload Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "Files uploaded to: $S3_DEST"
echo -e ""
echo -e "Verify with:"
echo -e "  aws s3 ls $S3_DEST"
echo -e ""
