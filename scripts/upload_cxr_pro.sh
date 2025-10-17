#!/bin/bash

# Upload CXR-PRO dataset to S3

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
echo -e "${GREEN}Upload CXR-PRO Dataset to S3${NC}"
echo -e "${GREEN}========================================${NC}"

# Load environment
source .env

# Validate
if [ -z "$OUTPUT_BUCKET" ] || [ -z "$AWS_REGION" ]; then
    echo -e "${RED}Error: OUTPUT_BUCKET or AWS_REGION not set${NC}"
    exit 1
fi

# Get local path from user
read -p "Enter the local path where you downloaded CXR-PRO files (e.g., ~/Downloads/cxr-pro): " LOCAL_PATH

# Expand path
LOCAL_PATH="${LOCAL_PATH/#\~/$HOME}"

# Validate path exists
if [ ! -d "$LOCAL_PATH" ]; then
    echo -e "${RED}Error: Directory $LOCAL_PATH does not exist${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Checking for CXR-PRO files in $LOCAL_PATH...${NC}"

# Check for required files
FILES_FOUND=0
if [ -f "$LOCAL_PATH/cxr.h5" ]; then
    echo -e "${GREEN}✓ Found cxr.h5${NC}"
    FILES_FOUND=$((FILES_FOUND + 1))
fi

if [ -f "$LOCAL_PATH/mimic_train_impressions.csv" ]; then
    echo -e "${GREEN}✓ Found mimic_train_impressions.csv${NC}"
    FILES_FOUND=$((FILES_FOUND + 1))
fi

if [ -f "$LOCAL_PATH/mimic_test_impressions.csv" ]; then
    echo -e "${GREEN}✓ Found mimic_test_impressions.csv${NC}"
    FILES_FOUND=$((FILES_FOUND + 1))
fi

if [ $FILES_FOUND -eq 0 ]; then
    echo -e "${RED}Error: No CXR-PRO files found in $LOCAL_PATH${NC}"
    echo -e "${YELLOW}Expected files:${NC}"
    echo -e "  - cxr.h5"
    echo -e "  - mimic_train_impressions.csv"
    echo -e "  - mimic_test_impressions.csv"
    exit 1
fi

echo -e "\n${YELLOW}Found $FILES_FOUND file(s)${NC}"

# Define S3 destination
S3_DEST="s3://${OUTPUT_BUCKET}/cxr-pro/1.0.0/"

echo -e "\n${YELLOW}Uploading to: $S3_DEST${NC}"
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled"
    exit 0
fi

# Upload files
echo -e "\n${YELLOW}Uploading CXR-PRO files...${NC}"

if [ -f "$LOCAL_PATH/cxr.h5" ]; then
    echo -e "\n${YELLOW}Uploading cxr.h5 (this may take a while for large files)...${NC}"
    aws_cli s3 cp "$LOCAL_PATH/cxr.h5" "${S3_DEST}cxr.h5" --region "$AWS_REGION"
    echo -e "${GREEN}✓ cxr.h5 uploaded${NC}"
fi

if [ -f "$LOCAL_PATH/mimic_train_impressions.csv" ]; then
    echo -e "\n${YELLOW}Uploading mimic_train_impressions.csv...${NC}"
    aws_cli s3 cp "$LOCAL_PATH/mimic_train_impressions.csv" "${S3_DEST}mimic_train_impressions.csv" --region "$AWS_REGION"
    echo -e "${GREEN}✓ mimic_train_impressions.csv uploaded${NC}"
fi

if [ -f "$LOCAL_PATH/mimic_test_impressions.csv" ]; then
    echo -e "\n${YELLOW}Uploading mimic_test_impressions.csv...${NC}"
    aws_cli s3 cp "$LOCAL_PATH/mimic_test_impressions.csv" "${S3_DEST}mimic_test_impressions.csv" --region "$AWS_REGION"
    echo -e "${GREEN}✓ mimic_test_impressions.csv uploaded${NC}"
fi

# Upload any other CSV or metadata files
echo -e "\n${YELLOW}Checking for additional CSV files...${NC}"
ADDITIONAL_FILES=$(find "$LOCAL_PATH" -maxdepth 1 -name "*.csv" ! -name "mimic_train_impressions.csv" ! -name "mimic_test_impressions.csv" 2>/dev/null || true)

if [ ! -z "$ADDITIONAL_FILES" ]; then
    echo -e "${YELLOW}Found additional CSV files:${NC}"
    echo "$ADDITIONAL_FILES"
    read -p "Upload these files too? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        while IFS= read -r file; do
            filename=$(basename "$file")
            echo -e "${YELLOW}Uploading $filename...${NC}"
            aws_cli s3 cp "$file" "${S3_DEST}${filename}" --region "$AWS_REGION"
            echo -e "${GREEN}✓ $filename uploaded${NC}"
        done <<< "$ADDITIONAL_FILES"
    fi
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Upload Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "Files uploaded to: $S3_DEST"
echo -e ""
echo -e "Verify with:"
echo -e "  aws s3 ls $S3_DEST"
echo -e ""
