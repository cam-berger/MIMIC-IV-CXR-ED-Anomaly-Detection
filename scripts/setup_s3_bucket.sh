#!/bin/bash

# Setup S3 bucket for MIMIC preprocessing outputs

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
echo -e "${GREEN}Setting up S3 Output Bucket${NC}"
echo -e "${GREEN}========================================${NC}"

# Load environment
source .env

# Validate
if [ -z "$OUTPUT_BUCKET" ] || [ -z "$AWS_REGION" ]; then
    echo -e "${RED}Error: OUTPUT_BUCKET or AWS_REGION not set${NC}"
    exit 1
fi

# Check if bucket exists
echo -e "\n${YELLOW}Checking if bucket exists...${NC}"
if aws_cli s3 ls "s3://${OUTPUT_BUCKET}" 2>/dev/null; then
    echo -e "${GREEN}✓ Bucket ${OUTPUT_BUCKET} already exists${NC}"
else
    # Create bucket
    echo -e "\n${YELLOW}Creating bucket ${OUTPUT_BUCKET}...${NC}"

    if [ "$AWS_REGION" == "us-east-1" ]; then
        # us-east-1 doesn't need LocationConstraint
        aws_cli s3 mb "s3://${OUTPUT_BUCKET}"
    else
        # Other regions need LocationConstraint
        aws_cli s3 mb "s3://${OUTPUT_BUCKET}" --region "$AWS_REGION"
    fi

    echo -e "${GREEN}✓ Bucket created${NC}"
fi

# Set up bucket structure (optional - will be created automatically by pipeline)
echo -e "\n${YELLOW}Setting up folder structure...${NC}"

# Create marker files to establish folder structure
echo "Folder for Phase 1 results" | aws_cli s3 cp - "s3://${OUTPUT_BUCKET}/processing/phase1_results/.folder"
echo "Folder for Phase 2 results" | aws_cli s3 cp - "s3://${OUTPUT_BUCKET}/processing/phase2_results/.folder"
echo "Folder for chunks" | aws_cli s3 cp - "s3://${OUTPUT_BUCKET}/processing/chunks/.folder"
echo "Folder for patient data" | aws_cli s3 cp - "s3://${OUTPUT_BUCKET}/patients/.folder"
echo "Folder for index files" | aws_cli s3 cp - "s3://${OUTPUT_BUCKET}/index/.folder"

echo -e "${GREEN}✓ Folder structure created${NC}"

# Enable versioning (optional but recommended)
echo -e "\n${YELLOW}Enabling bucket versioning...${NC}"
aws_cli s3api put-bucket-versioning \
    --bucket "$OUTPUT_BUCKET" \
    --versioning-configuration Status=Enabled \
    2>/dev/null || echo "Versioning already enabled or not permitted"

echo -e "${GREEN}✓ Versioning enabled${NC}"

# Set lifecycle policy to clean up old chunks (optional)
echo -e "\n${YELLOW}Setting up lifecycle policy for temp files...${NC}"

cat > /tmp/lifecycle-policy.json << 'EOF'
{
  "Rules": [
    {
      "Id": "DeleteOldChunks",
      "Status": "Enabled",
      "Prefix": "processing/chunks/",
      "Expiration": {
        "Days": 7
      }
    }
  ]
}
EOF

aws_cli s3api put-bucket-lifecycle-configuration \
    --bucket "$OUTPUT_BUCKET" \
    --lifecycle-configuration file:///tmp/lifecycle-policy.json \
    2>/dev/null || echo "Could not set lifecycle policy (permissions may be required)"

rm -f /tmp/lifecycle-policy.json

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}S3 Bucket Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "Bucket: s3://${OUTPUT_BUCKET}"
echo -e "Region: ${AWS_REGION}"
echo -e ""
echo -e "Bucket structure:"
echo -e "  - s3://${OUTPUT_BUCKET}/processing/phase1_results/"
echo -e "  - s3://${OUTPUT_BUCKET}/processing/phase2_results/"
echo -e "  - s3://${OUTPUT_BUCKET}/processing/chunks/"
echo -e "  - s3://${OUTPUT_BUCKET}/patients/"
echo -e "  - s3://${OUTPUT_BUCKET}/index/"
echo -e ""
echo -e "${GREEN}You can now run the preprocessing pipeline!${NC}"
echo -e ""
