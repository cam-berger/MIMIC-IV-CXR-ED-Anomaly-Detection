#!/bin/bash

# Deploy Docker image to AWS ECR and register Batch job definition

set -e

# Ensure AWS CLI is in PATH
# Use the actual user's home directory, not root's
ACTUAL_USER_HOME="/home/dev"
export PATH="$ACTUAL_USER_HOME/.local/bin:$PATH"

# Set AWS credentials path to use dev user's credentials even if running as root
export AWS_CONFIG_FILE="$ACTUAL_USER_HOME/.aws/config"
export AWS_SHARED_CREDENTIALS_FILE="$ACTUAL_USER_HOME/.aws/credentials"

# Function to run AWS CLI using ml_env conda environment (Python 3.11)
# Python 3.13 is not yet supported by awscli
aws_cli() {
    /home/dev/miniconda3/envs/ml_env/bin/python -m awscli "$@"
}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Deploying to AWS${NC}"
echo -e "${GREEN}========================================${NC}"

# Load environment
source .env

# Validate
if [ -z "$AWS_ACCOUNT_ID" ] || [ -z "$AWS_REGION" ]; then
    echo -e "${RED}Error: AWS configuration missing${NC}"
    exit 1
fi

if [ -z "$ECR_REPOSITORY_URI" ]; then
    echo -e "${RED}Error: ECR_REPOSITORY_URI not set${NC}"
    echo -e "${YELLOW}Run ./scripts/setup_aws.sh first${NC}"
    exit 1
fi

# Step 1: Build Docker image
echo -e "\n${YELLOW}Step 1: Building Docker image...${NC}"

docker build -t mimic-preprocessor:latest -f docker/Dockerfile .

echo -e "${GREEN}✓ Docker image built${NC}"

# Step 2: Login to ECR
echo -e "\n${YELLOW}Step 2: Logging in to ECR...${NC}"

aws_cli ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin \
    "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo -e "${GREEN}✓ Logged in to ECR${NC}"

# Step 3: Tag image
echo -e "\n${YELLOW}Step 3: Tagging image...${NC}"

docker tag mimic-preprocessor:latest "$ECR_REPOSITORY_URI:latest"

echo -e "${GREEN}✓ Image tagged${NC}"

# Step 4: Push to ECR
echo -e "\n${YELLOW}Step 4: Pushing to ECR...${NC}"

docker push "$ECR_REPOSITORY_URI:latest"

echo -e "${GREEN}✓ Image pushed to ECR${NC}"

# Step 5: Register/Update Job Definition
echo -e "\n${YELLOW}Step 5: Registering Batch job definition...${NC}"

# Create job definition JSON
cat > /tmp/job-definition.json << EOF
{
  "jobDefinitionName": "mimic-preprocess-job",
  "type": "container",
  "containerProperties": {
    "image": "$ECR_REPOSITORY_URI:latest",
    "vcpus": 4,
    "memory": 16000,
    "jobRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/BatchJobRole",
    "environment": [
      {
        "name": "OUTPUT_BUCKET",
        "value": "$OUTPUT_BUCKET"
      },
      {
        "name": "TEMP_BUCKET",
        "value": "$TEMP_BUCKET"
      },
      {
        "name": "AWS_REGION",
        "value": "$AWS_REGION"
      }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/aws/batch/mimic-preprocessing",
        "awslogs-region": "$AWS_REGION",
        "awslogs-stream-prefix": "mimic"
      }
    }
  }
}
EOF

# Create log group if it doesn't exist
aws_cli logs create-log-group \
    --log-group-name /aws/batch/mimic-preprocessing \
    2>/dev/null || true

# Register job definition
aws_cli batch register-job-definition \
    --cli-input-json file:///tmp/job-definition.json

echo -e "${GREEN}✓ Job definition registered${NC}"

# Cleanup
rm /tmp/job-definition.json

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "Your pipeline is now deployed to AWS"
echo -e "ECR Image: $ECR_REPOSITORY_URI:latest"
echo -e ""