#!/bin/bash

# AWS Setup Script for MIMIC Preprocessing Pipeline
# This script creates all necessary AWS resources

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MIMIC Preprocessing AWS Setup${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    exit 1
fi

# Load configuration
source .env || {
    echo -e "${RED}Error: .env file not found${NC}"
    echo -e "${YELLOW}Please copy .env.example to .env and configure it${NC}"
    exit 1
}

# Validate required variables
if [ -z "$AWS_ACCOUNT_ID" ] || [ -z "$AWS_REGION" ]; then
    echo -e "${RED}Error: AWS_ACCOUNT_ID and AWS_REGION must be set in .env${NC}"
    exit 1
fi

echo -e "${GREEN}Using AWS Account: $AWS_ACCOUNT_ID${NC}"
echo -e "${GREEN}Region: $AWS_REGION${NC}"
echo ""

# Function to check if resource exists
resource_exists() {
    local resource_type=$1
    local resource_name=$2
    
    case $resource_type in
        "s3")
            aws s3 ls "s3://$resource_name" &> /dev/null
            ;;
        "iam-role")
            aws iam get-role --role-name "$resource_name" &> /dev/null
            ;;
        *)
            return 1
            ;;
    esac
}

# Step 1: Create S3 Buckets
echo -e "${YELLOW}Step 1: Creating S3 Buckets...${NC}"

create_bucket() {
    local bucket_name=$1
    
    if resource_exists "s3" "$bucket_name"; then
        echo -e "  Bucket $bucket_name already exists"
    else
        echo -e "  Creating bucket: $bucket_name"
        
        if [ "$AWS_REGION" == "us-east-1" ]; then
            aws s3 mb "s3://$bucket_name" --region "$AWS_REGION"
        else
            aws s3 mb "s3://$bucket_name" --region "$AWS_REGION" \
                --create-bucket-configuration LocationConstraint="$AWS_REGION"
        fi
        
        # Enable versioning
        aws s3api put-bucket-versioning \
            --bucket "$bucket_name" \
            --versioning-configuration Status=Enabled
        
        echo -e "${GREEN}  ✓ Created bucket: $bucket_name${NC}"
    fi
}

create_bucket "$OUTPUT_BUCKET"
create_bucket "$TEMP_BUCKET"

# Step 2: Create IAM Roles
echo -e "\n${YELLOW}Step 2: Creating IAM Roles...${NC}"

# Batch Service Role
echo -e "  Creating Batch Service Role..."

if resource_exists "iam-role" "BatchServiceRole"; then
    echo -e "  BatchServiceRole already exists"
else
    cat > /tmp/batch-service-role-trust.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "batch.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    aws iam create-role \
        --role-name BatchServiceRole \
        --assume-role-policy-document file:///tmp/batch-service-role-trust.json

    aws iam attach-role-policy \
        --role-name BatchServiceRole \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole

    echo -e "${GREEN}  ✓ Created BatchServiceRole${NC}"
fi

# ECS Instance Role
echo -e "  Creating ECS Instance Role..."

if resource_exists "iam-role" "ecsInstanceRole"; then
    echo -e "  ecsInstanceRole already exists"
else
    cat > /tmp/ecs-instance-role-trust.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    aws iam create-role \
        --role-name ecsInstanceRole \
        --assume-role-policy-document file:///tmp/ecs-instance-role-trust.json

    aws iam attach-role-policy \
        --role-name ecsInstanceRole \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role

    # Create instance profile
    aws iam create-instance-profile --instance-profile-name ecsInstanceRole
    aws iam add-role-to-instance-profile \
        --instance-profile-name ecsInstanceRole \
        --role-name ecsInstanceRole

    echo -e "${GREEN}  ✓ Created ecsInstanceRole${NC}"
fi

# Batch Job Role
echo -e "  Creating Batch Job Role..."

if resource_exists "iam-role" "BatchJobRole"; then
    echo -e "  BatchJobRole already exists"
else
    cat > /tmp/batch-job-role-trust.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    aws iam create-role \
        --role-name BatchJobRole \
        --assume-role-policy-document file:///tmp/batch-job-role-trust.json

    # Create policy for S3 access
    cat > /tmp/batch-job-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::physionet-open/*",
        "arn:aws:s3:::$OUTPUT_BUCKET/*",
        "arn:aws:s3:::$TEMP_BUCKET/*",
        "arn:aws:s3:::$OUTPUT_BUCKET",
        "arn:aws:s3:::$TEMP_BUCKET"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
EOF

    aws iam put-role-policy \
        --role-name BatchJobRole \
        --policy-name BatchJobS3Access \
        --policy-document file:///tmp/batch-job-policy.json

    echo -e "${GREEN}  ✓ Created BatchJobRole${NC}"
fi

# Step 3: Create VPC and Security Group (if needed)
echo -e "\n${YELLOW}Step 3: Setting up VPC and Security Group...${NC}"

# Get default VPC
DEFAULT_VPC=$(aws ec2 describe-vpcs \
    --filters "Name=isDefault,Values=true" \
    --query "Vpcs[0].VpcId" \
    --output text)

if [ "$DEFAULT_VPC" == "None" ] || [ -z "$DEFAULT_VPC" ]; then
    echo -e "${RED}  Error: No default VPC found${NC}"
    echo -e "${YELLOW}  Please create a VPC manually or use an existing one${NC}"
    exit 1
fi

echo -e "  Using VPC: $DEFAULT_VPC"

# Get subnets
SUBNETS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$DEFAULT_VPC" \
    --query "Subnets[*].SubnetId" \
    --output text)

echo -e "  Found subnets: $SUBNETS"

# Create security group
SG_NAME="mimic-preprocessing-sg"
SG_ID=$(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$DEFAULT_VPC" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null)

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    echo -e "  Creating security group: $SG_NAME"
    
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "Security group for MIMIC preprocessing" \
        --vpc-id "$DEFAULT_VPC" \
        --query "GroupId" \
        --output text)
    
    # Allow outbound traffic (needed for S3 access)
    aws ec2 authorize-security-group-egress \
        --group-id "$SG_ID" \
        --protocol all \
        --port all \
        --cidr 0.0.0.0/0 2>/dev/null || true
    
    echo -e "${GREEN}  ✓ Created security group: $SG_ID${NC}"
else
    echo -e "  Security group already exists: $SG_ID"
fi

# Step 4: Create ECR Repository
echo -e "\n${YELLOW}Step 4: Creating ECR Repository...${NC}"

ECR_REPO_NAME="mimic-preprocessor"

if aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" &> /dev/null; then
    echo -e "  ECR repository already exists: $ECR_REPO_NAME"
else
    aws ecr create-repository \
        --repository-name "$ECR_REPO_NAME" \
        --image-scanning-configuration scanOnPush=true
    
    echo -e "${GREEN}  ✓ Created ECR repository: $ECR_REPO_NAME${NC}"
fi

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO_NAME}"
echo -e "  ECR URI: $ECR_URI"

# Step 5: Create AWS Batch Compute Environment
echo -e "\n${YELLOW}Step 5: Creating AWS Batch Compute Environment...${NC}"

COMPUTE_ENV_NAME="mimic-preprocessing-env"

if aws batch describe-compute-environments \
    --compute-environments "$COMPUTE_ENV_NAME" \
    --query "computeEnvironments[0].computeEnvironmentName" \
    --output text 2>/dev/null | grep -q "$COMPUTE_ENV_NAME"; then
    echo -e "  Compute environment already exists: $COMPUTE_ENV_NAME"
else
    # Convert subnets to JSON array
    SUBNET_ARRAY=$(echo $SUBNETS | tr ' ' '\n' | jq -R . | jq -s .)
    
    cat > /tmp/compute-env.json << EOF
{
  "computeEnvironmentName": "$COMPUTE_ENV_NAME",
  "type": "MANAGED",
  "state": "ENABLED",
  "computeResources": {
    "type": "EC2",
    "minvCpus": 0,
    "maxvCpus": 256,
    "desiredvCpus": 0,
    "instanceTypes": ["optimal"],
    "subnets": $SUBNET_ARRAY,
    "securityGroupIds": ["$SG_ID"],
    "instanceRole": "arn:aws:iam::${AWS_ACCOUNT_ID}:instance-profile/ecsInstanceRole"
  },
  "serviceRole": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/BatchServiceRole"
}
EOF

    aws batch create-compute-environment --cli-input-json file:///tmp/compute-env.json
    
    echo -e "${GREEN}  ✓ Created compute environment: $COMPUTE_ENV_NAME${NC}"
    echo -e "  Waiting for compute environment to become VALID..."
    
    aws batch wait compute-environment-valid --compute-environments "$COMPUTE_ENV_NAME"
fi

# Step 6: Create Job Queue
echo -e "\n${YELLOW}Step 6: Creating AWS Batch Job Queue...${NC}"

JOB_QUEUE_NAME="mimic-preprocessing-queue"

if aws batch describe-job-queues \
    --job-queues "$JOB_QUEUE_NAME" \
    --query "jobQueues[0].jobQueueName" \
    --output text 2>/dev/null | grep -q "$JOB_QUEUE_NAME"; then
    echo -e "  Job queue already exists: $JOB_QUEUE_NAME"
else
    aws batch create-job-queue \
        --job-queue-name "$JOB_QUEUE_NAME" \
        --state ENABLED \
        --priority 1 \
        --compute-environment-order order=1,computeEnvironment="$COMPUTE_ENV_NAME"
    
    echo -e "${GREEN}  ✓ Created job queue: $JOB_QUEUE_NAME${NC}"
fi

# Step 7: Update .env with created resources
echo -e "\n${YELLOW}Step 7: Updating .env file...${NC}"

cat >> .env << EOF

# Auto-generated AWS resources
VPC_ID=$DEFAULT_VPC
SUBNET_IDS=$SUBNETS
SECURITY_GROUP_ID=$SG_ID
ECR_REPOSITORY_URI=$ECR_URI
COMPUTE_ENVIRONMENT_NAME=$COMPUTE_ENV_NAME
JOB_QUEUE_NAME=$JOB_QUEUE_NAME
EOF

echo -e "${GREEN}  ✓ Updated .env file${NC}"

# Step 8: Summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}AWS Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e ""
echo -e "Resources created:"
echo -e "  - S3 Buckets: $OUTPUT_BUCKET, $TEMP_BUCKET"
echo -e "  - IAM Roles: BatchServiceRole, ecsInstanceRole, BatchJobRole"
echo -e "  - Security Group: $SG_ID"
echo -e "  - ECR Repository: $ECR_URI"
echo -e "  - Batch Compute Environment: $COMPUTE_ENV_NAME"
echo -e "  - Batch Job Queue: $JOB_QUEUE_NAME"
echo -e ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Build and push Docker image: ./scripts/deploy_to_aws.sh"
echo -e "  2. Run preprocessing pipeline: python scripts/run_local.py"
echo -e ""

# Cleanup temp files
rm -f /tmp/batch-*.json /tmp/ecs-*.json /tmp/compute-env.json