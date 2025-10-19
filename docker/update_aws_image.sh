#!/bin/bash
set -e

echo "=========================================="
echo "Updating MIMIC Preprocessor Docker Image"
echo "=========================================="

# Configuration
AWS_REGION="us-west-2"
AWS_PROFILE="default"
AWS_ACCOUNT_ID="851454408197"
ECR_REPO="mimic-preprocessor"
VERSION_TAG="v1.1-nlp"

# Get to project root
cd "$(dirname "$0")/.."

echo ""
echo "Step 1: Building Docker image..."
docker build -t ${ECR_REPO}:latest -f docker/Dockerfile .

echo ""
echo "Step 2: Authenticating with ECR..."
aws ecr get-login-password --region ${AWS_REGION} --profile ${AWS_PROFILE} | \
    docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

echo ""
echo "Step 3: Tagging images..."
docker tag ${ECR_REPO}:latest \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest

docker tag ${ECR_REPO}:latest \
    ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${VERSION_TAG}

echo ""
echo "Step 4: Pushing to ECR..."
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${VERSION_TAG}

echo ""
echo "=========================================="
echo "✓ Image updated successfully!"
echo "=========================================="
echo ""
echo "Images pushed:"
echo "  • ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:latest"
echo "  • ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${VERSION_TAG}"
echo ""
echo "To verify:"
echo "  aws ecr describe-images --repository-name ${ECR_REPO} --region ${AWS_REGION} --profile ${AWS_PROFILE}"
echo ""
