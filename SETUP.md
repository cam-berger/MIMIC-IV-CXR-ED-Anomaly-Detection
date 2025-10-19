# Setup Instructions

## Prerequisites

1. AWS Account with appropriate permissions
2. PhysioNet account with MIMIC-IV access
3. Python 3.9+
4. Docker installed
5. AWS CLI configured

## Step-by-Step Setup

### 1. Clone Repository

\`\`\`bash
git clone https://github.com/cam-berger/MIMIC-IV-CXR-ED-Anomaly-Detection
cd mimic-multimodal-preprocessor
\`\`\`

### 2. Create Virtual Environment

\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
\`\`\`

### 3. Configure Environment

\`\`\`bash
cp .env.example .env
# Edit .env with your AWS credentials and settings
\`\`\`

### 4. Setup AWS Resources

\`\`\`bash
./scripts/setup_aws.sh
\`\`\`

This creates:
- S3 buckets
- IAM roles
- Batch compute environment
- Job queue
- ECR repository

### 5. Build and Deploy Docker Image

\`\`\`bash
./scripts/deploy_to_aws.sh
\`\`\`

### 6. Run Pipeline

**Local Testing (small subset):**
\`\`\`bash
python scripts/run_local.py --mode local --phase 1
python scripts/run_local.py --mode local --phase 2
python scripts/run_local.py --mode local --phase 3
\`\`\`

**AWS Batch (full dataset):**
\`\`\`bash
python scripts/run_local.py --mode batch --phase 1
python scripts/run_local.py --mode batch --phase 2
python scripts/run_local.py --mode batch --phase 3
\`\`\`

**All phases:**
\`\`\`bash
python scripts/run_local.py --mode batch --all-phases
\`\`\`

## Monitoring

View logs in AWS CloudWatch:
\`\`\`bash
aws logs tail /aws/batch/mimic-preprocessing --follow
\`\`\`

Check job status:
\`\`\`bash
aws batch describe-jobs --jobs <job-id>
\`\`\`

## Troubleshooting

See docs/TROUBLESHOOTING.md
\`\`\`

---

## **Final Steps: Commit and Push Everything**
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Add all files
git add .

# Commit
git commit -m "Complete implementation with AWS integration and deployment scripts"

# Push to GitHub
git push origin feature/phase1-implementation

# Merge to develop
git checkout develop
git merge feature/phase1-implementation
git push origin develop

# Tag release
git tag -a v0.1.0 -m "Initial release with basic preprocessing pipeline"
git push origin v0.1.0