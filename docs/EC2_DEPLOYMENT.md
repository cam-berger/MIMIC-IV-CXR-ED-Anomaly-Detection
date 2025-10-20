# EC2 Deployment Guide for Phase 1 Preprocessing

This guide covers deploying the MIMIC-IV-CXR-ED Phase 1 preprocessing pipeline on AWS EC2.

## Prerequisites

### 1. AWS Account Setup
- AWS account with appropriate permissions
- IAM role with S3 read/write access
- SSH key pair for EC2 access

### 2. S3 Bucket Setup
```bash
# Create S3 bucket for MIMIC data (if not already done)
aws s3 mb s3://your-mimic-data-bucket --region us-east-1

# Upload MIMIC data to S3 (if not already uploaded)
aws s3 sync /local/path/to/physionet.org/files/ s3://your-mimic-data-bucket/physionet.org/files/
```

## EC2 Instance Setup

### 1. Launch EC2 Instance

**Recommended Instance Type:** `r5.2xlarge` or larger
- **vCPUs:** 8
- **Memory:** 64 GB (important for processing large files)
- **Storage:** 100 GB EBS (gp3)
- **AMI:** Ubuntu 22.04 LTS or Amazon Linux 2023

**Launch Command:**
```bash
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type r5.2xlarge \
  --key-name your-keypair \
  --security-group-ids sg-xxxxxxxxx \
  --subnet-id subnet-xxxxxxxxx \
  --iam-instance-profile Name=MIMIC-Processor-Role \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=MIMIC-Phase1-Processor}]'
```

### 2. Configure IAM Role

Create an IAM role with the following policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::your-mimic-data-bucket",
        "arn:aws:s3:::your-mimic-data-bucket/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-output-bucket/*"
      ]
    }
  ]
}
```

### 3. SSH into Instance

```bash
ssh -i your-keypair.pem ubuntu@<instance-public-ip>
```

## Software Installation

### 1. Update System
```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Python 3.11+
```bash
sudo apt install -y python3.11 python3.11-venv python3-pip git
```

### 3. Clone Repository
```bash
git clone https://github.com/yourusername/MIMIC-IV-CXR-ED-Anomaly-Detection.git
cd MIMIC-IV-CXR-ED-Anomaly-Detection
```

### 4. Create Virtual Environment
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 5. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Verify AWS Access
```bash
# Instance should have IAM role attached - verify with:
aws s3 ls s3://your-mimic-data-bucket/physionet.org/files/
```

## Running Phase 1

### 1. Test Run (10 subjects)
```bash
python src/phase1_stay_identification.py \
  --base-path physionet.org/files \
  --s3-bucket your-mimic-data-bucket \
  --num-subjects 10 \
  --output-path test/phase1_output \
  --output-s3-bucket your-output-bucket \
  --time-window 24
```

### 2. Monitor Progress
```bash
# In another terminal, monitor system resources
htop

# Monitor memory usage
watch -n 1 free -h

# Check logs
tail -f logs/phase1.log  # if logging is configured
```

### 3. Full Production Run
```bash
# Run with nohup to keep running after SSH disconnect
nohup python src/phase1_stay_identification.py \
  --base-path physionet.org/files \
  --s3-bucket your-mimic-data-bucket \
  --num-subjects 1000 \
  --output-path production/phase1_output \
  --output-s3-bucket your-output-bucket \
  --time-window 24 \
  > phase1_run.log 2>&1 &

# Check process
ps aux | grep phase1
```

### 4. Screen Session (Alternative)
```bash
# Start a screen session
screen -S phase1

# Run the script
python src/phase1_stay_identification.py \
  --base-path physionet.org/files \
  --s3-bucket your-mimic-data-bucket \
  --num-subjects 1000 \
  --output-path production/phase1_output \
  --output-s3-bucket your-output-bucket

# Detach: Ctrl+A, then D
# Reattach: screen -r phase1
```

## Output Verification

### 1. Check S3 Output
```bash
aws s3 ls s3://your-output-bucket/production/phase1_output/

# Expected files:
# - linked_data.parquet
# - train_data.parquet
# - validate_data.parquet
# - test_data.parquet
# - metadata.json
```

### 2. Download Metadata for Verification
```bash
aws s3 cp s3://your-output-bucket/production/phase1_output/metadata.json .
cat metadata.json
```

## Performance Optimization

### 1. Memory Management
- Monitor memory usage during processing
- Adjust `chunk_size` in code if needed (default: 100,000 rows)
- Increase instance size if OOM errors occur

### 2. Network Optimization
- Use VPC endpoints for S3 to avoid data transfer charges
- Consider using S3 Transfer Acceleration for faster uploads

### 3. Cost Optimization
```bash
# Stop instance when not in use
aws ec2 stop-instances --instance-ids i-xxxxxxxxx

# Start when needed
aws ec2 start-instances --instance-ids i-xxxxxxxxx

# Or use spot instances for 70% cost savings
```

## Troubleshooting

### Memory Issues
```bash
# Check memory usage
free -h
df -h

# If running out of memory, reduce num-subjects or use larger instance
```

### S3 Access Issues
```bash
# Verify IAM role
aws sts get-caller-identity

# Test S3 access
aws s3 ls s3://your-mimic-data-bucket/
```

### Python Dependencies
```bash
# Reinstall if issues
pip install --force-reinstall -r requirements.txt
```

## Automation with Cron

Create a cron job for regular processing:

```bash
# Edit crontab
crontab -e

# Add job (run daily at 2 AM)
0 2 * * * cd /home/ubuntu/MIMIC-IV-CXR-ED-Anomaly-Detection && /home/ubuntu/MIMIC-IV-CXR-ED-Anomaly-Detection/venv/bin/python src/phase1_stay_identification.py --base-path physionet.org/files --s3-bucket your-mimic-data-bucket --num-subjects 1000 --output-path production/phase1_$(date +\%Y\%m\%d) --output-s3-bucket your-output-bucket >> /home/ubuntu/phase1_cron.log 2>&1
```

## Estimated Costs

### EC2 Instance (r5.2xlarge)
- On-Demand: ~$0.50/hour
- Spot: ~$0.15/hour (70% savings)

### Processing Time Estimates
- 10 subjects: ~5-10 minutes
- 100 subjects: ~30-60 minutes
- 1000 subjects: ~5-8 hours

### S3 Storage
- Input data: ~5TB (depends on your dataset)
- Output data: ~100GB (estimated for 1000 subjects)

## Next Steps

After Phase 1 completes successfully:

1. Verify output data quality
2. Proceed to Phase 2 (Clinical Extraction)
3. Run Phase 3 (Integration)

## Support

For issues or questions:
- Check logs in `phase1_run.log`
- Review error messages in S3 CloudWatch if configured
- Consult repository documentation
