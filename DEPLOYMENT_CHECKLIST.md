# MIMIC Phase 1 - AWS EC2 Deployment Checklist

## Pre-Deployment Verification

### ✅ Code Updates
- [x] **phase1_stay_identification.py** - Updated with S3 support
- [x] **phase1_preprocess.py** - Updated with S3 support (NEW)
- [x] Removed `--skip-large-files` flag
- [x] Added chunked reading for memory efficiency
- [x] All imports properly declared
- [x] S3Helper class for unified S3/local operations

### ✅ Dependencies
- [x] `requirements.txt` updated with version constraints
- [x] Added `pydicom>=2.4.0` (DICOM image support)
- [x] Added `boto3>=1.28.0` (AWS S3)
- [x] Added `torch>=2.0.0` (Deep learning - NEW)
- [x] Added `torchvision>=0.15.0` (Image transforms - NEW)
- [x] Added `transformers>=4.30.0` (NLP/tokenizers - NEW)
- [x] Added `sentence-transformers>=2.2.0` (Embeddings - NEW)
- [x] Added `faiss-cpu>=1.7.4` (Vector search - NEW)
- [x] Added `opencv-python>=4.8.0` (Image processing - NEW)
- [x] Version upper bounds added for stability
- [x] Created `scripts/install_dependencies.sh` (NEW)
- [x] Created `INSTALL_DEPENDENCIES.md` (NEW)

### ✅ Repository Cleanup
- [x] Removed `__pycache__` directories
- [x] Removed `*.egg-info` directories
- [x] Removed test result files
- [x] Removed `.env.save` backup file
- [x] Updated `.gitignore` to be more comprehensive

### ✅ Documentation
- [x] Created `docs/EC2_DEPLOYMENT.md` guide
- [x] Created `scripts/setup_ec2.sh` automation script
- [x] Created `scripts/verify_deployment.sh` verification script (NEW)
- [x] Created `DEPLOYMENT_SUMMARY.md` (NEW)
- [x] Created `INSTALL_DEPENDENCIES.md` (NEW)

## AWS Prerequisites

### IAM & Permissions
- [ ] IAM role created with S3 read/write permissions
- [ ] IAM role policy includes:
  - `s3:GetObject` on input bucket
  - `s3:ListBucket` on input bucket
  - `s3:PutObject` on output bucket
- [ ] SSH key pair generated/available

### S3 Buckets
- [ ] Input bucket created and populated with MIMIC data
- [ ] Output bucket created
- [ ] Verify bucket structure:
  ```
  s3://your-bucket/physionet.org/files/
    ├── mimiciv/3.1/
    ├── mimic-cxr-jpg/2.1.0/
    ├── reflacx/
    └── mimic-iv-ed/2.2/
  ```

### Network Configuration
- [ ] VPC configured
- [ ] Security group allows SSH (port 22)
- [ ] (Optional) VPC endpoint for S3 created for cost savings

## EC2 Instance Setup

### Instance Configuration
- [ ] Instance type: `r5.2xlarge` or larger (64GB RAM minimum)
- [ ] AMI: Ubuntu 22.04 LTS
- [ ] Storage: 100GB gp3 EBS volume
- [ ] IAM role attached to instance
- [ ] Instance tagged appropriately

### Software Installation
```bash
# On EC2 instance:
wget https://raw.githubusercontent.com/yourusername/MIMIC-IV-CXR-ED-Anomaly-Detection/main/scripts/setup_ec2.sh
chmod +x setup_ec2.sh
./setup_ec2.sh
```

Or manual installation:
- [ ] Git installed
- [ ] Python 3.11+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed from `requirements.txt`
- [ ] AWS CLI configured (or IAM role attached)

## Initial Testing

### Test S3 Access
```bash
# Verify IAM role
aws sts get-caller-identity

# Test read access
aws s3 ls s3://your-input-bucket/physionet.org/files/

# Test write access
echo "test" > test.txt
aws s3 cp test.txt s3://your-output-bucket/test/
aws s3 rm s3://your-output-bucket/test/test.txt
```

### Test Phase 1 (Small Dataset)
```bash
python src/phase1_stay_identification.py \
  --base-path physionet.org/files \
  --s3-bucket your-input-bucket \
  --num-subjects 10 \
  --output-path test/phase1_test \
  --output-s3-bucket your-output-bucket
```

Expected runtime: 5-10 minutes

### Verify Output
```bash
# Check output files exist
aws s3 ls s3://your-output-bucket/test/phase1_test/

# Download and inspect metadata
aws s3 cp s3://your-output-bucket/test/phase1_test/metadata.json .
cat metadata.json
```

Expected output files:
- `linked_data.parquet`
- `train_data.parquet`
- `validate_data.parquet`
- `test_data.parquet`
- `metadata.json`

## Production Deployment

### Pre-Flight Checks
- [ ] Test run completed successfully
- [ ] Memory usage monitored and acceptable
- [ ] Network throughput adequate
- [ ] Cost estimates reviewed

### Production Run
```bash
# Use screen to keep session alive
screen -S phase1

# Run with full dataset
python src/phase1_stay_identification.py \
  --base-path physionet.org/files \
  --s3-bucket your-input-bucket \
  --num-subjects 1000 \
  --output-path production/phase1_$(date +%Y%m%d) \
  --output-s3-bucket your-output-bucket \
  --time-window 24

# Detach: Ctrl+A, then D
# Reattach: screen -r phase1
```

### Monitoring
- [ ] Monitor system resources: `htop`
- [ ] Monitor memory: `watch -n 1 free -h`
- [ ] Monitor disk: `watch -n 10 df -h`
- [ ] Monitor CloudWatch (if configured)
- [ ] Check logs periodically

### Expected Performance
- **10 subjects**: ~5-10 minutes
- **100 subjects**: ~30-60 minutes
- **1000 subjects**: ~5-8 hours
- **Full dataset**: ~days (varies by dataset size)

## Post-Deployment

### Validation
- [ ] All output files present in S3
- [ ] Metadata JSON contains expected counts
- [ ] No errors in logs
- [ ] Data quality spot checks passed

### Cost Review
- [ ] Review EC2 costs
- [ ] Review S3 storage costs
- [ ] Review data transfer costs
- [ ] Consider stopping/terminating instance if not needed

### Cleanup
```bash
# Stop instance (preserves data)
aws ec2 stop-instances --instance-ids i-xxxxxxxxx

# Or terminate (destroys data - ensure S3 backup!)
aws ec2 terminate-instances --instance-ids i-xxxxxxxxx
```

## Troubleshooting

### Common Issues

**Memory errors:**
- Increase instance size to r5.4xlarge (128GB)
- Reduce chunk_size in code
- Process fewer subjects at a time

**S3 access denied:**
- Verify IAM role permissions
- Check bucket policies
- Ensure instance has IAM role attached

**Slow processing:**
- Check network throughput
- Consider VPC endpoint for S3
- Verify instance type meets requirements

**Python dependencies:**
```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check specific package
pip show boto3
```

## Next Steps

After successful Phase 1 deployment:
1. ✅ Validate output data quality
2. ⏳ Deploy Phase 2 (Clinical Extraction)
3. ⏳ Deploy Phase 3 (Integration)
4. ⏳ Set up automated pipeline (optional)

## Emergency Contacts

- AWS Support: (if you have support plan)
- Repository Issues: https://github.com/yourusername/MIMIC-IV-CXR-ED-Anomaly-Detection/issues
- MIMIC Data: https://physionet.org/

## Rollback Plan

If deployment fails:
1. Check logs for error messages
2. Verify all prerequisites met
3. Test with minimal dataset (--num-subjects 1)
4. Revert to local testing if needed
5. Document issue and adjust deployment guide

---

**Last Updated:** $(date +%Y-%m-%d)
**Version:** 1.0.0
**Status:** Ready for Production ✅
