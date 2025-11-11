# Google Cloud Dataflow Setup Guide

## Overview

This guide helps you set up and run the Phase 1 split creation using Google Cloud Dataflow to avoid OOM failures.

## When to Use Dataflow

Use Dataflow when:
- Phase 1 preprocessing fails with OOM during split creation
- You have 100+ batch files (>100GB of intermediate data)
- Single-machine processing is too slow or unstable
- You need distributed, fault-tolerant processing

## Prerequisites

1. **Google Cloud Project** with Dataflow API enabled
2. **GCS Bucket** with your Phase 1 batch files
3. **Local Python environment** (3.8+)
4. **Google Cloud SDK** installed

## Step 1: Enable Required APIs

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable dataflow.googleapis.com
gcloud services enable compute.googleapis.com
gcloud services enable storage-api.googleapis.com
gcloud services enable logging.googleapis.com
gcloud services enable monitoring.googleapis.com
```

## Step 2: Set Up Service Account (Optional but Recommended)

```bash
# Create service account
gcloud iam service-accounts create dataflow-worker \
    --display-name="Dataflow Worker Service Account"

# Grant necessary roles
PROJECT_ID=$(gcloud config get-value project)
SERVICE_ACCOUNT="dataflow-worker@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/dataflow.worker"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/storage.objectAdmin"

# Create and download key
gcloud iam service-accounts keys create dataflow-key.json \
    --iam-account=${SERVICE_ACCOUNT}

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/dataflow-key.json"
```

## Step 3: Install Dataflow Dependencies

```bash
# Install Dataflow requirements
pip install -r requirements_dataflow.txt

# Verify installation
python -c "import apache_beam; print(f'Apache Beam version: {apache_beam.__version__}')"
```

## Step 4: Verify Your GCS Bucket Structure

Your GCS bucket should contain the intermediate batch files from Phase 1:

```
gs://YOUR_BUCKET/
├── preprocessed/
│   ├── batch_0.pt
│   ├── batch_1.pt
│   ├── batch_2.pt
│   └── ... (100+ batch files)
└── ...
```

Check your files:

```bash
# List batch files
gsutil ls gs://YOUR_BUCKET/preprocessed/batch_*.pt | head -10

# Count batch files
gsutil ls gs://YOUR_BUCKET/preprocessed/batch_*.pt | wc -l

# Check total size
gsutil du -sh gs://YOUR_BUCKET/preprocessed/
```

## Step 5: Run Dataflow Pipeline

### Option A: First-Time Setup (Compute Stratification Indices)

If you haven't created stratification indices yet, run this first:

```bash
python src/phase1_dataflow_split.py \
    --project_id YOUR_PROJECT_ID \
    --gcs_bucket YOUR_BUCKET_NAME \
    --batch_files_prefix preprocessed/batch_ \
    --output_prefix preprocessed/ \
    --region us-central1 \
    --runner DataflowRunner \
    --max_num_workers 10 \
    --machine_type n1-standard-4 \
    --compute_indices_only
```

This will:
1. Scan all batch files
2. Extract stratification keys (view position, clinical data availability)
3. Create train/val/test indices (70/15/15 split)
4. Save indices to `gs://YOUR_BUCKET/preprocessed/stratification_indices.json`

### Option B: Run Full Pipeline

Once indices are computed (or if they already exist), run the full pipeline:

```bash
python src/phase1_dataflow_split.py \
    --project_id YOUR_PROJECT_ID \
    --gcs_bucket YOUR_BUCKET_NAME \
    --batch_files_prefix preprocessed/batch_ \
    --output_prefix preprocessed/ \
    --region us-central1 \
    --runner DataflowRunner \
    --max_num_workers 10 \
    --machine_type n1-standard-4 \
    --chunk_size 50
```

### Option C: Test Locally First (DirectRunner)

Before running on Dataflow, test with a small subset using DirectRunner:

```bash
# Copy a few batch files to a test prefix
gsutil -m cp gs://YOUR_BUCKET/preprocessed/batch_[0-5].pt \
    gs://YOUR_BUCKET/test_preprocessed/

# Run locally
python src/phase1_dataflow_split.py \
    --project_id YOUR_PROJECT_ID \
    --gcs_bucket YOUR_BUCKET_NAME \
    --batch_files_prefix test_preprocessed/batch_ \
    --output_prefix test_preprocessed/ \
    --region us-central1 \
    --runner DirectRunner \
    --chunk_size 10
```

## Step 6: Monitor Pipeline

### View in Cloud Console

1. Go to [Dataflow Console](https://console.cloud.google.com/dataflow)
2. Find your job (named like `phase1dataflowsplit-...`)
3. Monitor:
   - Job graph (visualize pipeline)
   - Worker logs
   - Throughput metrics
   - Error messages

### View Logs

```bash
# Stream logs
gcloud logging read "resource.type=dataflow_step" --limit 50 --format json

# Filter by job ID
gcloud logging read "resource.labels.job_id=YOUR_JOB_ID" --limit 50
```

### Check Output Files

```bash
# List output chunks
gsutil ls gs://YOUR_BUCKET/preprocessed/*_chunk_*.pt

# Count chunks by split
echo "Train chunks:" $(gsutil ls gs://YOUR_BUCKET/preprocessed/train_chunk_*.pt | wc -l)
echo "Val chunks:" $(gsutil ls gs://YOUR_BUCKET/preprocessed/val_chunk_*.pt | wc -l)
echo "Test chunks:" $(gsutil ls gs://YOUR_BUCKET/preprocessed/test_chunk_*.pt | wc -l)

# Check chunk sizes
gsutil du -sh gs://YOUR_BUCKET/preprocessed/train_chunk_*.pt | head -5
```

## Step 7: Verify Output

After the pipeline completes, verify the output:

```python
# Download and inspect a sample chunk
import torch
from google.cloud import storage

client = storage.Client(project='YOUR_PROJECT_ID')
bucket = client.bucket('YOUR_BUCKET_NAME')

# Download one train chunk
blob = bucket.blob('preprocessed/train_chunk_1234567890_1234.pt')
with open('sample_chunk.pt', 'wb') as f:
    blob.download_to_file(f)

# Load and inspect
records = torch.load('sample_chunk.pt')
print(f"Chunk contains {len(records)} records")
print(f"First record keys: {records[0].keys()}")
print(f"Image shape: {records[0]['image'].shape}")
```

## Configuration Options

### Pipeline Arguments

| Argument | Description | Default | Example |
|----------|-------------|---------|---------|
| `--project_id` | GCP project ID | **Required** | `my-project-123` |
| `--gcs_bucket` | GCS bucket name | **Required** | `bergermimiciv` |
| `--batch_files_prefix` | Prefix for batch files | **Required** | `preprocessed/batch_` |
| `--output_prefix` | Output prefix | **Required** | `preprocessed/` |
| `--region` | GCP region | `us-central1` | `us-east1` |
| `--runner` | Beam runner | `DataflowRunner` | `DirectRunner` |
| `--max_num_workers` | Max workers | `10` | `20` |
| `--machine_type` | Worker machine | `n1-standard-4` | `n1-highmem-8` |
| `--disk_size_gb` | Worker disk (GB) | `100` | `200` |
| `--chunk_size` | Records/chunk | `50` | `100` |
| `--train_ratio` | Train split ratio | `0.7` | `0.8` |
| `--val_ratio` | Val split ratio | `0.15` | `0.1` |
| `--test_ratio` | Test split ratio | `0.15` | `0.1` |
| `--random_seed` | Random seed | `42` | `123` |

### Machine Type Selection

Choose based on your data size and budget:

| Machine Type | vCPUs | RAM | Best For | Cost/hr* |
|--------------|-------|-----|----------|----------|
| `n1-standard-1` | 1 | 3.75 GB | Testing | $0.048 |
| `n1-standard-2` | 2 | 7.5 GB | Small batches | $0.095 |
| `n1-standard-4` | 4 | 15 GB | **Recommended** | $0.190 |
| `n1-standard-8` | 8 | 30 GB | Large batches | $0.380 |
| `n1-highmem-4` | 4 | 26 GB | Memory-intensive | $0.237 |
| `n1-highmem-8` | 8 | 52 GB | Very large batches | $0.474 |

*Prices are approximate for us-central1 region as of 2024.

### Cost Estimation

Example: Processing 100 batch files (~100GB total)

```
Workers: 10 × n1-standard-4
Runtime: ~30 minutes (estimated)
Cost: 10 workers × $0.19/hr × 0.5 hr = $0.95
Storage: $0.02/GB/month × 100GB = $2/month
Total: ~$1-2 for the job + $2/month storage
```

## Troubleshooting

### Issue: "Dataflow API not enabled"

```bash
gcloud services enable dataflow.googleapis.com
```

### Issue: "Permission denied" errors

Check service account permissions:

```bash
gcloud projects get-iam-policy YOUR_PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:*dataflow*"
```

Grant required roles:

```bash
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
    --role="roles/dataflow.worker"
```

### Issue: "No module named 'apache_beam'"

```bash
pip install --upgrade apache-beam[gcp]
```

### Issue: Workers failing with OOM

Increase machine memory:

```bash
--machine_type n1-highmem-8 \
--disk_size_gb 200
```

Or reduce chunk size:

```bash
--chunk_size 25
```

### Issue: Pipeline stuck or slow

1. Check worker logs in Cloud Console
2. Increase number of workers:
   ```bash
   --max_num_workers 20
   ```
3. Check for stragglers (unbalanced work distribution)

### Issue: "Batch files not found"

Verify prefix:

```bash
gsutil ls gs://YOUR_BUCKET/YOUR_PREFIX/batch_*.pt | head -5
```

Ensure prefix ends correctly:
- Correct: `preprocessed/batch_`
- Wrong: `preprocessed/` (missing `batch_` prefix)

### Issue: Incorrect split sizes

The stratification indices might be computed from different batch order. Solutions:

1. **Recompute indices** with `--compute_indices_only`
2. **Verify batch file ordering** (should be sorted alphabetically)
3. **Check batch file consistency** (ensure no batches were added/removed)

## Advanced Usage

### Custom Split Ratios

Change train/val/test split:

```bash
python src/phase1_dataflow_split.py \
    ... \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

### Process Specific Batches Only

Create a subset for testing:

```bash
# Copy specific batches
gsutil -m cp gs://YOUR_BUCKET/preprocessed/batch_{0..9}.pt \
    gs://YOUR_BUCKET/debug_preprocessed/

# Process subset
python src/phase1_dataflow_split.py \
    --batch_files_prefix debug_preprocessed/batch_ \
    --output_prefix debug_output/ \
    ...
```

### Use with Phase 2

After Dataflow completes, use the chunked output in Phase 2:

```python
# In your Phase 2 training script
from pathlib import Path
import torch

def load_split_chunks(split_name, gcs_prefix):
    """Load all chunks for a split"""
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket('YOUR_BUCKET')

    prefix = f"{gcs_prefix}{split_name}_chunk_"
    blobs = list(bucket.list_blobs(prefix=prefix))

    all_records = []
    for blob in blobs:
        # Download and load each chunk
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
            blob.download_to_filename(tmp.name)
            records = torch.load(tmp.name)
            all_records.extend(records)

    return all_records

# Use in dataloader
train_data = load_split_chunks('train', 'preprocessed/')
```

## Cleanup

After successful completion, you can clean up:

```bash
# Keep only final chunks, delete intermediate batches (optional)
# BE CAREFUL: Only do this after verifying splits are correct!
gsutil -m rm gs://YOUR_BUCKET/preprocessed/batch_*.pt

# Delete staging/temp directories
gsutil -m rm -r gs://YOUR_BUCKET/dataflow_staging/
gsutil -m rm -r gs://YOUR_BUCKET/dataflow_temp/
```

## FAQ

**Q: How long will the pipeline take?**
A: Depends on data size and workers. Example: 100 batch files (~100GB) with 10 workers typically takes 20-40 minutes.

**Q: Can I resume a failed pipeline?**
A: Dataflow doesn't support resume, but it's fault-tolerant. If a worker fails, work is redistributed. If the entire job fails, re-run it (it will skip already-written chunks if using unique timestamps).

**Q: How much will this cost?**
A: See cost estimation section. Typically $1-5 for a single run.

**Q: Can I run this locally?**
A: Yes, use `--runner DirectRunner`, but it won't be distributed. Better for testing small subsets.

**Q: What if my batch files are in different buckets?**
A: The current implementation assumes a single bucket. You would need to modify `ListBatchFilesDoFn` to handle multiple buckets.

**Q: Do I need to recompute indices every time?**
A: No. Indices are saved to `stratification_indices.json` and reused automatically. Only recompute if you change batch files or split ratios.

## Next Steps

After successful Dataflow completion:

1. Verify output chunks are correct
2. Update Phase 2 to load from chunked format
3. Consider automating this with Cloud Composer or Cloud Scheduler for regular runs
4. Set up monitoring and alerting for production pipelines

## Support

For issues:
1. Check [Apache Beam documentation](https://beam.apache.org/documentation/)
2. Review [Dataflow troubleshooting guide](https://cloud.google.com/dataflow/docs/guides/troubleshooting-your-pipeline)
3. Open an issue in the repository

## References

- [Apache Beam Python SDK](https://beam.apache.org/documentation/sdks/python/)
- [Google Cloud Dataflow](https://cloud.google.com/dataflow/docs)
- [Dataflow Pricing](https://cloud.google.com/dataflow/pricing)
- [Best Practices for Dataflow](https://cloud.google.com/dataflow/docs/guides/best-practices)
