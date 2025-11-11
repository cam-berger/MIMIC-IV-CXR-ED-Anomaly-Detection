# OOM Failure Solution: Google Cloud Dataflow

## Problem

Phase 1 preprocessing fails with Out-Of-Memory (OOM) errors during the last stage (split creation), even after reducing max_workers to 1:

```
OOM killer: Out of memory: Killed process
total-vm: 20870852kB, anon-rss: 14862576kB (14.86GB RSS)
```

**Root cause:** The split creation stage must load ~1GB batch files sequentially and route records to train/val/test splits. Even with aggressive memory management (max_workers=1, chunking, garbage collection), single-machine processing hits memory limits with 100+ batch files.

## Solution: Distributed Processing with Dataflow

**Google Cloud Dataflow** provides a distributed, fault-tolerant solution that:
- Processes batch files in parallel across multiple workers
- Eliminates single-machine memory constraints
- Automatically scales based on workload
- Provides fault tolerance with automatic retries

## Quick Start

### 1. Enable Dataflow API

```bash
gcloud services enable dataflow.googleapis.com
```

### 2. Install Dependencies

```bash
pip install -r requirements_dataflow.txt
```

### 3. Configure

```bash
cp dataflow_config.json.example dataflow_config.json
nano dataflow_config.json  # Edit with your settings
```

Minimal configuration:
```json
{
  "project_id": "your-project-123",
  "gcs_bucket": "your-bucket-name",
  "batch_files_prefix": "preprocessed/batch_",
  "output_prefix": "preprocessed/"
}
```

### 4. Run

```bash
./run_dataflow_split.sh
```

Or manually:

```bash
# Step 1: Compute stratification indices (first time only)
python src/phase1_dataflow_split.py \
    --project_id your-project-123 \
    --gcs_bucket your-bucket-name \
    --batch_files_prefix preprocessed/batch_ \
    --output_prefix preprocessed/ \
    --runner DataflowRunner \
    --region us-central1 \
    --compute_indices_only

# Step 2: Run full pipeline
python src/phase1_dataflow_split.py \
    --project_id your-project-123 \
    --gcs_bucket your-bucket-name \
    --batch_files_prefix preprocessed/batch_ \
    --output_prefix preprocessed/ \
    --runner DataflowRunner \
    --region us-central1 \
    --max_num_workers 10
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dataflow Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. List Batch Files (GCS)                                 │
│     └─> gs://bucket/batch_0.pt, batch_1.pt, ...           │
│                                                             │
│  2. Load & Flatten (Parallel across workers)               │
│     Worker 1: batch_0.pt → [record_0, record_1, ...]      │
│     Worker 2: batch_1.pt → [record_50, record_51, ...]    │
│     Worker 3: batch_2.pt → [record_100, record_101, ...]  │
│                                                             │
│  3. Assign Splits (Based on stratification indices)        │
│     record_0 → train                                       │
│     record_1 → val                                         │
│     record_2 → test                                        │
│                                                             │
│  4. Group & Write Chunks                                   │
│     train: [records] → train_chunk_123.pt (50 records)    │
│     val: [records] → val_chunk_456.pt (50 records)        │
│     test: [records] → test_chunk_789.pt (50 records)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Memory Efficiency

| Approach | Memory Usage | Processing Time | Cost |
|----------|-------------|-----------------|------|
| **Original (max_workers=4)** | 14GB+ (fails) | N/A | $0 |
| **Fixed (max_workers=1)** | 2GB (still fails for you) | 3+ hours | $0 |
| **Dataflow (10 workers)** | ~2GB per worker (isolated) | 30 min | ~$1-2 |

Each Dataflow worker:
- Processes 1-2 batches at a time (~2GB)
- Writes output chunks immediately
- Releases memory after each batch
- Never holds entire dataset in memory

### Stratification

The pipeline maintains the same 70/15/15 train/val/test stratified split as the original phase1_preprocess.py:

1. **Pre-compute indices** (one-time):
   - Scan all batch files
   - Extract stratification keys (view position, clinical data availability)
   - Create stratified splits with reproducible seed (42)
   - Save to `stratification_indices.json`

2. **Apply during pipeline**:
   - Each record gets a global index
   - Assign to split based on pre-computed indices
   - Maintains exact same distribution as single-machine version

## Output Format

After completion, your GCS bucket will contain:

```
gs://your-bucket/preprocessed/
├── batch_0.pt                         # Original intermediate batches
├── batch_1.pt
├── ...
├── stratification_indices.json         # Split indices (reusable)
├── train_chunk_1678901234_1001.pt     # Train split chunks
├── train_chunk_1678901235_2002.pt
├── ...
├── val_chunk_1678901234_3003.pt       # Val split chunks
├── ...
└── test_chunk_1678901234_4004.pt      # Test split chunks
    ...
```

Each chunk file:
- Contains 50 records (configurable)
- Approximately 250-500MB (depends on record size)
- Can be loaded directly with `torch.load()`

## Comparison with Original Approach

| Aspect | Original (phase1_preprocess.py) | Dataflow Solution |
|--------|--------------------------------|-------------------|
| **Memory** | 2-14GB on single machine | 2GB per worker (distributed) |
| **Scalability** | Limited by single machine | Auto-scales to 100s of workers |
| **Fault Tolerance** | Fails and must restart | Automatic retries, work redistribution |
| **Processing Time** | 3+ hours sequential | 30 min with 10 workers |
| **Cost** | $0 (local/VM) | $1-2 per run |
| **Setup Complexity** | Simple (Python script) | Moderate (GCP setup) |
| **Use Case** | Small datasets (<50 batches) | Large datasets (100+ batches) |

## Cost Estimation

Example: 100 batch files (~100GB total data)

### Dataflow Job Cost
```
Workers: 10 × n1-standard-4 ($0.19/hr each)
Runtime: ~30 minutes
Dataflow overhead: ~20%

Total compute: 10 × $0.19 × 0.5 hr × 1.2 = $1.14
Dataflow service fee: ~$0.30
Total per run: ~$1.50
```

### Storage Cost
```
GCS storage: $0.02/GB/month × 100GB = $2.00/month
GCS operations: ~$0.05 (one-time for batch reads/writes)
```

### Total First Month
```
One-time job: $1.50
Storage: $2.00
Total: ~$3.50
```

Compare to:
- **VM with enough RAM** (n1-highmem-32: 208GB RAM): $1.70/hr = $1,224/month if always on
- **Occasional runs**: Dataflow is much more cost-effective

## Monitoring

### Cloud Console

1. Go to [Dataflow Console](https://console.cloud.google.com/dataflow)
2. View your job:
   - Job graph (visual pipeline)
   - Worker metrics
   - Throughput graphs
   - Error logs

### Command Line

```bash
# List running jobs
gcloud dataflow jobs list --region=us-central1 --status=active

# View job details
gcloud dataflow jobs describe JOB_ID --region=us-central1

# Stream logs
gcloud logging read "resource.type=dataflow_step" --limit 50
```

### Check Output

```bash
# Count output chunks
gsutil ls gs://your-bucket/preprocessed/train_chunk_*.pt | wc -l
gsutil ls gs://your-bucket/preprocessed/val_chunk_*.pt | wc -l
gsutil ls gs://your-bucket/preprocessed/test_chunk_*.pt | wc -l

# Verify chunk sizes
gsutil du -sh gs://your-bucket/preprocessed/train_chunk_*.pt | head -5

# Download and inspect
gsutil cp gs://your-bucket/preprocessed/train_chunk_12345_6789.pt .
python -c "import torch; data=torch.load('train_chunk_12345_6789.pt'); print(f'{len(data)} records')"
```

## Troubleshooting

### Common Issues

**Problem:** "Dataflow API not enabled"
```bash
gcloud services enable dataflow.googleapis.com
```

**Problem:** "Permission denied"
```bash
# Grant your account Dataflow Admin role
gcloud projects add-iam-policy-binding YOUR_PROJECT \
    --member="user:your.email@example.com" \
    --role="roles/dataflow.admin"
```

**Problem:** Workers failing with OOM
```bash
# Use high-memory machines
python src/phase1_dataflow_split.py \
    --machine_type n1-highmem-8 \
    --disk_size_gb 200 \
    ...
```

**Problem:** Pipeline slow or stuck
```bash
# Increase workers
python src/phase1_dataflow_split.py \
    --max_num_workers 20 \
    ...
```

**Problem:** "No batch files found"
```bash
# Verify prefix
gsutil ls gs://your-bucket/preprocessed/batch_*.pt | head -5

# Ensure prefix is correct (should end with "batch_" not just directory)
--batch_files_prefix preprocessed/batch_  # Correct
--batch_files_prefix preprocessed/        # Wrong
```

## Advanced Usage

### Custom Split Ratios

```bash
python src/phase1_dataflow_split.py \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    ...
```

### Test with Subset

```bash
# Copy first 10 batches to test location
gsutil -m cp gs://bucket/preprocessed/batch_{0..9}.pt gs://bucket/test/

# Run on subset
python src/phase1_dataflow_split.py \
    --batch_files_prefix test/batch_ \
    --output_prefix test_output/ \
    --runner DirectRunner  # Run locally for testing
```

### Use Different Region

```bash
# Choose region closest to your bucket
python src/phase1_dataflow_split.py \
    --region us-east1 \
    ...
```

### Larger Chunks (More Memory per Worker)

```bash
python src/phase1_dataflow_split.py \
    --chunk_size 100 \
    --machine_type n1-highmem-8 \
    ...
```

## Integration with Phase 2

After Dataflow creates chunked splits, load them in your training pipeline:

```python
# In your Phase 2 training script
from pathlib import Path
import torch
from google.cloud import storage
import tempfile

class ChunkedDataset(torch.utils.data.Dataset):
    """Dataset that loads from multiple chunk files"""

    def __init__(self, bucket_name, prefix, split_name):
        self.bucket_name = bucket_name
        self.split_name = split_name

        # List all chunks
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        prefix_path = f"{prefix}{split_name}_chunk_"
        blobs = list(bucket.list_blobs(prefix=prefix_path))

        # Load all chunks
        self.records = []
        for blob in blobs:
            with tempfile.NamedTemporaryFile(suffix='.pt') as tmp:
                blob.download_to_filename(tmp.name)
                chunk_records = torch.load(tmp.name)
                self.records.extend(chunk_records)

        print(f"Loaded {len(self.records)} records from {len(blobs)} chunks")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

# Use in your training
train_dataset = ChunkedDataset(
    bucket_name='your-bucket',
    prefix='preprocessed/',
    split_name='train'
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

## When NOT to Use Dataflow

Stick with the original `phase1_preprocess.py` if:
- You have <50 batch files (<50GB data)
- You have a VM with 32GB+ RAM
- You're not experiencing OOM failures
- You want to avoid GCP setup and costs
- You're doing development/testing with small datasets

Use Dataflow if:
- You're experiencing OOM failures
- You have 100+ batch files (100GB+)
- You want faster processing (distributed)
- You need fault tolerance for long-running jobs
- You're comfortable with GCP and small costs ($1-2/run)

## Next Steps

1. **Complete setup**: Follow [DATAFLOW_SETUP.md](DATAFLOW_SETUP.md) for detailed setup instructions
2. **Run pipeline**: Use `./run_dataflow_split.sh` for guided execution
3. **Verify output**: Check that chunks are created correctly
4. **Integrate with Phase 2**: Update training pipeline to load chunked data
5. **Optimize**: Adjust workers, machine types, chunk sizes based on your data

## Files Created

```
src/
└── phase1_dataflow_split.py        # Main Dataflow pipeline script

docs/
├── DATAFLOW_SETUP.md                # Detailed setup guide
└── OOM_SOLUTION.md                  # This file

Root directory/
├── dataflow_config.json.example     # Configuration template
├── run_dataflow_split.sh            # Quick-start script
├── setup_dataflow.py                # Setup file for workers
└── requirements_dataflow.txt        # Dataflow dependencies
```

## References

- [Apache Beam Documentation](https://beam.apache.org/documentation/)
- [Google Cloud Dataflow](https://cloud.google.com/dataflow/docs)
- [Dataflow Best Practices](https://cloud.google.com/dataflow/docs/guides/best-practices)
- [Dataflow Pricing](https://cloud.google.com/dataflow/pricing)

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review [DATAFLOW_SETUP.md](DATAFLOW_SETUP.md) FAQ
3. Check Dataflow job logs in Cloud Console
4. Open an issue in this repository with:
   - Error messages from logs
   - Your configuration (redact sensitive info)
   - GCS bucket structure (`gsutil ls` output)
   - Dataflow job ID
