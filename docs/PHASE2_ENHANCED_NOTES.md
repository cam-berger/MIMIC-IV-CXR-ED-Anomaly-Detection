# Phase 2: Enhanced Pseudo-Note Generation and RAG Integration

## Overview

Phase 2 integrates with Phase 1 outputs to generate narrative pseudo-notes from structured clinical data and enhance them with RAG (Retrieval-Augmented Generation) medical knowledge. This aligns with the research hypothesis: **context-aware knowledge augmentation of clinical notes improves chest X-ray abnormality detection**.

## What Phase 2 Does

1. **Reads Phase 1 Outputs**: Loads preprocessed data (train/val/test splits) as `.pt` files
2. **Generates Pseudo-Notes**: Converts structured clinical features into narrative text
   - Example: `{age: 65, HR: 85, O2sat: 92}` → `"Patient is a 65 year old M. Vital signs: heart rate 85 bpm, oxygen saturation 92%."`
3. **RAG Knowledge Enhancement**: Retrieves relevant medical knowledge and augments pseudo-notes
4. **Tokenizes for ModernBERT**: Prepares enhanced notes for Clinical ModernBERT encoder
5. **Saves Enhanced Data**: Outputs enriched records ready for model training

## Key Features

- **Unified with Phase 1**: Uses same `DataConfig` and `GCSHelper` from phase1_preprocess_streaming.py
- **GCS + Local Support**: Works with both Google Cloud Storage and local filesystems
- **Narrative Generation**: Converts structured vitals/demographics into clinical narrative
- **RAG Integration**: FAISS-based retrieval of relevant medical knowledge
- **Medical Knowledge Base**: 15+ medical knowledge documents covering common ED presentations
- **Abbreviation Expansion**: Expands common medical abbreviations (HTN → hypertension)
- **Preserves Original Data**: Keeps all Phase 1 data intact, adds enhanced fields

## Architecture

```
Phase 1 Output (train_data.pt)
    ↓
PseudoNoteGenerator
    ↓ (structured data → narrative)
Pseudo-Note: "Patient is a 65 year old M. Vital signs: HR 85bpm, O2sat 92%..."
    ↓
RAGEnhancer
    ↓ (retrieve medical knowledge)
Enhanced Note: "[CLINICAL PRESENTATION] ...pseudo-note... [MEDICAL CONTEXT] ...knowledge..."
    ↓
TextEnhancer (Tokenizer)
    ↓
Enhanced Record (train_data_enhanced.pt)
```

## Output Format

Phase 2 **adds** the following fields to each Phase 1 record:

```python
enhanced_record = {
    # Original Phase 1 fields (preserved)
    'subject_id': ...,
    'study_id': ...,
    'image': ...,
    'clinical_features': ...,

    # NEW Phase 2 fields
    'pseudo_note': str,                    # Narrative clinical note
    'enhanced_note': str,                  # RAG-enhanced note
    'enhanced_text_tokens': {              # Tokenized for ModernBERT
        'input_ids': torch.Tensor,
        'attention_mask': torch.Tensor
    },
    'phase2_processed': True
}
```

## Usage

### Basic Usage (Local)

```bash
# Process Phase 1 local outputs
python src/phase2_enhanced_notes.py \
  --input-path ~/MIMIC_Data/processed/phase1_output \
  --max-text-length 8192 \
  --top-k-retrieval 5
```

### GCS Usage

```bash
# Process Phase 1 GCS outputs
python src/phase2_enhanced_notes.py \
  --input-path processed/phase1_with_path_fixes_raw \
  --gcs-bucket bergermimiciv \
  --gcs-project-id YOUR_PROJECT_ID \
  --max-text-length 8192 \
  --top-k-retrieval 5
```

### Quick Testing with Small Samples

```bash
# Use small sample files (train_small.pt) for fast testing
python src/phase2_enhanced_notes.py \
  --input-path processed/phase1_output \
  --gcs-bucket bergermimiciv \
  --use-small-sample
```

### Custom RAG Settings

```bash
# Customize RAG retrieval
python src/phase2_enhanced_notes.py \
  --input-path processed/phase1_output \
  --top-k-retrieval 10 \
  --embedding-model sentence-transformers/all-mpnet-base-v2
```

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-path` | Yes | - | Path to Phase 1 output directory |
| `--output-path` | No | Same as input | Output path (if different from input) |
| `--gcs-bucket` | No | None | GCS bucket name (enables GCS mode) |
| `--gcs-project-id` | No | None | GCP project ID for requester pays |
| `--max-text-length` | No | 8192 | Max text length for ModernBERT |
| `--top-k-retrieval` | No | 5 | Number of knowledge docs to retrieve |
| `--embedding-model` | No | all-MiniLM-L6-v2 | Sentence transformer for RAG |
| `--use-small-sample` | No | False | Use small sample files for testing |

## Components

### 1. PseudoNoteGenerator

Converts structured clinical features into narrative text:

**Input**: Clinical features tensor + feature names
```python
clinical_features = torch.tensor([65.0, 1.0, 98.6, 85.0, 18.0, 92.0, ...])
feature_names = ['age', 'gender', 'temperature', 'heartrate', ...]
```

**Output**: Narrative pseudo-note
```
"Patient is a 65 year old M. Chief complaint: shortness of breath.
Vital signs: temperature 98.6°F, heart rate 85 bpm, respiratory rate 18 breaths/min,
oxygen saturation 92%. Triage acuity: urgent (Level 3)."
```

**Features**:
- Demographics (age, gender)
- Chief complaint with abbreviation expansion
- Vital signs formatting
- Triage acuity mapping
- Pain scores

### 2. RAGEnhancer

Retrieves and integrates medical knowledge:

**Medical Knowledge Base**:
- 15+ clinical knowledge documents
- Covers: pneumonia, CHF, COVID-19, COPD, pulmonary embolism, etc.
- FAISS index for efficient retrieval
- Sentence-transformer embeddings

**Retrieval Process**:
1. Encode pseudo-note → embedding vector
2. Search FAISS index for similar medical knowledge
3. Retrieve top-k most relevant documents
4. Combine: `[CLINICAL PRESENTATION] note [MEDICAL CONTEXT] knowledge`

### 3. TextEnhancer

Tokenizes enhanced notes for Clinical ModernBERT:

**Tokenization**:
- Max length: 8192 tokens (ModernBERT extended context)
- Padding: max_length
- Truncation: enabled
- Returns: input_ids + attention_mask

## Example Output

### Input (Phase 1 Clinical Features)
```python
{
    'age': 65.0,
    'gender': 1.0,  # Male
    'temperature': 98.6,
    'heartrate': 85.0,
    'resprate': 18.0,
    'o2sat': 92.0,
    'sbp': 140.0,
    'dbp': 85.0,
    'acuity': 3.0,
    'chiefcomplaint': 'sob'
}
```

### Output (Phase 2)

**Pseudo-Note**:
```
Patient is a 65 year old M. Chief complaint: shortness of breath.
Vital signs: temperature 98.6°F, heart rate 85 bpm, respiratory rate 18 breaths/min,
oxygen saturation 92%, blood pressure 140/85 mmHg. Triage acuity: urgent (Level 3).
```

**Enhanced Note** (with RAG knowledge):
```
[CLINICAL PRESENTATION] Patient is a 65 year old M. Chief complaint: shortness of breath...
[MEDICAL CONTEXT] Shortness of breath can result from cardiac, pulmonary, or metabolic causes.
Chest X-ray helps differentiate between these etiologies. Congestive heart failure manifests
as cardiomegaly, pulmonary edema, pleural effusions...
```

## Integration with Model Training

Phase 2 outputs are ready for Enhanced MDF-Net training:

```python
# Load Phase 2 enhanced data
train_data = torch.load('train_data_enhanced.pt')

for record in train_data:
    # Visual pathway
    image = record['image']  # From Phase 1: 518x518 tensor

    # Text pathway
    text_tokens = record['enhanced_text_tokens']  # From Phase 2: tokenized
    input_ids = text_tokens['input_ids']
    attention_mask = text_tokens['attention_mask']

    # Cross-modal fusion
    outputs = model(
        image=image,
        input_ids=input_ids,
        attention_mask=attention_mask
    )
```

## Performance

- **Speed**: Processes ~1000 records in 2-3 minutes (with RAG retrieval)
- **Memory**: ~4GB RAM for standard datasets
- **Storage**: Adds ~20% to Phase 1 file sizes (due to text data)

## Troubleshooting

### "FileNotFoundError: train_data.pt"
- Ensure Phase 1 has completed successfully
- Check `--input-path` points to Phase 1 output directory
- Verify files exist: `train_data.pt`, `val_data.pt`, `test_data.pt`

### "ImportError: phase1_preprocess_streaming"
- Ensure you're running from project root
- Add to PYTHONPATH: `export PYTHONPATH="${PYTHONPATH}:/path/to/src"`

### "GCS authentication error"
- Run: `gcloud auth application-default login`
- Set project: `gcloud config set project YOUR_PROJECT_ID`

### RAG retrieval is slow
- Reduce `--top-k-retrieval` (default: 5 → try 3)
- Use faster embedding model (all-MiniLM-L6-v2 is already optimized)

## Extending the Knowledge Base

To add more medical knowledge:

1. Edit `RAGEnhancer.build_knowledge_base()` in `phase2_enhanced_notes.py`
2. Add knowledge documents to `sample_knowledge` list:
```python
sample_knowledge = [
    "Your medical knowledge text here...",
    "More knowledge...",
    # ... existing knowledge
]
```
3. Re-run Phase 2

**Production**: Load knowledge from external database (PubMed, medical textbooks, etc.)

## Next Steps

After Phase 2:
1. **Phase 3**: Multi-modal fusion and final integration (if needed)
2. **Model Training**: Use enhanced records to train Enhanced MDF-Net
3. **Evaluation**: Compare model performance with vs without RAG enhancement

## References

- Phase 1: `src/phase1_preprocess_streaming.py`
- DataConfig: Shared configuration class
- GCSHelper: Unified GCS/local file handler
- README: Main project documentation

## Contact

For questions about Phase 2:
- Review code: `src/phase2_enhanced_notes.py`
- Check logs: Processing statistics and sample outputs
- Open issue: GitHub issues for bugs/feature requests
