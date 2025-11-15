# Official MIMIC-CXR Splits Implementation

## Overview

This document describes the implementation of official MIMIC-CXR train/validation/test splits in the preprocessing pipeline.

## Critical Issue Fixed

**Previous Behavior (INCORRECT):**
- The preprocessing code created **custom stratified splits** with 70/15/15 ratios
- Used random splitting with `np.random.seed(42)`
- **Risk of data leakage**: Different images from the same patient could appear in train and test sets

**New Behavior (CORRECT):**
- Uses the **official MIMIC-CXR patient-level splits** from `mimic-cxr-2.0.0-split.csv.gz`
- All images from the same patient are in the same split (train, validate, or test)
- **Ensures reproducibility** with published MIMIC-CXR research
- **Prevents data leakage** through patient-level splitting

## Why This Matters

### 1. **Reproducibility**
Published papers using MIMIC-CXR use the official splits. Custom splits make it impossible to compare results fairly.

### 2. **Data Leakage Prevention**
The official splits ensure patient-level separation:
- Patient `10000032` might have 5 chest X-rays
- **Correct**: All 5 X-rays in the SAME split (e.g., all in training)
- **Incorrect**: Some X-rays in training, some in testing → model has "seen" this patient before

### 3. **Benchmarking**
Fair comparison with other methods requires using the same data splits.

## Official Split Statistics

From MIMIC-CXR v2.0.0 documentation:

| Split      | Studies   | Percentage |
|------------|-----------|------------|
| Train      | ~227,827  | ~74%       |
| Validate   | ~1,808    | ~0.6%      |
| Test       | ~3,269    | ~1.1%      |

Note: The validation set is relatively small. Some researchers combine validate+test for final evaluation.

## Configuration

### Option 1: Use Official Splits (RECOMMENDED)

```python
from src.phase1_preprocess_streaming import DataConfig

config = DataConfig(
    mimic_cxr_path="/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-cxr-jpg/2.1.0",
    use_official_splits=True,  # ← Enabled by default
    official_split_file="mimic-cxr-2.0.0-split.csv.gz"
)
```

### Option 2: Custom Splits (NOT RECOMMENDED)

Only use this for experimental purposes where you explicitly need different splits:

```python
config = DataConfig(
    mimic_cxr_path="/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-cxr-jpg/2.1.0",
    use_official_splits=False,  # Disable official splits
    train_split=0.7,
    val_split=0.15,
    test_split=0.15
)
```

**Warning:** If you disable official splits, your code will log prominent warnings about reproducibility concerns.

## Implementation Details

### How It Works

1. **Load Official Splits** (at initialization):
   ```python
   # Loads mimic-cxr-2.0.0-split.csv.gz
   # Creates mapping: (subject_id, study_id) -> 'train'/'validate'/'test'
   ```

2. **Map Records to Splits** (during preprocessing):
   ```python
   # For each processed record:
   #   - Extract subject_id and study_id
   #   - Look up official split assignment
   #   - Assign record to corresponding split
   ```

3. **Handle Edge Cases**:
   - Records not in official splits → **excluded** (logged as warnings)
   - Records missing IDs → **excluded** (logged as warnings)
   - Split normalization: 'validate' → 'val' for consistency

### Files Modified

1. **`src/phase1_preprocess_streaming.py`**:
   - Added `use_official_splits` and `official_split_file` to `DataConfig`
   - Added `load_official_splits()` method to `DatasetCreator`
   - Added `create_official_split_map_streaming()` method
   - Modified `create_splits_from_batches_streaming()` to use official splits

## Verification

### Check Your Preprocessing Output

After running preprocessing, verify the split distribution:

```python
import torch

# Load your preprocessed data
train_data = torch.load('train_00000.pt')
val_data = torch.load('val_00000.pt')
test_data = torch.load('test_00000.pt')

print(f"Train samples: {len(train_data)}")
print(f"Val samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")

# Check for patient leakage
train_patients = set(r['subject_id'] for r in train_data)
val_patients = set(r['subject_id'] for r in val_data)
test_patients = set(r['subject_id'] for r in test_data)

# Should be ZERO
train_val_overlap = train_patients & val_patients
train_test_overlap = train_patients & test_patients
val_test_overlap = val_patients & test_patients

assert len(train_val_overlap) == 0, "Patient leakage: train/val overlap!"
assert len(train_test_overlap) == 0, "Patient leakage: train/test overlap!"
assert len(val_test_overlap) == 0, "Patient leakage: val/test overlap!"

print("✓ No patient leakage detected!")
```

## Running Preprocessing with Official Splits

### Command Line

```bash
python src/phase1_preprocess_streaming.py \
    --mimic-cxr-path /home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-cxr-jpg/2.1.0 \
    --mimic-iv-path /home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimiciv/3.1 \
    --output-path /home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/preprocessed \
    --skip-to-combine
```

The official splits will be used automatically (default behavior).

### Expected Console Output

```
============================================================
Loading official MIMIC-CXR splits
============================================================
Loading split file from: /home/dev/.../mimic-cxr-2.0.0-split.csv.gz
Loaded 377,110 study assignments

Split distribution:
  train :  227,827 studies ( 60.4%)
  val   :    1,808 studies (  0.5%)
  test  :    3,269 studies (  0.9%)
============================================================
✓ Official splits loaded successfully
✓ Patient-level splitting ensures no data leakage
============================================================
```

## Troubleshooting

### Error: "Split file not found"

**Solution:** Ensure `mimic-cxr-2.0.0-split.csv.gz` exists:

```bash
ls -lh /home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz
```

If missing, re-download from PhysioNet: https://physionet.org/content/mimic-cxr-jpg/2.1.0/

### Warning: "Records not in official split file"

This is **normal** if you're using additional datasets (e.g., MIMIC-IV-ED) that contain studies not in MIMIC-CXR-JPG v2.0.0.

Records without official split assignments will be **excluded** from training/validation/testing.

### Fallback to Custom Splits

If the official split file cannot be loaded, the code will:
1. Log error messages explaining what went wrong
2. Display prominent warnings about using custom splits
3. Fall back to the old stratified splitting behavior

**Recommendation:** Fix the split file path rather than relying on the fallback.

## Migration Guide

### If You Have Existing Preprocessed Data

**You MUST re-preprocess** if you previously used custom splits:

```bash
# 1. Backup old data
mv /path/to/preprocessed /path/to/preprocessed_old_custom_splits

# 2. Re-run preprocessing with official splits
python src/phase1_preprocess_streaming.py --skip-to-combine

# 3. Verify new splits are patient-level
python scripts/verify_data_loading.py
```

### Updating Your Training Configs

No changes needed! The training code uses the split assignments from preprocessed files.

## References

1. **MIMIC-CXR-JPG Documentation**:
   https://physionet.org/content/mimic-cxr-jpg/2.1.0/

2. **Official Split File**:
   `mimic-cxr-2.0.0-split.csv.gz` (included in MIMIC-CXR-JPG download)

3. **Published Papers Using MIMIC-CXR**:
   - Johnson et al. "MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs." (2019)
   - Most papers using MIMIC-CXR reference the official splits for reproducibility

## Summary

✅ **Official splits enabled by default** (`use_official_splits=True`)
✅ **Patient-level splitting prevents data leakage**
✅ **Reproducible with published research**
✅ **Automatic fallback with warnings if split file unavailable**
✅ **Backward compatible** (can disable if needed for experiments)

**Action Required:**
- If you have existing preprocessed data using custom splits → **re-preprocess**
- If starting fresh → **no action needed** (official splits used automatically)
