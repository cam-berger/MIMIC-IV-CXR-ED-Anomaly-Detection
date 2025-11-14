# Configuration Files

This directory contains YAML configuration files for different training scenarios.

## Available Configurations

### 1. `base.yaml` - Stage 1: Feature Extraction
- **Use case**: Initial training with frozen encoders
- **Modalities**: Vision + Text + Clinical (full multi-modal)
- **Freeze encoders**: Yes
- **Learning rate**: 1e-3
- **Epochs**: 15
- **Goal**: Learn optimal fusion without disrupting pre-trained encoders

**Command:**
```bash
python src/training/train_lightning.py \
  --config configs/base.yaml \
  --gpus 4
```

---

### 2. `stage2.yaml` - Stage 2: End-to-End Fine-Tuning
- **Use case**: Fine-tune entire model after Stage 1
- **Modalities**: Vision + Text + Clinical (full multi-modal)
- **Freeze encoders**: No
- **Learning rates**: Discriminative (encoders: 1e-5, fusion/head: 1e-4)
- **Epochs**: 30
- **Goal**: Adapt pre-trained models to ED abnormality detection

**Command:**
```bash
# Resume from Stage 1 checkpoint
python src/training/train_lightning.py \
  --config configs/stage2.yaml \
  --resume-from logs/checkpoints/epoch=14-val_mean_auroc=0.8500.ckpt \
  --gpus 4
```

---

### 3. `vision_only.yaml` - Vision-Only Baseline
- **Use case**: Baseline for comparison
- **Modalities**: Vision only (BiomedCLIP-CXR)
- **Parameters**: ~87M
- **Goal**: Establish single-modality baseline

**Command:**
```bash
python src/training/train_lightning.py \
  --config configs/vision_only.yaml \
  --gpus 4
```

---

### 4. `text_only.yaml` - Text-Only Baseline
- **Use case**: Baseline for comparison
- **Modalities**: Text only (Clinical ModernBERT)
- **Parameters**: ~149M
- **Goal**: Establish single-modality baseline

**Command:**
```bash
python src/training/train_lightning.py \
  --config configs/text_only.yaml \
  --gpus 4
```

---

### 5. `clinical_only.yaml` - Clinical-Only Baseline
- **Use case**: Baseline for comparison
- **Modalities**: Clinical features only (MLP)
- **Parameters**: ~200K
- **Goal**: Establish single-modality baseline

**Command:**
```bash
python src/training/train_lightning.py \
  --config configs/clinical_only.yaml \
  --gpus 4
```

---

## Training Workflow

### Recommended Sequence:

1. **Train all baselines** (for comparison):
   ```bash
   ./scripts/train_all_baselines.sh --gpus 4
   ```

2. **Train Stage 1** (multi-modal, frozen encoders):
   ```bash
   python src/training/train_lightning.py \
     --config configs/base.yaml \
     --gpus 4 \
     --experiment-name "stage1_run1"
   ```

3. **Train Stage 2** (multi-modal, fine-tune all):
   ```bash
   python src/training/train_lightning.py \
     --config configs/stage2.yaml \
     --resume-from logs/checkpoints/best_stage1.ckpt \
     --gpus 4 \
     --experiment-name "stage2_run1"
   ```

4. **Compare results** in TensorBoard:
   ```bash
   tensorboard --logdir tb_logs
   ```

---

## Configuration Override

You can override any config parameter via command line:

```bash
python src/training/train_lightning.py \
  --config configs/base.yaml \
  --batch-size 32 \
  --lr 5e-4 \
  --max-epochs 20
```

---

## Key Hyperparameters

| Parameter | Stage 1 | Stage 2 | Baselines |
|-----------|---------|---------|-----------|
| Freeze encoders | Yes | No | No |
| Learning rate | 1e-3 | 1e-4 (fusion/head) | 1e-4 |
| Encoder LR | N/A | 1e-5 | N/A |
| Epochs | 15 | 30 | 30 |
| Batch size | 16 | 16 | 16 |
| Gradient accum | 2 | 2 | 2 |
| Effective batch | 32 | 32 | 32 |

---

## Expected Results

Based on the MDF-NET paper baseline:

| Model | Expected Mean AUROC |
|-------|---------------------|
| Clinical-only | ~0.74 |
| Text-only | ~0.79 |
| Vision-only | ~0.82 |
| **Full model (Stage 1)** | **~0.85** |
| **Full model (Stage 2)** | **~0.86-0.87** |

Your results may vary based on:
- Data quality
- Hyperparameter tuning
- Random seed
- Number of training epochs
