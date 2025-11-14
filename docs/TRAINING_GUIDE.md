# Training Guide: Fine-Tuning Enhanced MDF-Net

## Overview

This guide provides comprehensive instructions for fine-tuning the Enhanced MDF-Net model for chest X-ray abnormality detection. The training pipeline supports multi-GPU distributed training, baseline comparisons, hyperparameter optimization, and comprehensive experiment tracking with TensorBoard.

## Table of Contents

1. [Training Strategy](#training-strategy)
2. [Baseline Experiments](#baseline-experiments)
3. [Multi-Stage Training](#multi-stage-training)
4. [Hyperparameter Optimization](#hyperparameter-optimization)
5. [Distributed Training](#distributed-training)
6. [Experiment Tracking](#experiment-tracking)
7. [Training Scripts](#training-scripts)
8. [Benchmarking](#benchmarking)

## Training Strategy

### Multi-Stage Approach

The Enhanced MDF-Net uses a **two-stage training strategy** to maximize performance:

**Stage 1: Feature Extraction (10-15 epochs)**
- **Freeze**: BiomedCLIP and Clinical ModernBERT encoders
- **Train**: Fusion layer + Classification head
- **Learning Rate**: 1e-3
- **Goal**: Learn optimal cross-modal fusion without disrupting pre-trained representations

**Stage 2: End-to-End Fine-Tuning (20-30 epochs)**
- **Unfreeze**: All layers
- **Train**: Full model with discriminative learning rates
- **Learning Rates**:
  - Encoders: 1e-5 (very low to preserve pre-training)
  - Fusion layer: 1e-4
  - Classification head: 1e-4
- **Goal**: Adapt pre-trained models to ED abnormality detection

### Training Configuration

```python
# Default hyperparameters
BATCH_SIZE = 16          # Per GPU (effective batch: 16 x 4 GPUs = 64)
MAX_EPOCHS_STAGE1 = 15
MAX_EPOCHS_STAGE2 = 30
WARMUP_EPOCHS = 3
GRADIENT_ACCUMULATION = 2  # Effective batch: 128

# Learning rates
LR_STAGE1 = 1e-3
LR_ENCODERS_STAGE2 = 1e-5
LR_FUSION_STAGE2 = 1e-4
LR_HEAD_STAGE2 = 1e-4

# Regularization
WEIGHT_DECAY = 1e-4
DROPOUT_FUSION = 0.3
DROPOUT_HEAD1 = 0.3
DROPOUT_HEAD2 = 0.2

# Loss
BCE_WEIGHT = 0.7
FOCAL_WEIGHT = 0.3
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
```

## Baseline Experiments

### Single-Modality Baselines

To validate the benefit of multi-modal fusion, train three baseline models:

#### 1. Vision-Only Baseline
```bash
python src/training/train.py \
  --config configs/vision_only.yaml \
  --data-path processed/phase1_output \
  --output-dir experiments/vision_only \
  --modality vision \
  --gpu-ids 0,1,2,3
```

**Architecture:**
- BiomedCLIP-CXR encoder
- 2-layer classification head
- ~87M parameters

#### 2. Text-Only Baseline
```bash
python src/training/train.py \
  --config configs/text_only.yaml \
  --data-path processed/phase1_output \
  --output-dir experiments/text_only \
  --modality text \
  --gpu-ids 0,1,2,3
```

**Architecture:**
- Clinical ModernBERT encoder
- 2-layer classification head
- ~149M parameters

#### 3. Clinical-Only Baseline
```bash
python src/training/train.py \
  --config configs/clinical_only.yaml \
  --data-path processed/phase1_output \
  --output-dir experiments/clinical_only \
  --modality clinical \
  --gpu-ids 0,1,2,3
```

**Architecture:**
- Dense feature encoder
- 2-layer classification head
- ~525K parameters

### Multi-Modal Experiments

#### Full Model (All Three Modalities)
```bash
python src/training/train.py \
  --config configs/full_model.yaml \
  --data-path processed/phase1_output \
  --output-dir experiments/full_model \
  --modality all \
  --gpu-ids 0,1,2,3
```

#### Ablation Studies

**Vision + Text (No Clinical)**
```bash
python src/training/train.py \
  --config configs/vision_text.yaml \
  --data-path processed/phase1_output \
  --output-dir experiments/vision_text \
  --modality vision,text \
  --gpu-ids 0,1,2,3
```

**Vision + Clinical (No Text)**
```bash
python src/training/train.py \
  --config configs/vision_clinical.yaml \
  --data-path processed/phase1_output \
  --output-dir experiments/vision_clinical \
  --modality vision,clinical \
  --gpu-ids 0,1,2,3
```

**Text + Clinical (No Vision)**
```bash
python src/training/train.py \
  --config configs/text_clinical.yaml \
  --data-path processed/phase1_output \
  --output-dir experiments/text_clinical \
  --modality text,clinical \
  --gpu-ids 0,1,2,3
```

## Multi-Stage Training

### Stage 1: Feature Extraction

```bash
# Train fusion and classification layers with frozen encoders
python src/training/train_stage1.py \
  --data-path processed/phase1_output \
  --output-dir experiments/stage1 \
  --batch-size 16 \
  --epochs 15 \
  --lr 1e-3 \
  --warmup-epochs 3 \
  --gpu-ids 0,1,2,3 \
  --tensorboard-dir runs/stage1
```

**Stage 1 Workflow:**
1. Load pre-trained BiomedCLIP and ModernBERT
2. Freeze encoder parameters
3. Initialize fusion layer and classification head randomly
4. Train with high learning rate (1e-3)
5. Monitor validation AUROC
6. Save best checkpoint

**Expected Results:**
- Training time: ~2-3 hours on 4x H100
- Validation AUROC: ~0.80-0.82
- Convergence: ~10-12 epochs

### Stage 2: End-to-End Fine-Tuning

```bash
# Fine-tune entire model with discriminative learning rates
python src/training/train_stage2.py \
  --data-path processed/phase1_output \
  --checkpoint experiments/stage1/best_model.pth \
  --output-dir experiments/stage2 \
  --batch-size 16 \
  --epochs 30 \
  --lr-encoders 1e-5 \
  --lr-fusion 1e-4 \
  --lr-head 1e-4 \
  --warmup-epochs 3 \
  --gpu-ids 0,1,2,3 \
  --tensorboard-dir runs/stage2
```

**Stage 2 Workflow:**
1. Load Stage 1 checkpoint
2. Unfreeze all layers
3. Apply discriminative learning rates
4. Fine-tune with early stopping
5. Save checkpoints every epoch
6. Monitor multiple metrics (AUROC, AUPRC, F1)

**Expected Results:**
- Training time: ~6-8 hours on 4x H100
- Validation AUROC: ~0.85-0.88
- Improvement over Stage 1: +3-6% AUROC

## Hyperparameter Optimization

### Grid Search

```bash
# Run grid search over key hyperparameters
python src/training/hyperparam_search.py \
  --data-path processed/phase1_output \
  --output-dir experiments/grid_search \
  --search-type grid \
  --gpu-ids 0,1,2,3
```

**Search Space:**
```python
{
    'lr_stage1': [1e-3, 5e-4, 2e-3],
    'lr_encoders_stage2': [1e-5, 5e-6, 2e-5],
    'lr_fusion_stage2': [1e-4, 5e-5, 2e-4],
    'weight_decay': [1e-4, 1e-5, 1e-3],
    'dropout_fusion': [0.2, 0.3, 0.4],
    'dropout_head': [0.2, 0.3, 0.4],
    'bce_weight': [0.6, 0.7, 0.8],
    'focal_alpha': [0.2, 0.25, 0.3],
    'focal_gamma': [1.5, 2.0, 2.5]
}
```

### Random Search

```bash
# Faster alternative: random search
python src/training/hyperparam_search.py \
  --data-path processed/phase1_output \
  --output-dir experiments/random_search \
  --search-type random \
  --n-trials 50 \
  --gpu-ids 0,1,2,3
```

### Bayesian Optimization (Optuna)

```bash
# Most efficient: Bayesian optimization
python src/training/hyperparam_search.py \
  --data-path processed/phase1_output \
  --output-dir experiments/bayesian \
  --search-type bayesian \
  --n-trials 100 \
  --gpu-ids 0,1,2,3
```

**Optimization Objective:**
- Primary: Mean AUROC across all 14 abnormalities
- Secondary: Minimum AUROC (worst-performing class)
- Constraint: Training time < 12 hours

## Distributed Training

### Multi-GPU on Single Node

```bash
# Distributed Data Parallel (DDP) on 4 GPUs
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --master_port=29500 \
  src/training/train_distributed.py \
  --data-path processed/phase1_output \
  --output-dir experiments/ddp_4gpu \
  --batch-size 16 \
  --epochs 30 \
  --use-amp
```

**Performance:**
- 4x H100 GPUs: ~3.8x speedup (ideal: 4x)
- Mixed precision (AMP): ~2x additional speedup
- Effective batch size: 64 (16 per GPU)

### Multi-Node Training (Optional)

```bash
# Master node (node 0)
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr="10.0.0.1" \
  --master_port=29500 \
  src/training/train_distributed.py \
  --data-path gs://bergermimiciv/processed/phase1_output \
  --output-dir gs://bergermimiciv/experiments/ddp_8gpu \
  --batch-size 16

# Worker node (node 1)
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr="10.0.0.1" \
  --master_port=29500 \
  src/training/train_distributed.py \
  --data-path gs://bergermimiciv/processed/phase1_output \
  --output-dir gs://bergermimiciv/experiments/ddp_8gpu \
  --batch-size 16
```

## Experiment Tracking

### TensorBoard Integration

```bash
# Start TensorBoard server
tensorboard --logdir experiments/ --port 6006

# Access at http://localhost:6006
```

**Logged Metrics:**

**Per-Epoch Metrics:**
- Training loss (BCE, Focal, Combined)
- Validation loss
- Mean AUROC (across all abnormalities)
- Per-abnormality AUROC (14 curves)
- Mean AUPRC
- Macro/Micro F1
- Learning rates (per parameter group)

**Per-Batch Metrics:**
- Batch loss
- Gradient norms
- GPU memory usage
- Throughput (samples/sec)

**Model Checkpoints:**
- Best validation AUROC
- Best mean AUROC
- Best rare-class performance
- Latest checkpoint (every 5 epochs)

### Logging Structure

```
experiments/
├── vision_only/
│   ├── checkpoints/
│   │   ├── best_auroc.pth
│   │   ├── best_mean.pth
│   │   └── epoch_*.pth
│   ├── logs/
│   │   └── train.log
│   ├── tensorboard/
│   │   └── events.out.tfevents.*
│   └── config.yaml
├── text_only/
│   └── ...
├── full_model/
│   └── ...
└── results_summary.csv
```

## Training Scripts

### Quick Start: Full Training Pipeline

```bash
# Complete training pipeline: Stage 1 → Stage 2 → Evaluation
bash scripts/train_full_pipeline.sh \
  --data-path processed/phase1_output \
  --output-dir experiments/full_training \
  --gpu-ids 0,1,2,3
```

### Step-by-Step Training

```bash
# 1. Baseline: Vision Only
python src/training/train.py \
  --config configs/vision_only.yaml \
  --gpu-ids 0,1,2,3

# 2. Baseline: Text Only
python src/training/train.py \
  --config configs/text_only.yaml \
  --gpu-ids 0,1,2,3

# 3. Baseline: Clinical Only
python src/training/train.py \
  --config configs/clinical_only.yaml \
  --gpu-ids 0,1,2,3

# 4. Full Model: Stage 1
python src/training/train_stage1.py \
  --config configs/full_model_stage1.yaml \
  --gpu-ids 0,1,2,3

# 5. Full Model: Stage 2
python src/training/train_stage2.py \
  --config configs/full_model_stage2.yaml \
  --checkpoint experiments/full_model_stage1/best_model.pth \
  --gpu-ids 0,1,2,3

# 6. Evaluate All Models
python src/evaluation/evaluate_all.py \
  --experiments-dir experiments/ \
  --output-dir results/
```

## Benchmarking

### Comparison with MDF-NET Paper

**Target Metrics (from MDF-NET paper):**

| Model | Mean AUROC | Atelectasis | Cardiomegaly | Edema | Pneumonia | Pneumothorax |
|-------|------------|-------------|--------------|-------|-----------|--------------|
| Vision-only | 0.810 | 0.792 | 0.881 | 0.901 | 0.768 | 0.887 |
| MDF-NET | 0.856 | 0.831 | 0.912 | 0.931 | 0.821 | 0.914 |

**Our Goals:**
- **Enhanced MDF-NET**: Mean AUROC ≥ 0.860 (+0.004 over MDF-NET)
- **Vision-only baseline**: Mean AUROC ≥ 0.810 (match paper)
- **Improvement**: +5-6% AUROC with multi-modal fusion

### Performance Metrics

**Training Speed:**
- 4x H100 GPUs (80GB each)
- Batch size: 16 per GPU (64 total)
- Throughput: ~200 samples/sec
- Stage 1 (15 epochs): ~2-3 hours
- Stage 2 (30 epochs): ~6-8 hours
- **Total training time**: ~8-11 hours

**Resource Usage:**
- GPU memory: ~35GB per GPU (with mixed precision)
- RAM: ~64GB (data loading + caching)
- Storage: ~50GB (checkpoints + logs)

## Training Tips

### Best Practices

1. **Start with small samples**
   ```bash
   # Verify training loop works on small data
   python src/training/train.py \
     --data-path processed/phase1_output \
     --use-small-sample \
     --epochs 3 \
     --gpu-ids 0
   ```

2. **Monitor gradient norms**
   - Stage 1: Gradients should be ~1.0-10.0
   - Stage 2: Gradients should be ~0.1-1.0
   - If exploding (>100): Reduce learning rate or add gradient clipping

3. **Watch for overfitting**
   - Train-val loss gap increasing: Add dropout
   - Val AUROC plateaus early: Reduce model complexity
   - Use early stopping (patience=5)

4. **Learning rate warmup**
   - Essential for stable training
   - Warmup for first 3 epochs
   - Prevents early divergence

5. **Checkpoint strategy**
   - Save best validation AUROC (primary)
   - Save best mean AUROC (across classes)
   - Save every 5 epochs for analysis
   - Keep last 3 checkpoints only (disk space)

### Common Issues

**Issue: Out of Memory**
```bash
# Solution 1: Reduce batch size
--batch-size 8  # Down from 16

# Solution 2: Use gradient accumulation
--gradient-accumulation 4  # Effective batch: 8 x 4 = 32

# Solution 3: Enable mixed precision
--use-amp
```

**Issue: Training diverges (NaN loss)**
```bash
# Solution: Add gradient clipping
--max-grad-norm 1.0

# Or reduce learning rate
--lr 5e-4  # Down from 1e-3
```

**Issue: Slow convergence**
```bash
# Solution: Increase learning rate (carefully)
--lr 2e-3  # Up from 1e-3

# Or reduce warmup epochs
--warmup-epochs 1
```

**Issue: Poor rare-class performance**
```bash
# Solution: Increase focal loss weight
--focal-weight 0.5  # Up from 0.3

# Or increase focal gamma
--focal-gamma 3.0  # Up from 2.0
```

## Next Steps

After training completes:

1. **Evaluate models**: See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
2. **Analyze results**: Compare baselines vs full model
3. **Generate visualizations**: Attention maps, ROC curves, confusion matrices
4. **Error analysis**: Identify failure modes
5. **Model selection**: Choose best checkpoint for deployment

## Additional Resources

- **Model Architecture**: [README.md](../README.md#model-architecture)
- **Loss Functions**: [README.md](../README.md#loss-functions)
- **Data Pipeline**: [PHASE3_INTEGRATION.md](PHASE3_INTEGRATION.md)
- **Evaluation Guide**: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)

## Summary

This training guide provides:
- ✅ Multi-stage training strategy (Stage 1 + Stage 2)
- ✅ Baseline comparison experiments (Vision/Text/Clinical-only)
- ✅ Hyperparameter optimization (Grid/Random/Bayesian)
- ✅ Distributed multi-GPU training (DDP)
- ✅ TensorBoard experiment tracking
- ✅ Comprehensive training scripts
- ✅ Benchmarking against MDF-NET paper
- ✅ Best practices and troubleshooting

**Estimated time to reproduce**: 1-2 days
- Setup + small sample testing: 2-4 hours
- Baseline experiments: 12-16 hours
- Full model training: 8-11 hours
- Hyperparameter search (optional): 24-48 hours
