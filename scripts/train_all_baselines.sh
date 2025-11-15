#!/bin/bash
#
# Train All Baseline Models
# ==========================
# Trains vision-only, text-only, and clinical-only baselines sequentially
#
# Usage:
#   ./scripts/train_all_baselines.sh [--gpus N] [--data-root PATH]
#

set -e  # Exit on error

# Parse arguments
GPUS=1
DATA_ROOT="/media/dev/MIMIC_DATA/phase1_with_path_fixes_raw"

while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--gpus N] [--data-root PATH]"
      exit 1
      ;;
  esac
done

echo "=========================================="
echo "Training All Baseline Models"
echo "=========================================="
echo "GPUs: $GPUS"
echo "Data root: $DATA_ROOT"
echo ""

# Vision-Only Baseline
echo "=========================================="
echo "1/3: Training Vision-Only Baseline"
echo "=========================================="
python src/training/train_lightning.py \
  --config configs/vision_only.yaml \
  --data-root "$DATA_ROOT" \
  --gpus $GPUS \
  --experiment-name "vision_only_baseline"

echo ""
echo "✓ Vision-only baseline completed"
echo ""

# Text-Only Baseline
echo "=========================================="
echo "2/3: Training Text-Only Baseline"
echo "=========================================="
python src/training/train_lightning.py \
  --config configs/text_only.yaml \
  --data-root "$DATA_ROOT" \
  --gpus $GPUS \
  --experiment-name "text_only_baseline"

echo ""
echo "✓ Text-only baseline completed"
echo ""

# Clinical-Only Baseline
echo "=========================================="
echo "3/3: Training Clinical-Only Baseline"
echo "=========================================="
python src/training/train_lightning.py \
  --config configs/clinical_only.yaml \
  --data-root "$DATA_ROOT" \
  --gpus $GPUS \
  --experiment-name "clinical_only_baseline"

echo ""
echo "✓ Clinical-only baseline completed"
echo ""

echo "=========================================="
echo "All Baselines Completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - logs/vision_only_baseline/"
echo "  - logs/text_only_baseline/"
echo "  - logs/clinical_only_baseline/"
echo ""
echo "View results with TensorBoard:"
echo "  tensorboard --logdir tb_logs"
echo ""
