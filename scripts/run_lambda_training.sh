#!/bin/bash
# Lambda AI Training Script
# =========================
# Automates training on Lambda Labs instances

set -e

echo "=========================================="
echo "Lambda AI Training Script"
echo "=========================================="
echo ""

# Activate environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml_env

# Navigate to project
cd ~/MIMIC-IV-CXR-ED-Anomaly-Detection

# Verify GPU
echo "Checking GPU availability..."
nvidia-smi
python -c "import torch; print(f'PyTorch sees {torch.cuda.device_count()} GPU(s)')"
echo ""

# Run pipeline test first
echo "=========================================="
echo "Step 1: Running pipeline validation test"
echo "=========================================="
python scripts/test_training_pipeline.py --config configs/lambda_validation.yaml
echo ""

# If test passes, start training
echo "=========================================="
echo "Step 2: Starting training"
echo "=========================================="
echo "Training will run in background..."
echo "Monitor with: tail -f ~/MIMIC-IV-CXR-ED-Anomaly-Detection/training.log"
echo ""

# Start training with logging
nohup python src/training/train_lightning.py \
  --config configs/lambda_validation.yaml \
  --experiment-name "lambda_validation_run1" \
  > training.log 2>&1 &

# Save PID
echo $! > training.pid
echo "Training PID: $(cat training.pid)"
echo ""

# Show initial logs
sleep 5
tail -30 training.log

echo ""
echo "=========================================="
echo "Training started successfully!"
echo "=========================================="
echo ""
echo "Monitoring commands:"
echo "  tail -f training.log              # Watch logs"
echo "  watch -n 10 nvidia-smi           # Monitor GPU"
echo "  tensorboard --logdir tb_logs     # View metrics"
echo "  kill \$(cat training.pid)         # Stop training"
echo ""
echo "Checkpoints saved to: logs/checkpoints/"
echo "TensorBoard logs: tb_logs/"
echo ""
