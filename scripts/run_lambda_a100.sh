#!/bin/bash
# Lambda A100 Training Script
# ============================
# Optimized for A100 instances with larger batches and faster training

set -e

echo "=========================================="
echo "Lambda A100 Training Script"
echo "=========================================="
echo ""

# Activate environment
echo "Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml_env

# Navigate to project
cd ~/MIMIC-IV-CXR-ED-Anomaly-Detection || {
    echo "Error: Project directory not found"
    echo "Please clone the repo first:"
    echo "  git clone https://github.com/cam-berger/MIMIC-IV-CXR-ED-Anomaly-Detection.git"
    exit 1
}

# Verify GPU
echo "Verifying A100 GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'✓ PyTorch sees {torch.cuda.device_count()} GPU(s): {torch.cuda.get_device_name(0)}')"
echo ""

# Check A100-specific features
echo "Checking A100 capabilities..."
python << 'EOF'
import torch
device = torch.device('cuda:0')
props = torch.cuda.get_device_properties(device)
print(f"GPU: {props.name}")
print(f"Memory: {props.total_memory / 1e9:.1f} GB")
print(f"Compute Capability: {props.major}.{props.minor}")
print(f"TF32 Support: {torch.backends.cuda.matmul.allow_tf32}")
print(f"BF16 Support: {props.major >= 8}")  # A100 is compute 8.0
EOF
echo ""

# Run pipeline test first
echo "=========================================="
echo "Step 1: Running pipeline validation test"
echo "=========================================="
python scripts/test_training_pipeline.py --config configs/lambda_a100.yaml || {
    echo ""
    echo "❌ Pipeline test failed!"
    echo "Please check the error above and fix before training."
    exit 1
}
echo ""

# If test passes, start training
echo "=========================================="
echo "Step 2: Starting training on A100"
echo "=========================================="
echo "Configuration:"
echo "  - Batch size: 16 (larger for A100's VRAM)"
echo "  - Mixed precision: FP16 (A100 tensor cores)"
echo "  - Workers: 12 (utilizing A100 instance CPUs)"
echo "  - Max epochs: 10 (validation run)"
echo ""
echo "Training will run in background..."
echo "Monitor with: tail -f ~/MIMIC-IV-CXR-ED-Anomaly-Detection/training.log"
echo ""

# Start training with logging
nohup python src/training/train_lightning.py \
  --config configs/lambda_a100.yaml \
  --experiment-name "lambda_a100_run1" \
  > training.log 2>&1 &

# Save PID
echo $! > training.pid
echo "Training PID: $(cat training.pid)"
echo ""

# Wait a moment for training to start
sleep 10

# Show initial logs
echo "Initial training output:"
echo "----------------------------------------"
tail -50 training.log
echo "----------------------------------------"
echo ""

echo "=========================================="
echo "✅ Training started successfully!"
echo "=========================================="
echo ""
echo "Monitoring commands:"
echo "  tail -f training.log              # Watch logs in real-time"
echo "  watch -n 5 nvidia-smi            # Monitor GPU usage"
echo "  tensorboard --logdir tb_logs --host 0.0.0.0 --port 6006"
echo "  kill \$(cat training.pid)         # Stop training"
echo ""
echo "Files:"
echo "  Checkpoints: logs/checkpoints/"
echo "  TensorBoard: tb_logs/"
echo "  Training log: training.log"
echo ""
echo "Estimated completion: ~1-1.5 hours (A100 is fast!)"
echo "Estimated cost: ~\$1.30-2.00 (depending on 40GB vs 80GB A100)"
echo ""
