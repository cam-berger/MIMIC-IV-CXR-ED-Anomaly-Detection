#!/bin/bash
# Lambda Labs A100 Setup Script
# ==============================
# Sets up environment on Lambda PyTorch image

set -e

echo "=========================================="
echo "Lambda Labs A100 Setup"
echo "=========================================="
echo ""

# Check GPU
echo "Step 1: Verifying A100 GPU..."
nvidia-smi
echo ""

# Check if using Lambda's PyTorch image (conda should be pre-installed)
if command -v conda &> /dev/null; then
    echo "✓ Conda detected (Lambda PyTorch image)"

    # Check PyTorch version
    echo "Step 2: Checking PyTorch installation..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"
    echo ""

    # Create or use existing environment
    echo "Step 3: Setting up conda environment..."
    if conda env list | grep -q "ml_env"; then
        echo "✓ ml_env already exists, activating..."
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate ml_env
    else
        echo "Creating ml_env environment..."
        conda create -n ml_env python=3.11 -y
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate ml_env

        # PyTorch should already be installed on Lambda image
        # But install in conda env to be safe
        echo "Installing PyTorch in ml_env..."
        conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
    fi
else
    echo "⚠️  Conda not found. Are you using a Lambda PyTorch image?"
    echo "Attempting to install miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
    ~/miniconda3/bin/conda init
    echo ""
    echo "Please restart your shell and run this script again."
    exit 1
fi

# Install additional dependencies
echo "Step 4: Installing additional dependencies..."
pip install --upgrade pip

# Install from requirements.txt (skip torch/torchvision as already installed)
echo "Installing core dependencies..."
pip install pandas>=2.0.0 \
    numpy>=1.24.0,<2.0 \
    tqdm>=4.65.0 \
    pillow>=10.0.0 \
    opencv-python-headless>=4.8.0 \
    transformers>=4.30.0 \
    sentence-transformers>=2.2.0 \
    tokenizers>=0.13.0 \
    faiss-cpu>=1.7.0 \
    scikit-learn>=1.3.0 \
    scipy>=1.11.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    pytorch-lightning>=2.0.0 \
    tensorboard>=2.13.0

# Install OpenCLIP for BiomedCLIP
echo "Installing OpenCLIP for BiomedCLIP..."
pip install open-clip-torch>=2.20.0

# Install spaCy
echo "Installing spaCy..."
pip install spacy>=3.5.0
python -m spacy download en_core_web_sm

echo ""
echo "Step 5: Verifying installation..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import pytorch_lightning as pl; print(f'✓ Lightning {pl.__version__}')"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"
python -c "import open_clip; print(f'✓ OpenCLIP available')"
echo ""

# Check GPU with PyTorch
echo "Step 6: Verifying GPU access from PyTorch..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0)}'); print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
echo ""

echo "=========================================="
echo "✅ Lambda A100 setup complete!"
echo "=========================================="
echo ""
echo "GPU Details:"
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv
echo ""
echo "Next steps:"
echo "  1. Upload data: rsync -avz local_data/ ubuntu@\$(hostname -I | awk '{print \$1}'):~/data/"
echo "  2. Update config: Edit configs/lambda_a100.yaml data_root path"
echo "  3. Test pipeline: python scripts/test_training_pipeline.py --config configs/lambda_a100.yaml"
echo "  4. Start training: ./scripts/run_lambda_a100.sh"
echo ""
