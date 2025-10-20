#!/bin/bash
# EC2 Setup Script for MIMIC Phase 1 Processing
# This script automates the setup of the environment on a fresh EC2 instance

set -e  # Exit on error

echo "========================================="
echo "MIMIC Phase 1 - EC2 Setup Script"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu/Debian
if ! command -v apt &> /dev/null; then
    print_error "This script is designed for Ubuntu/Debian. Please install dependencies manually."
    exit 1
fi

# Update system
print_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
print_info "Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install Git if not present
print_info "Ensuring Git is installed..."
sudo apt install -y git

# Install system dependencies
print_info "Installing system dependencies..."
sudo apt install -y build-essential libssl-dev libffi-dev

# Create project directory
PROJECT_DIR="$HOME/MIMIC-IV-CXR-ED-Anomaly-Detection"

if [ -d "$PROJECT_DIR" ]; then
    print_warning "Project directory already exists. Skipping clone."
    cd "$PROJECT_DIR"
    git pull
else
    print_info "Cloning repository..."
    cd "$HOME"
    git clone https://github.com/yourusername/MIMIC-IV-CXR-ED-Anomaly-Detection.git
    cd "$PROJECT_DIR"
fi

# Create virtual environment
print_info "Creating Python virtual environment..."
python3.11 -m venv venv

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
print_info "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_info "Creating necessary directories..."
mkdir -p logs
mkdir -p data/temp

# Verify AWS credentials
print_info "Verifying AWS credentials..."
if aws sts get-caller-identity &> /dev/null; then
    print_info "AWS credentials verified ✓"
    aws sts get-caller-identity
else
    print_warning "AWS credentials not configured. Please configure or attach IAM role to EC2 instance."
fi

# Create activation script
print_info "Creating activation script..."
cat > activate.sh << 'EOL'
#!/bin/bash
# Quick activation script
cd ~/MIMIC-IV-CXR-ED-Anomaly-Detection
source venv/bin/activate
echo "Environment activated! You can now run phase1_stay_identification.py"
EOL
chmod +x activate.sh

# Create example run script
print_info "Creating example run script..."
cat > run_phase1_test.sh << 'EOL'
#!/bin/bash
# Test run script for Phase 1
# Edit the parameters below before running

# Configuration
S3_BUCKET="your-mimic-data-bucket"
OUTPUT_BUCKET="your-output-bucket"
BASE_PATH="physionet.org/files"
NUM_SUBJECTS=10
OUTPUT_PATH="test/phase1_$(date +%Y%m%d_%H%M%S)"

# Activate environment
source venv/bin/activate

# Run Phase 1
python src/phase1_stay_identification.py \
  --base-path "$BASE_PATH" \
  --s3-bucket "$S3_BUCKET" \
  --num-subjects "$NUM_SUBJECTS" \
  --output-path "$OUTPUT_PATH" \
  --output-s3-bucket "$OUTPUT_BUCKET" \
  --time-window 24

echo "Phase 1 processing complete!"
echo "Output saved to: s3://$OUTPUT_BUCKET/$OUTPUT_PATH"
EOL
chmod +x run_phase1_test.sh

# System information
print_info "System Information:"
echo "  CPU: $(nproc) cores"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "  Disk: $(df -h / | tail -1 | awk '{print $4}') available"
echo "  Python: $(python --version)"

echo ""
print_info "========================================="
print_info "Setup Complete!"
print_info "========================================="
echo ""
echo "Next steps:"
echo "  1. Edit run_phase1_test.sh with your S3 bucket names"
echo "  2. Run: ./run_phase1_test.sh"
echo "  3. Or activate environment: source activate.sh"
echo ""
echo "To verify S3 access:"
echo "  aws s3 ls s3://your-mimic-data-bucket/"
echo ""
print_info "Happy processing!"
