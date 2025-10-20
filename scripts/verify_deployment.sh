#!/bin/bash
# Deployment Verification Script
# Checks that all prerequisites are met before deployment

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "MIMIC Phase 1 - Deployment Verification"
echo "========================================="
echo ""

# Check counters
passed=0
failed=0
warnings=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((passed++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((failed++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((warnings++))
}

# Check Python version
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
        check_pass "Python $PYTHON_VERSION (>= 3.11)"
    else
        check_fail "Python $PYTHON_VERSION (need >= 3.11)"
    fi
else
    check_fail "Python 3 not found"
fi

# Check virtual environment
echo ""
echo "Checking virtual environment..."
if [ -d "venv" ] || [ -d ".venv" ]; then
    check_pass "Virtual environment exists"

    # Check if activated
    if [ -n "$VIRTUAL_ENV" ]; then
        check_pass "Virtual environment activated"
    else
        check_warn "Virtual environment not activated (run: source venv/bin/activate)"
    fi
else
    check_fail "Virtual environment not found (run: python3.11 -m venv venv)"
fi

# Check dependencies
echo ""
echo "Checking Python dependencies..."
dependencies=("pandas" "numpy" "boto3" "pydicom" "tqdm" "pyarrow")

for dep in "${dependencies[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        VERSION=$(python3 -c "import $dep; print($dep.__version__)" 2>/dev/null || echo "unknown")
        check_pass "$dep ($VERSION)"
    else
        check_fail "$dep not installed"
    fi
done

# Check AWS CLI
echo ""
echo "Checking AWS configuration..."
if command -v aws &> /dev/null; then
    AWS_VERSION=$(aws --version | cut -d' ' -f1 | cut -d'/' -f2)
    check_pass "AWS CLI installed ($AWS_VERSION)"

    # Check credentials
    if aws sts get-caller-identity &> /dev/null; then
        ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
        ARN=$(aws sts get-caller-identity --query Arn --output text 2>/dev/null)
        check_pass "AWS credentials valid"
        echo "   Account: $ACCOUNT"
        echo "   ARN: $ARN"
    else
        check_fail "AWS credentials not configured or invalid"
    fi
else
    check_warn "AWS CLI not installed (optional for local mode)"
fi

# Check required files
echo ""
echo "Checking project files..."
required_files=(
    "src/phase1_stay_identification.py"
    "requirements.txt"
    "docs/EC2_DEPLOYMENT.md"
    "DEPLOYMENT_CHECKLIST.md"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        check_pass "$file exists"
    else
        check_fail "$file not found"
    fi
done

# Check directory structure
echo ""
echo "Checking directory structure..."
required_dirs=("src" "scripts" "docs" "logs" "config")

for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        check_pass "$dir/ directory exists"
    else
        check_warn "$dir/ directory not found (will be created if needed)"
    fi
done

# Check disk space
echo ""
echo "Checking system resources..."
DISK_AVAIL=$(df -h . | tail -1 | awk '{print $4}')
check_pass "Disk space available: $DISK_AVAIL"

# Check memory
TOTAL_MEM=$(free -h | grep Mem | awk '{print $2}')
AVAIL_MEM=$(free -h | grep Mem | awk '{print $7}')
echo "   Total memory: $TOTAL_MEM"
echo "   Available memory: $AVAIL_MEM"

MEM_GB=$(free -m | grep Mem | awk '{print $2}')
if [ "$MEM_GB" -ge 16000 ]; then
    check_pass "Memory: ${MEM_GB}MB (>= 16GB recommended)"
elif [ "$MEM_GB" -ge 8000 ]; then
    check_warn "Memory: ${MEM_GB}MB (16GB+ recommended for production)"
else
    check_fail "Memory: ${MEM_GB}MB (< 8GB - insufficient)"
fi

# Summary
echo ""
echo "========================================="
echo "Verification Summary"
echo "========================================="
echo -e "${GREEN}Passed:${NC} $passed"
echo -e "${YELLOW}Warnings:${NC} $warnings"
echo -e "${RED}Failed:${NC} $failed"
echo ""

if [ $failed -eq 0 ]; then
    if [ $warnings -eq 0 ]; then
        echo -e "${GREEN}✓ All checks passed! Ready for deployment.${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠ Some warnings present. Review before deployment.${NC}"
        exit 0
    fi
else
    echo -e "${RED}✗ Some checks failed. Fix issues before deployment.${NC}"
    exit 1
fi
