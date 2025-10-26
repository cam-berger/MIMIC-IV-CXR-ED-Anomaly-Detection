# Dockerfile for MIMIC-IV-CXR-ED Preprocessing Pipeline
# Google Cloud Platform support with multi-bucket GCS

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    git \
    curl \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download ModernBERT tokenizer (used for text preprocessing)
RUN python -c "from transformers import AutoTokenizer; \
    AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')"

# Copy source code
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY LOCAL_TESTING.md .
COPY README.md .

# Create directories for data mounting
RUN mkdir -p /data/input /data/output /data/mimic_data

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Default command - can be overridden
CMD ["python", "src/phase1_preprocess.py", "--help"]
