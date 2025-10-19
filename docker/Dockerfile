# Dockerfile for MIMIC Preprocessing Pipeline

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# Added build-essential for compiling NLP libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with increased timeout and retries
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt

# Download SciSpacy medical NLP model
RUN pip install --no-cache-dir \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz

# Download BioBERT model weights (cached for transformers)
# This will download on first use, but we can pre-cache it
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1'); \
    AutoModel.from_pretrained('dmis-lab/biobert-v1.1')"

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "src.aws_processor"]