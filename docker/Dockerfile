# Dockerfile for MIMIC Preprocessing Pipeline

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with increased timeout and retries
RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt

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