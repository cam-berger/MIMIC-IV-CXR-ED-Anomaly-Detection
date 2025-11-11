"""
Setup file for Google Cloud Dataflow workers

This file ensures that Dataflow workers have all necessary dependencies installed.
It's referenced by the --setup_file flag in the pipeline.
"""

import setuptools

# Read requirements
with open('requirements_dataflow.txt', 'r') as f:
    REQUIRED_PACKAGES = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setuptools.setup(
    name='mimic-iv-cxr-ed-anomaly-detection-dataflow',
    version='1.0.0',
    description='Dataflow pipeline for MIMIC-IV-CXR-ED preprocessing',
    author='Your Name',
    author_email='your.email@example.com',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
)
