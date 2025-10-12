# MIMIC Multi-Modal Dataset Preprocessor

A scalable cloud-based pipeline for preprocessing and integrating MIMIC-IV, MIMIC-CXR, and MIMIC-ED datasets into a unified multi-modal dataset for medical abnormality detection and classification.

## Overview

This project implements the MIMIC-Eye methodology to create patient-level multi-modal datasets that link:
- Clinical data (vital signs, demographics, lab results)
- Medical images (chest X-rays)
- Medications and prescriptions
- Radiology reports
- Blood tests and procedures

## Key Features

- **Patient-Centric Organization**: Each patient has their own folder containing all related data
- **Temporal Alignment**: Links chest X-rays to emergency department stays and clinical data using timestamp matching
- **Cloud-Native**: Optimized for AWS, GCP, and Azure with parallel processing
- **Scalable**: Handles 300K+ patients and 377K+ chest X-rays
- **Reproducible**: Follows standardized MIMIC-Eye integration methodology

## Pipeline Phases

1. **Phase 1 - Stay ID Identification**: Links chest X-rays to ED visits via temporal matching
2. **Phase 2 - Clinical Data Extraction**: Gathers vital signs, labs, medications, and demographics
3. **Phase 3 - Integration**: Creates patient-level folder structure with all modalities

## Use Cases

- Multi-modal deep learning for medical diagnosis
- Abnormality detection and classification
- Clinical context-aware image analysis
- Radiologist decision-making pattern research

## Requirements

- Access to MIMIC-IV, MIMIC-CXR, and MIMIC-ED (requires PhysioNet credentialing)
- Cloud computing account (AWS/GCP/Azure)
- Python 3.9+

## Citation

Based on the MIMIC-Eye methodology:
