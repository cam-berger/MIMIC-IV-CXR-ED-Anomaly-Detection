# MIMIC Multi-Modal Dataset 
HYPOTHESIS: Context-aware knowledge augmentation of clinical notes, when fused with visual features through cross-modal attention, will improve both the accuracy and interpretability of chest X-ray abnormality detection compared to models using raw clinical notes or images alone, with the improvement being most significant for rare conditions and complex multi-abnormality cases.

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
