# Documentation Index

This directory contains comprehensive documentation for the MIMIC-IV-CXR-ED Anomaly Detection project.

## Quick Navigation

### 🚀 Getting Started
- **[Quick Start Guide](QUICK_START.md)** - Fast-track setup and deployment
- **[Local Testing Guide](LOCAL_TESTING.md)** - Test preprocessing locally before cloud deployment
- **[Deployment Quick Start](DEPLOYMENT_QUICKSTART.md)** - Quick reference for GCP deployment

### 🏗️ Architecture & Implementation

#### CXR-PRO Integration (4-Modal Architecture)
- **[CXR-PRO Integration Guide](CXR_PRO_INTEGRATION.md)** - Complete guide for prior-free radiology reports
  - GILBERT model usage (BioBERT-based prior removal)
  - Phase 1 adapter for integrating CXR-PRO impressions
  - 4-modal Enhanced MDF-Net architecture
  - Data quality validation and troubleshooting
- **[CXR-PRO Implementation Summary](CXR_PRO_IMPLEMENTATION_SUMMARY.md)** - Technical implementation details and status

#### Phase 3 Training Pipeline
- **[Enhanced RAG Adapter](ENHANCED_RAG_ADAPTER.md)** - Auto-detection and conversion between Enhanced RAG and Standard formats
- **[Phase 3 Integration Guide](PHASE3_INTEGRATION.md)** - Multi-modal data integration and final dataset preparation
- **[Enhanced RAG Integration Summary](ENHANCED_RAG_INTEGRATION_SUMMARY.md)** - Complete Phase 3 integration guide
- **[Official Splits Documentation](OFFICIAL_SPLITS_FIX.md)** - Patient-level splits implementation (377K studies)

#### Data Processing
- **[Phase 2 Enhanced Notes](PHASE2_ENHANCED_NOTES.md)** - Pseudo-note generation and RAG enhancement
- **[Phase 2 Refactoring Summary](PHASE2_REFACTORING_SUMMARY.md)** - Complete Phase 2 implementation details
- **[OOM Solution Guide](OOM_SOLUTION.md)** - Memory management and optimization strategies
- **[Dataflow Setup](DATAFLOW_SETUP.md)** - Google Cloud Dataflow configuration

### 📊 Training & Evaluation
- **[Training Guide](TRAINING_GUIDE.md)** - Fine-tuning Enhanced MDF-Net with multi-GPU support
- **[Evaluation Guide](EVALUATION_GUIDE.md)** - Metrics, confusion matrices, and correlation analysis
- **[Training Plan Day 1](TRAINING_PLAN_DAY1.md)** - Initial training schedule and milestones

### 🔧 Troubleshooting & Maintenance
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Comprehensive troubleshooting for training pipeline
- **[Bug Fixes Summary](BUG_FIXES_SUMMARY.md)** - Critical bug fixes (DataLoader, BiomedCLIP, ModernBERT, dimensions)
- **[Codebase Audit](CODEBASE_AUDIT.md)** - Complete audit report and production readiness assessment

### 💡 Advanced Topics
- **[Maximizing Claude Code](MAXIMIZING_CLAUDE_CODE.md)** - Best practices for using Claude Code with this project

## Documentation Structure

```
docs/
├── README.md (this file)
│
├── Getting Started/
│   ├── QUICK_START.md
│   ├── LOCAL_TESTING.md
│   └── DEPLOYMENT_QUICKSTART.md
│
├── Architecture/
│   ├── CXR_PRO_INTEGRATION.md (NEW - 4-Modal)
│   ├── CXR_PRO_IMPLEMENTATION_SUMMARY.md
│   ├── ENHANCED_RAG_ADAPTER.md
│   ├── PHASE3_INTEGRATION.md
│   └── ENHANCED_RAG_INTEGRATION_SUMMARY.md
│
├── Data Processing/
│   ├── PHASE2_ENHANCED_NOTES.md
│   ├── PHASE2_REFACTORING_SUMMARY.md
│   ├── OOM_SOLUTION.md
│   ├── OFFICIAL_SPLITS_FIX.md
│   └── DATAFLOW_SETUP.md
│
├── Training & Evaluation/
│   ├── TRAINING_GUIDE.md
│   ├── EVALUATION_GUIDE.md
│   └── TRAINING_PLAN_DAY1.md
│
└── Maintenance/
    ├── TROUBLESHOOTING.md
    ├── BUG_FIXES_SUMMARY.md
    ├── CODEBASE_AUDIT.md
    └── MAXIMIZING_CLAUDE_CODE.md
```

## Key Features by Documentation

### 3-Modal Architecture (354M parameters)
- Vision: BiomedCLIP-CXR
- Text: Clinical ModernBERT
- Clinical Features
- Documentation: `ENHANCED_RAG_ADAPTER.md`, `PHASE3_INTEGRATION.md`

### 4-Modal Architecture with CXR-PRO (327M parameters) - NEW!
- Vision: BiomedCLIP-CXR
- Clinical Text: ModernBERT
- **Radiology Text: BiomedBERT** (CXR-PRO prior-free impressions)
- Clinical Features
- Documentation: `CXR_PRO_INTEGRATION.md`, `CXR_PRO_IMPLEMENTATION_SUMMARY.md`

## Implementation Status

| Component | Status | Documentation |
|-----------|--------|---------------|
| **CXR-PRO Integration** | ✅ Complete | [CXR_PRO_INTEGRATION.md](CXR_PRO_INTEGRATION.md) |
| **4-Modal Architecture** | ✅ Complete | [CXR_PRO_INTEGRATION.md](CXR_PRO_INTEGRATION.md) |
| **3-Modal Architecture** | ✅ Complete | [ENHANCED_RAG_ADAPTER.md](ENHANCED_RAG_ADAPTER.md) |
| **Enhanced RAG Pipeline** | ✅ Complete | [ENHANCED_RAG_INTEGRATION_SUMMARY.md](ENHANCED_RAG_INTEGRATION_SUMMARY.md) |
| **Official Splits (377K)** | ✅ Complete | [OFFICIAL_SPLITS_FIX.md](OFFICIAL_SPLITS_FIX.md) |
| **Multi-Chunk Loading** | ✅ Complete | [BUG_FIXES_SUMMARY.md](BUG_FIXES_SUMMARY.md) |
| **NaN Bug Fixes** | ✅ Complete | [BUG_FIXES_SUMMARY.md](BUG_FIXES_SUMMARY.md) |
| **Training Pipeline** | ✅ Complete | [TRAINING_GUIDE.md](TRAINING_GUIDE.md) |
| **Quality Validation** | ✅ Complete | [CXR_PRO_INTEGRATION.md](CXR_PRO_INTEGRATION.md) |

## Recent Updates

**November 2025**:
- ✅ **CXR-PRO Integration**: 4-modal architecture with prior-free radiology reports
- ✅ **GILBERT Model**: BioBERT-based prior removal (HuggingFace: `rajpurkarlab/gilbert`)
- ✅ **Quality Validation**: Comprehensive validation for <0.05% prior references
- ✅ **Bug Fixes**: NaN issues, multi-chunk loading, early stopping improvements
- ✅ **Enhanced Documentation**: Complete guides for all features

## Contributing to Documentation

When adding new documentation:
1. Place `.md` files in the `docs/` directory
2. Update this README.md index
3. Add cross-references in related documents
4. Follow existing formatting conventions
5. Include code examples where applicable

## Questions or Issues?

- Check the [Troubleshooting Guide](TROUBLESHOOTING.md)
- Review the [Codebase Audit](CODEBASE_AUDIT.md)
- Open an issue on GitHub
- Consult the main [README](../README.md)

---

**Last Updated**: 2025-11-15
**Total Documentation Files**: 21
**Status**: Production Ready | CXR-PRO Integration Complete
