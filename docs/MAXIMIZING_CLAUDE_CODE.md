# Maximizing Claude Code for ML Research: Complete Strategy

## 🎯 Goal: Get Maximum Value from Your Claude Code Credits

This guide shows you how to leverage Claude Code for **every aspect** of your research project - from initial code generation to final paper writing.

---

## 📊 What Claude Code Can Do For You

### Code Generation (High Value)
- ✅ Complete training pipelines
- ✅ Data loading and preprocessing
- ✅ Model architecture implementations
- ✅ Evaluation and metrics
- ✅ Visualization scripts
- ✅ Experiment management

### Code Review & Debugging (High Value)
- ✅ Find bugs in existing code
- ✅ Optimize slow code
- ✅ Fix import errors
- ✅ Debug CUDA/GPU issues
- ✅ Memory leak detection

### Research Assistance (Medium-High Value)
- ✅ Literature review summaries
- ✅ Experiment design suggestions
- ✅ Hyperparameter recommendations
- ✅ Statistical analysis guidance
- ✅ Paper writing and LaTeX formatting

### Documentation (Medium Value)
- ✅ README files
- ✅ Code docstrings
- ✅ API documentation
- ✅ Tutorial notebooks

---

## 🚀 Phase-by-Phase Claude Code Usage Plan

## PHASE 1: Complete Code Infrastructure (Week 1)

**Goal**: Generate ALL the code you need before starting experiments

### Task 1.1: Training Infrastructure ⭐⭐⭐ (HIGHEST VALUE)

**Prompt to Claude Code**:
```
Generate a complete PyTorch Lightning training script for the Enhanced MDF-Net model with:

Requirements:
1. Load model from src/model/enhanced_mdfnet.py
2. Load loss from src/model/losses.py (CombinedLoss)
3. Multi-GPU training with DistributedDataParallel
4. Mixed precision training (torch.cuda.amp)
5. Gradient accumulation (accumulate 2 steps)
6. TensorBoard logging (log every 50 steps)
7. Model checkpointing:
   - Save best model by validation AUROC
   - Save every 5 epochs
   - Resume from checkpoint if exists
8. Early stopping (patience=10 epochs)
9. Learning rate scheduling:
   - Cosine annealing with warmup (3 epochs warmup)
   - Support discriminative learning rates (different LR for encoders/fusion/head)
10. Metrics:
    - Compute AUROC for all 14 abnormality classes
    - Compute mean AUROC
    - Log per-class and mean metrics
11. Command-line arguments for all hyperparameters
12. Support for vision-only, text-only, clinical-only, and full multi-modal modes
13. Configuration file support (YAML)

Architecture details:
- Model: EnhancedMDFNet (src/model/enhanced_mdfnet.py)
- Input: Dict with 'image', 'text_input_ids', 'text_attention_mask', 'clinical_features', 'labels'
- Output: 14-class multi-label predictions
- Loss: CombinedLoss (70% BCE + 30% Focal)

Save as: src/training/train_lightning.py

Also generate a simple config file template as: configs/base.yaml
```

**Expected Output**: ~500-800 line fully functional training script

**Value**: Saves you 8-12 hours of coding + 4-6 hours of debugging

---

### Task 1.2: Data Loading Pipeline ⭐⭐⭐

**Prompt to Claude Code**:
```
Create PyTorch Dataset and DataLoader classes for the Phase 3 output data.

Requirements:
1. Dataset class that loads .pt files from Phase 3 output:
   - Files are named: train_final.pt, val_final.pt, test_final.pt
   - Each .pt file contains a list of dicts with keys:
     'image' (torch.Tensor [3, 518, 518])
     'text_input_ids' (torch.Tensor)
     'text_attention_mask' (torch.Tensor)
     'clinical_features' (torch.Tensor [45])
     'labels' (Dict with 14 binary labels)

2. Support for local and GCS paths:
   - Local: /path/to/data/train_final.pt
   - GCS: gs://bucket/path/train_final.pt

3. Memory-efficient loading:
   - Don't load entire .pt file into RAM at once
   - Load records on-demand using torch.load with mmap

4. Data augmentation for images (training only):
   - Random horizontal flip (p=0.5)
   - Random rotation (±10 degrees)
   - Color jitter (brightness=0.2, contrast=0.2)
   - Use torchvision.transforms

5. Weighted sampling for class imbalance:
   - Oversample minority classes
   - Compute sample weights from label distribution

6. Distributed training support:
   - Use DistributedSampler for multi-GPU

7. Collate function to batch variable-length sequences:
   - Pad text sequences to max length in batch
   - Stack images and clinical features

8. DataLoader factory function with sensible defaults:
   - Batch size: 16
   - Num workers: 4
   - Pin memory: True
   - Persistent workers: True

Save as: src/training/dataloader.py

Also create a test script that loads and visualizes a few samples:
Save as: scripts/test_dataloader.py
```

**Expected Output**: ~300-400 line data loading module + test script

**Value**: Saves you 4-6 hours of coding + 2-3 hours of debugging data loading issues

---

### Task 1.3: Baseline Training Scripts ⭐⭐

**Prompt to Claude Code**:
```
Create 3 baseline training scripts for single-modality models:

1. Vision-Only Baseline (src/training/train_vision_only.py):
   - Use only BiomedCLIP-CXR encoder from EnhancedMDFNet
   - Add 2-layer classification head (512 -> 256 -> 14)
   - Same loss function (CombinedLoss)
   - Same training hyperparameters for fair comparison

2. Text-Only Baseline (src/training/train_text_only.py):
   - Use only Clinical ModernBERT encoder from EnhancedMDFNet
   - Add 2-layer classification head (768 -> 256 -> 14)
   - Same loss function
   - Same training hyperparameters

3. Clinical-Only Baseline (src/training/train_clinical_only.py):
   - Use only clinical features (45 features)
   - MLP: 45 -> 128 -> 64 -> 14
   - Same loss function
   - Same training hyperparameters

Requirements for all 3 scripts:
- Use PyTorch Lightning (reuse code from train_lightning.py)
- Log to TensorBoard with unique experiment names
- Save checkpoints to experiments/{vision,text,clinical}_only/
- Report AUROC for all 14 classes
- Support multi-GPU training
- Same hyperparameters as full model for fair comparison

Also create launcher script that trains all 3 baselines:
Save as: scripts/train_all_baselines.sh
```

**Expected Output**: 3 training scripts + launcher script

**Value**: Saves you 6-8 hours of coding

---

### Task 1.4: Configuration Management ⭐⭐

**Prompt to Claude Code**:
```
Create a YAML-based configuration system for experiments.

Generate these config files:

1. configs/base.yaml - Default hyperparameters:
   - model: {freeze_encoders: true, dropout_fusion: 0.3, dropout_head1: 0.3, dropout_head2: 0.2}
   - training: {batch_size: 16, max_epochs: 15, gradient_accumulation: 2, lr: 1e-3}
   - optimizer: {name: AdamW, weight_decay: 1e-4}
   - scheduler: {name: cosine, warmup_epochs: 3}
   - data: {num_workers: 4, augmentation: true}

2. configs/vision_only.yaml - Vision baseline (inherits from base)
3. configs/text_only.yaml - Text baseline
4. configs/clinical_only.yaml - Clinical baseline
5. configs/full_model_stage1.yaml - Full model stage 1 (frozen encoders)
6. configs/full_model_stage2.yaml - Full model stage 2 (end-to-end fine-tuning)

Also create a Python class to load and merge configs:
- Support config inheritance (child configs extend parent)
- Support command-line overrides
- Validate required fields
- Pretty-print final config

Save as: src/training/config.py

Example usage:
```python
from src.training.config import load_config
cfg = load_config('configs/full_model_stage2.yaml', overrides={'training.lr': 1e-4})
```
```

**Expected Output**: 6 YAML configs + config loader class

**Value**: Saves you 2-3 hours + makes experiment management much easier

---

### Task 1.5: Metrics and Evaluation ⭐⭐⭐

**Prompt to Claude Code**:
```
Enhance the existing src/evaluation/metrics.py and correlation.py with additional functionality:

1. Add calibration analysis:
   - Reliability diagrams for each class
   - Expected Calibration Error (ECE) - already implemented but add visualization
   - Plot predicted probability vs actual frequency

2. Add per-class analysis utilities:
   - Generate per-class performance reports (AUROC, AUPRC, F1, etc.)
   - Identify best/worst performing classes
   - Analyze error patterns by class

3. Add threshold optimization:
   - Find optimal threshold for each class (maximize F1)
   - Find optimal threshold for each class (maximize Youden's J)
   - Support class-specific vs global threshold

4. Add subgroup analysis:
   - Performance by age group (<40, 40-60, 60-80, >80)
   - Performance by gender
   - Performance by acuity level
   - Performance by time of day (if available)

5. Add comprehensive evaluation script:
   - Load model checkpoint
   - Run inference on test set
   - Compute all metrics
   - Generate all visualizations
   - Save results to structured directory

Save enhanced metrics to: src/evaluation/metrics.py (update existing)
Save evaluation script as: src/evaluation/evaluate_model.py
```

**Expected Output**: Enhanced metrics module + comprehensive evaluation script

**Value**: Saves you 6-10 hours of analysis code writing

---

### Task 1.6: Visualization Suite ⭐⭐⭐

**Prompt to Claude Code**:
```
Create the missing src/evaluation/visualizations.py module with comprehensive plotting functions:

Classes to implement:

1. ConfusionMatrixPlotter:
   - Multi-label confusion matrix (one per class)
   - Combined confusion matrix (averaging across classes)
   - Heatmap style with annotations
   - Support for saving to file

2. ROCCurvePlotter:
   - ROC curves for all 14 classes on same plot
   - Individual ROC curve per class
   - Show AUC score on each curve
   - Add 95% confidence intervals (bootstrap)
   - Add random baseline (diagonal line)
   - Publication-quality formatting

3. PRCurvePlotter:
   - Precision-Recall curves for all 14 classes
   - Individual PR curve per class
   - Show AUPRC score on each curve
   - Add confidence intervals
   - Show baseline (prevalence line)

4. CalibrationPlotter:
   - Reliability diagrams for each class
   - Perfect calibration line
   - Show ECE score

5. AttentionVisualizer:
   - Overlay cross-modal attention on chest X-rays
   - Heatmap showing which image regions attended to
   - Support for saving attention maps

6. FeatureImportancePlotter:
   - Bar charts for clinical feature importance (from SHAP)
   - Horizontal bar chart sorted by importance
   - Different colors for positive/negative impact

7. TrainingCurvePlotter:
   - Plot training/validation loss curves
   - Plot training/validation AUROC curves
   - Support for multiple experiments on same plot
   - Load data from TensorBoard logs

All plotters should:
- Support matplotlib and seaborn
- Publication-quality (300 DPI, vector formats)
- Consistent styling across all plots
- Support for saving to PDF, PNG, SVG
- Clear legends and labels

Save as: src/evaluation/visualizations.py

Also create a demo script that generates all visualizations:
Save as: scripts/generate_all_plots.py
```

**Expected Output**: ~600-800 line visualization module + demo script

**Value**: Saves you 10-15 hours of matplotlib/seaborn wrestling

---

### Task 1.7: Experiment Management ⭐⭐

**Prompt to Claude Code**:
```
Create an experiment tracking and comparison system:

1. Experiment logger that tracks:
   - Hyperparameters used
   - Training time
   - Best validation AUROC
   - Final test AUROC
   - Model checkpoint path
   - Git commit hash
   - Date/time
   - Hardware used (GPU type, count)

2. Experiment database:
   - SQLite database to store all experiments
   - Schema: experiments table with fields above
   - CRUD operations

3. Experiment comparison tool:
   - Load all experiments from database
   - Generate comparison table (sorted by test AUROC)
   - Statistical significance testing (DeLong test)
   - Highlight best experiment
   - Export to CSV and LaTeX table

4. Experiment dashboard:
   - Command-line tool to view experiment status
   - Show running experiments
   - Show completed experiments with metrics
   - Pretty-printed table output

5. Integration with training script:
   - Automatically log experiments after training
   - No manual logging needed

Save as:
- src/training/experiment_tracker.py (logger and database)
- scripts/compare_experiments.py (comparison tool)
- scripts/experiment_dashboard.py (CLI dashboard)
```

**Expected Output**: Experiment tracking system

**Value**: Saves you 4-6 hours + makes managing dozens of experiments much easier

---

### Task 1.8: Debugging and Profiling Tools ⭐⭐

**Prompt to Claude Code**:
```
Create debugging utilities to help diagnose training issues:

1. Gradient flow analyzer:
   - Hook into model to track gradient statistics
   - Detect vanishing gradients (grad norm < 1e-8)
   - Detect exploding gradients (grad norm > 100)
   - Plot gradient norms per layer
   - Save plot showing which layers have gradient issues

2. Activation statistics:
   - Track activation means and stds per layer
   - Detect dead neurons (activation always near 0)
   - Plot activation distributions

3. Learning rate finder:
   - Implement LR range test (Leslie Smith method)
   - Train for 1 epoch with exponentially increasing LR
   - Plot loss vs learning rate
   - Suggest optimal LR

4. Batch inspector:
   - Visualize a batch of inputs
   - Show images, text snippets, clinical features, labels
   - Verify data augmentation is working
   - Check for data loading bugs

5. Memory profiler:
   - Track GPU memory usage over time
   - Identify memory leaks
   - Suggest batch size adjustments

6. Model summary tool:
   - Print detailed model architecture
   - Show parameter counts per layer
   - Show total parameters and memory footprint
   - Identify which components use most parameters

Save as: src/training/debug_utils.py

Also create test scripts:
- scripts/debug_gradients.py (run gradient analysis)
- scripts/find_lr.py (run LR finder)
- scripts/inspect_batch.py (visualize batch)
```

**Expected Output**: Comprehensive debugging toolkit

**Value**: Saves you 8-12 hours of debugging time throughout project

---

## PHASE 2: Dataset Analysis and Preparation (Week 1-2)

### Task 2.1: Dataset Statistics and Analysis ⭐⭐⭐

**Prompt to Claude Code**:
```
Create a comprehensive dataset analysis script:

Analyze the Phase 3 output data and generate detailed statistics:

1. Class distribution:
   - Count for each of 14 abnormality classes
   - Prevalence (%) for each class
   - Identify imbalanced classes
   - Recommend positive weights for WeightedBCELoss

2. Co-occurrence analysis:
   - Which abnormalities appear together frequently
   - 14x14 co-occurrence matrix (heatmap)
   - Identify common multi-label combinations

3. Clinical feature statistics:
   - Mean, std, min, max for each of 45 features
   - Missing value rates
   - Correlation matrix (features vs features)
   - Identify highly correlated features (>0.9)

4. Text statistics:
   - Text length distribution (tokens)
   - Average text length
   - Max text length
   - Verify all texts fit in 8192 token limit

5. Image statistics:
   - Image size distribution
   - Brightness distribution
   - Contrast distribution
   - Check for corrupted images

6. Split statistics:
   - Train/val/test sizes
   - Verify splits are balanced (similar class distributions)
   - Check for data leakage (no patient overlap)

7. Generate comprehensive report:
   - Markdown format with tables and figures
   - Save all plots to figures/dataset_analysis/
   - Save statistics to data_statistics.csv

Load data from: processed/phase3_output/{train,val,test}_final.pt
Save report to: reports/dataset_analysis.md

Also recommend:
- Optimal class weights for loss function
- Potential data quality issues
- Suggested data preprocessing steps
```

**Expected Output**: Dataset analysis script + comprehensive report

**Value**: Saves you 6-8 hours of exploratory data analysis

---

### Task 2.2: Create Debug/Dev Dataset ⭐⭐

**Prompt to Claude Code**:
```
Create a tiny debug dataset for fast iteration:

Requirements:
1. Sample 100 records from Phase 3 output:
   - 50 from train
   - 25 from val
   - 25 from test

2. Ensure diverse sampling:
   - Include samples from each abnormality class
   - Include positive and negative examples
   - Preserve class distribution roughly

3. Save to data/debug/:
   - data/debug/train_final.pt
   - data/debug/val_final.pt
   - data/debug/test_final.pt

4. Verify the debug dataset:
   - Can train 1 epoch in < 2 minutes on CPU
   - DataLoader works correctly
   - Model forward pass works
   - Loss computation works
   - Metrics computation works

This allows fast testing before running expensive GPU training.

Save as: scripts/create_debug_dataset.py

Also create a quick test script:
Save as: scripts/test_full_pipeline_debug.py
This should run 3 epochs of training on debug dataset and verify everything works.
```

**Expected Output**: Debug dataset creator + pipeline test

**Value**: Saves you 10-20 hours of debugging on expensive GPUs

---

## PHASE 3: Training and Experimentation (Week 2-4)

### Task 3.1: Training Monitoring ⭐⭐

**Prompt to Claude Code**:
```
Create real-time training monitoring tools:

1. Training monitor that:
   - Tails the training log file in real-time
   - Parses metrics (epoch, loss, AUROC, etc.)
   - Displays live progress with ETA
   - Plots metrics in terminal using plotext library
   - Shows GPU utilization (nvidia-smi)
   - Shows memory usage
   - Detects anomalies (NaN loss, gradient explosion)
   - Sends alerts (print to console, could add email/Slack)

2. Alert conditions:
   - Training finishes
   - Validation AUROC improves
   - Error/crash detected
   - NaN loss detected
   - GPU memory issues
   - Training stuck (no progress for 30 min)

3. Summary stats:
   - Training time elapsed
   - Estimated time remaining
   - Average epoch time
   - Best validation AUROC so far

Save as: scripts/monitor_training.py

Usage:
```bash
# In one terminal, start training:
python src/training/train_lightning.py --config configs/base.yaml

# In another terminal, monitor:
python scripts/monitor_training.py --log-file logs/train.log
```
```

**Expected Output**: Real-time training monitor

**Value**: Saves you hours of manually checking training progress

---

### Task 3.2: Hyperparameter Optimization ⭐⭐⭐

**Prompt to Claude Code**:
```
Create hyperparameter optimization using Optuna:

Search space:
1. Learning rates:
   - Stage 1 LR: [1e-4, 1e-3, 5e-3]
   - Stage 2 encoder LR: [1e-6, 1e-5, 5e-5]
   - Stage 2 fusion/head LR: [1e-5, 1e-4, 5e-4]

2. Dropout rates:
   - Fusion dropout: [0.1, 0.2, 0.3, 0.4]
   - Head dropout 1: [0.2, 0.3, 0.4]
   - Head dropout 2: [0.1, 0.2, 0.3]

3. Loss weights:
   - BCE weight: [0.5, 0.6, 0.7, 0.8]
   - Focal weight: [0.2, 0.3, 0.4, 0.5]
   (must sum to 1.0)

4. Training:
   - Batch size: [8, 16, 32]
   - Gradient accumulation: [1, 2, 4]
   - Weight decay: [1e-5, 1e-4, 1e-3]

Requirements:
1. Objective: Maximize mean validation AUROC
2. Use Optuna's TPE sampler (Bayesian optimization)
3. Enable pruning to stop bad trials early (saves GPU time)
4. Run 30 trials maximum
5. Support parallel trials (multi-GPU)
6. Save all trials to SQLite database
7. Generate visualization:
   - Optimization history (AUROC vs trial number)
   - Parameter importance plot
   - Parallel coordinate plot
8. Export best hyperparameters to YAML config

Integration:
- Reuse existing training code (train_lightning.py)
- Each trial trains for 10 epochs (enough to see if config is good)
- Use validation AUROC at epoch 10 as objective

Save as: src/training/optimize_hyperparameters.py

Also create visualization script:
Save as: scripts/visualize_optuna_study.py
```

**Expected Output**: Hyperparameter optimization framework

**Value**: Saves you 20-40 hours of manual hyperparameter tuning

---

### Task 3.3: Model Ensemble ⭐

**Prompt to Claude Code**:
```
Create model ensemble framework:

1. Train multiple models with different:
   - Random seeds
   - Hyperparameters
   - Architectures (vision-only, text-only, full model)

2. Ensemble methods:
   - Average predictions
   - Weighted average (weight by validation AUROC)
   - Stacking (train meta-model on predictions)

3. Ensemble evaluation:
   - Compare single model vs ensemble
   - Show improvement from ensembling
   - Analyze when ensemble helps most

4. Save ensemble model:
   - Single .pt file containing all models
   - Easy to load and use for inference

Save as:
- src/training/ensemble.py (ensemble class)
- scripts/create_ensemble.py (combine trained models)
- scripts/evaluate_ensemble.py (test ensemble)
```

**Expected Output**: Ensemble framework

**Value**: Could improve final AUROC by 1-2%, saves you 4-6 hours

---

## PHASE 4: Analysis and Results (Week 5-6)

### Task 4.1: Comprehensive Results Analysis ⭐⭐⭐

**Prompt to Claude Code**:
```
Create publication-ready results analysis:

Generate all analysis for your paper:

1. Performance comparison tables:
   - Baseline models vs full model
   - Mean AUROC across all classes
   - Per-class AUROC for all models
   - Statistical significance (DeLong test, p-values)
   - Export to LaTeX table format

2. ROC curves:
   - All 14 classes on one plot (full model)
   - Separate plots for each class comparing all models
   - 95% confidence intervals (bootstrap with 1000 iterations)
   - Publication-quality formatting

3. Precision-Recall curves:
   - Same as ROC curves but PR curves
   - Show AUPRC scores

4. Confusion matrices:
   - One per class (14 total)
   - Combined confusion matrix
   - Show false positive/false negative analysis

5. Attention visualization:
   - Select 20 interesting examples
   - Overlay cross-modal attention on X-rays
   - Show which image regions model focused on
   - Qualitative analysis of attention patterns

6. SHAP feature importance:
   - Clinical feature importance for each abnormality
   - Top 10 features per abnormality
   - Visualize with bar charts
   - Analyze which features matter most

7. Error analysis:
   - Identify samples with worst predictions
   - Analyze common failure modes
   - Categorize errors (false positives vs false negatives)
   - Find patterns in errors

8. Subgroup performance:
   - Performance by age group
   - Performance by gender
   - Performance by acuity level
   - Identify disparities

9. Ablation studies:
   - Remove each modality and test performance
   - Quantify contribution of each modality
   - Test importance of cross-modal attention

All figures should be:
- Publication-quality (300 DPI, PDF format)
- Consistent styling
- Clear labels and legends
- Saved to paper_figures/

All tables should be:
- LaTeX format (for direct inclusion in paper)
- Properly formatted with booktabs
- Saved to paper_tables/

Save as: src/analysis/generate_paper_results.py

This should be a comprehensive script that generates ALL figures and tables for your paper.
```

**Expected Output**: Complete results analysis pipeline

**Value**: Saves you 15-25 hours of analysis and figure generation

---

### Task 4.2: Jupyter Notebooks for Interactive Analysis ⭐⭐

**Prompt to Claude Code**:
```
Create interactive Jupyter notebooks for exploratory analysis:

1. notebooks/01_dataset_exploration.ipynb:
   - Load and visualize dataset
   - Explore class distributions
   - Visualize sample images and texts
   - Interactive plots with plotly

2. notebooks/02_training_analysis.ipynb:
   - Load TensorBoard logs
   - Plot training curves
   - Compare experiments
   - Interactive experiment comparison

3. notebooks/03_model_inspection.ipynb:
   - Load trained model
   - Inspect model architecture
   - Visualize attention weights
   - Run inference on sample inputs
   - Interactive attention visualization

4. notebooks/04_error_analysis.ipynb:
   - Load predictions on test set
   - Find worst predictions
   - Visualize failure cases
   - Interactive error exploration

5. notebooks/05_results_for_paper.ipynb:
   - Generate all paper figures
   - Generate all paper tables
   - Preview results before finalizing
   - Export to paper_figures/ and paper_tables/

Each notebook should:
- Be well-documented with markdown cells
- Include clear explanations
- Be executable from top to bottom
- Save outputs to appropriate directories
```

**Expected Output**: 5 comprehensive Jupyter notebooks

**Value**: Saves you 10-15 hours of analysis + provides interactive exploration

---

### Task 4.3: Automated Report Generation ⭐⭐

**Prompt to Claude Code**:
```
Create automatic experiment report generator:

Generate comprehensive markdown reports for each experiment:

1. Experiment summary:
   - Experiment name and date
   - Hyperparameters used
   - Hardware used
   - Training time
   - Git commit hash

2. Results:
   - Best validation AUROC
   - Test AUROC (per-class and mean)
   - Comparison to baseline
   - Confusion matrix summary

3. Training curves:
   - Embed training/validation loss plots
   - Embed AUROC curves

4. Insights:
   - What worked well
   - What didn't work
   - Suggested next steps

5. Attachments:
   - Link to checkpoint file
   - Link to TensorBoard logs
   - Link to full results

Auto-generate reports after each training run.

Save report to: reports/experiments/YYYY-MM-DD_experiment_name.md

Save as: src/training/report_generator.py

Integrate with training script to auto-generate after training completes.
```

**Expected Output**: Automated report generator

**Value**: Saves you 5-10 hours of documentation + ensures experiments are well-documented

---

## PHASE 5: Paper Writing Assistance (Week 6-7)

### Task 5.1: LaTeX Paper Template ⭐⭐

**Prompt to Claude Code**:
```
Generate LaTeX paper template for medical imaging conference (e.g., MICCAI, MIDL):

Structure:
1. Abstract (250 words)
2. Introduction
   - Problem statement
   - Motivation for multi-modal approach
   - Contributions
3. Related Work
   - Single-modal baselines
   - Multi-modal fusion methods
   - MIMIC-IV-CXR studies
4. Methods
   - Dataset description
   - Model architecture (Enhanced MDF-Net)
   - Training strategy
   - Evaluation metrics
5. Results
   - Baseline comparison
   - Full model performance
   - Ablation studies
   - Attention visualization
6. Discussion
   - Analysis of results
   - Limitations
   - Future work
7. Conclusion

Include:
- Proper bibliography style (BibTeX)
- Figure and table environments
- Algorithm environment for model architecture
- Proper citations (\cite{})
- Conference formatting

Save as: paper/main.tex

Also create:
- paper/references.bib (with key references pre-filled)
- paper/Makefile (for compiling)
- paper/figures/ (directory for figures)
```

**Expected Output**: Complete LaTeX paper template

**Value**: Saves you 2-4 hours of LaTeX setup

---

### Task 5.2: Paper Writing Prompts ⭐⭐⭐

After generating results, use Claude Code iteratively for paper writing:

**Prompt for Abstract**:
```
Based on my experimental results, help me write the abstract:

Results:
- Vision-only baseline: Mean AUROC 0.823
- Text-only baseline: Mean AUROC 0.791
- Clinical-only baseline: Mean AUROC 0.742
- Full multi-modal model: Mean AUROC 0.867
- Best improvement on rare classes (Pneumothorax: +8.2%, Lung Lesion: +11.4%)
- Cross-modal attention successfully identified relevant image regions

Write a 250-word abstract for a medical imaging conference that:
1. States the problem (chest X-ray abnormality detection in ED)
2. Describes approach (multi-modal fusion of images, notes, clinical data)
3. Highlights key results
4. Emphasizes clinical significance
```

**Prompt for Methods**:
```
Help me write the Methods section describing Enhanced MDF-Net:

Components:
- Vision encoder: BiomedCLIP-CXR (87M params)
- Text encoder: Clinical ModernBERT (149M params)
- Clinical encoder: 3-layer MLP
- Cross-modal attention fusion
- Classification head

Write clear, technical description suitable for ML/medical imaging audience.
Include architecture diagram description.
```

**Prompt for Results**:
```
Help me write the Results section:

Tables to include:
1. Table 1: Baseline comparison (vision, text, clinical, full model)
2. Table 2: Per-class AUROC for all 14 abnormalities
3. Table 3: Ablation study (removing each modality)

Figures to include:
1. Figure 1: ROC curves for all classes
2. Figure 2: Attention visualization examples
3. Figure 3: SHAP feature importance

Write clear narrative connecting tables and figures.
Highlight key findings.
```

**Value**: Saves you 10-20 hours of paper writing + improves quality

---

## PHASE 6: Code Quality and Documentation (Ongoing)

### Task 6.1: Comprehensive Documentation ⭐⭐

**Prompt to Claude Code**:
```
Add comprehensive docstrings to all code:

Go through these files and add detailed docstrings:
1. src/model/enhanced_mdfnet.py
2. src/model/losses.py
3. src/training/train_lightning.py
4. src/training/dataloader.py
5. src/evaluation/metrics.py
6. src/evaluation/correlation.py
7. src/evaluation/visualizations.py

For each:
- Add module-level docstring
- Add class docstrings with attributes
- Add method docstrings with Args, Returns, Raises
- Use Google style docstrings
- Include examples where helpful

Also generate API documentation using Sphinx:
- Create docs/conf.py
- Create docs/index.rst
- Generate HTML documentation
```

**Expected Output**: Fully documented code + generated docs

**Value**: Saves you 6-10 hours of documentation writing

---

### Task 6.2: Code Review and Optimization ⭐⭐

**Prompt to Claude Code**:
```
Review and optimize the training code:

Files to review:
- src/training/train_lightning.py
- src/training/dataloader.py

Check for:
1. Performance bottlenecks
2. Memory inefficiencies
3. Potential bugs
4. Code style issues
5. Missing error handling
6. Opportunities for parallelization

Suggest optimizations for:
- Faster data loading
- Lower memory usage
- Better GPU utilization
- Cleaner code structure

Provide specific code changes with explanations.
```

**Expected Output**: Code review report + optimization suggestions

**Value**: Could speed up training 10-30%, saves debugging time

---

### Task 6.3: Testing Suite ⭐

**Prompt to Claude Code**:
```
Create comprehensive pytest test suite:

Tests to create:

1. tests/test_model.py:
   - Test EnhancedMDFNet forward pass
   - Test different modality combinations
   - Test output shapes
   - Test loss computation
   - Test gradient flow

2. tests/test_dataloader.py:
   - Test dataset loading
   - Test data augmentation
   - Test batching
   - Test distributed sampler

3. tests/test_metrics.py:
   - Test AUROC computation
   - Test confusion matrix
   - Test calibration metrics
   - Test statistical tests

4. tests/test_training.py:
   - Test training loop (1 epoch on debug data)
   - Test checkpointing
   - Test early stopping
   - Test learning rate scheduling

5. tests/test_config.py:
   - Test config loading
   - Test config merging
   - Test config validation

All tests should:
- Use pytest fixtures
- Be fast (< 30 seconds total)
- Use debug/mock data
- Have clear assertions
- Cover edge cases

Also create:
- pytest.ini configuration
- conftest.py with shared fixtures
- scripts/run_tests.sh
```

**Expected Output**: Complete test suite

**Value**: Saves you 8-12 hours of test writing + catches bugs early

---

## 🎯 Priority Ranking: What to Do First

### Critical Path (Do These First):

1. **Task 1.1: Training Infrastructure** ⭐⭐⭐ - Blocks everything
2. **Task 1.2: Data Loading** ⭐⭐⭐ - Blocks training
3. **Task 2.2: Debug Dataset** ⭐⭐ - Enables fast iteration
4. **Task 1.8: Debugging Tools** ⭐⭐ - Saves debugging time
5. **Task 2.1: Dataset Analysis** ⭐⭐⭐ - Informs hyperparameter choices

### High Value (Do Next):

6. **Task 1.6: Visualization Suite** ⭐⭐⭐ - Needed for evaluation
7. **Task 1.5: Enhanced Metrics** ⭐⭐⭐ - Needed for evaluation
8. **Task 1.3: Baseline Scripts** ⭐⭐ - For fair comparison
9. **Task 3.2: Hyperparameter Optimization** ⭐⭐⭐ - Improves performance
10. **Task 4.1: Results Analysis** ⭐⭐⭐ - For paper

### Medium Value (Nice to Have):

11. **Task 1.4: Config Management** ⭐⭐ - Makes experiments easier
12. **Task 1.7: Experiment Tracking** ⭐⭐ - Organizes experiments
13. **Task 3.1: Training Monitoring** ⭐⭐ - Quality of life
14. **Task 4.2: Jupyter Notebooks** ⭐⭐ - Interactive exploration
15. **Task 5.1: LaTeX Template** ⭐⭐ - For paper writing

### Lower Priority (If Time):

16. **Task 3.3: Model Ensemble** ⭐ - Marginal improvement
17. **Task 4.3: Report Generator** ⭐⭐ - Nice documentation
18. **Task 6.1: Documentation** ⭐⭐ - Good practice
19. **Task 6.2: Code Review** ⭐⭐ - Optimization
20. **Task 6.3: Testing Suite** ⭐ - Quality assurance

---

## 📝 How to Use This Guide

### Week-by-Week Plan:

**Week 1: Infrastructure**
- Complete Tasks 1.1, 1.2, 2.2, 1.8, 2.1 (critical path)
- Test everything locally with debug dataset
- Fix any bugs before GPU training

**Week 2: Baseline Experiments**
- Complete Tasks 1.3, 1.4, 1.7 (baseline infrastructure)
- Train all 3 baselines
- Analyze baseline results

**Week 3-4: Full Model Training**
- Train full model Stage 1 and Stage 2
- Complete Task 3.1 (monitoring)
- Monitor training progress

**Week 5: Hyperparameter Optimization**
- Complete Task 3.2 (Optuna optimization)
- Run hyperparameter search
- Train final model with best hyperparameters

**Week 6: Evaluation**
- Complete Tasks 1.5, 1.6, 4.1 (evaluation and analysis)
- Generate all results
- Create all figures and tables

**Week 7: Paper Writing**
- Complete Tasks 5.1, 5.2 (paper template and writing)
- Write paper using generated results
- Iterate with Claude Code on paper sections

---

## 💬 Example Claude Code Conversations

### Debugging a Training Issue:

**You**: "My training loss is NaN after 3 epochs. Here's the log: [paste log]. Debug this."

**Claude Code**:
1. Analyzes log
2. Identifies likely causes
3. Suggests fixes with code changes
4. Updates training script

### Improving Model Performance:

**You**: "My validation AUROC is stuck at 0.82. Suggest improvements based on my current config: [paste config]"

**Claude Code**:
1. Analyzes current hyperparameters
2. Suggests specific changes
3. Explains why each change might help
4. Generates updated config file

### Generating a Figure:

**You**: "Create a ROC curve plot for my test results. Data is in results/test_predictions.csv with columns: true_label_0 to true_label_13, pred_prob_0 to pred_prob_13. Show all 14 classes on one plot with legend."

**Claude Code**:
1. Writes complete plotting script
2. Handles data loading
3. Creates publication-quality figure
4. Saves to specified path

---

## 🚀 Getting Started Right Now

**Your First 5 Prompts to Claude Code:**

1. "Let's fix the critical audit issues from the codebase audit. Start with adding the missing dependencies to requirements.txt."

2. "Generate the training infrastructure (Task 1.1): Complete PyTorch Lightning training script for Enhanced MDF-Net with all the features listed in the guide."

3. "Generate the data loading pipeline (Task 1.2): Dataset and DataLoader classes for Phase 3 output."

4. "Create a debug dataset (Task 2.2): Sample 100 records for fast testing."

5. "Generate debugging tools (Task 1.8): Gradient flow analyzer, LR finder, and batch inspector."

After these 5 prompts, you'll have:
- Fixed codebase ready to run
- Complete training infrastructure
- Data loading working
- Debug dataset for fast iteration
- Tools to diagnose any issues

**Estimated time**: 1-2 hours to generate all this code
**Manual coding time saved**: 40-60 hours

---

## 🎓 Maximizing Your Claude Code Credits

**Best Practices:**

1. **Be Specific**: More detailed prompts = better code
   - ❌ "Create a training script"
   - ✅ "Create a PyTorch Lightning training script with DDP, mixed precision, TensorBoard logging, checkpointing, early stopping, and cosine LR schedule"

2. **Iterate**: Don't expect perfect code first try
   - Generate → Test → Ask for fixes → Test again

3. **Provide Context**: Share relevant code/configs
   - "Here's my model architecture [paste code]. Generate a training script that uses it."

4. **Ask for Explanations**: Learn while coding
   - "Explain why you chose AdamW over Adam"
   - "Why is the learning rate 1e-3 for stage 1?"

5. **Use for Analysis**: Not just code generation
   - "Analyze these experimental results and suggest next experiments"
   - "Review this code for performance bottlenecks"

6. **Batch Related Tasks**:
   - "Generate all 3 baseline scripts at once"
   - More efficient than one-by-one

---

## 📊 Expected Outcomes

By following this guide and using Claude Code extensively, you should achieve:

**Time Savings:**
- **80-150 hours** of manual coding saved
- **20-40 hours** of debugging saved
- **15-30 hours** of analysis/visualization saved
- **Total: 115-220 hours saved**

**Code Quality:**
- Well-structured, documented code
- Best practices (DDP, mixed precision, etc.)
- Comprehensive test coverage
- Publication-ready figures

**Research Outcomes:**
- Complete baseline experiments
- Optimized full model
- Comprehensive evaluation
- Publication-ready results
- Conference paper draft

**Research Velocity:**
- Iterate 10x faster than manual coding
- Test ideas quickly with debug dataset
- Focus on research insights, not coding details
- More time for analysis and paper writing

---

## 🎯 Summary

**Key Message**: Use Claude Code for EVERYTHING code-related.

**Don't write code manually when Claude Code can:**
- Generate it in minutes
- Make it better than you would manually
- Document it properly
- Test it comprehensively

**Your job**:
- Design experiments
- Interpret results
- Write the paper
- Make research decisions

**Claude Code's job**:
- Write all the code
- Debug all the issues
- Generate all the figures
- Help write the paper

**Result**: 3-6 month project done in 6-8 weeks with higher quality code and results.

---

Start with Task 1.1 right now! 🚀
