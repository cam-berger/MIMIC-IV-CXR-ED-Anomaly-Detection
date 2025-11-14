# Evaluation Guide: Testing Enhanced MDF-Net

## Overview

This guide provides comprehensive instructions for evaluating the Enhanced MDF-Net model. It includes detailed methods for computing metrics, generating visualizations, and performing in-depth analysis of model performance.

## Table of Contents

1. [Evaluation Metrics](#evaluation-metrics)
2. [Confusion Matrices](#confusion-matrices)
3. [ROC & Precision-Recall Curves](#roc--precision-recall-curves)
4. [Correlation Analysis](#correlation-analysis)
5. [Attention Visualization](#attention-visualization)
6. [SHAP Feature Importance](#shap-feature-importance)
7. [Error Analysis](#error-analysis)
8. [Subgroup Analysis](#subgroup-analysis)
9. [Modality Ablation Studies](#modality-ablation-studies)
10. [Statistical Significance Testing](#statistical-significance-testing)

## Evaluation Metrics

### Primary Metrics

For each of the 14 abnormality classes, we compute:

#### 1. **AUROC** (Area Under ROC Curve)
- **Primary metric** for model selection
- Measures discrimination ability (separating positive from negative)
- Range: [0.5, 1.0] (0.5 = random, 1.0 = perfect)
- **Target**: Mean AUROC ≥ 0.860 (beating MDF-NET paper: 0.856)

#### 2. **AUPRC** (Area Under Precision-Recall Curve)
- Important for imbalanced classes
- More informative than AUROC when positive class is rare
- Focus on rare abnormalities: Fracture, Lung Lesion, Pleural Other

#### 3. **F1-Score**
- Harmonic mean of precision and recall
- Balances false positives and false negatives
- Computed at optimal threshold (maximizing F1)

#### 4. **Sensitivity** (Recall, True Positive Rate)
- Critical for medical applications
- Percentage of actual positives correctly identified
- High sensitivity = few missed cases (important for safety)

#### 5. **Specificity** (True Negative Rate)
- Percentage of actual negatives correctly identified
- High specificity = few false alarms

### Secondary Metrics

#### 6. **Precision** (Positive Predictive Value)
- Percentage of positive predictions that are correct
- Important for clinical workflow (avoiding unnecessary follow-ups)

#### 7. **NPV** (Negative Predictive Value)
- Percentage of negative predictions that are correct
- Reassurance that "no finding" is truly negative

### Calibration Metrics

#### 8. **Expected Calibration Error (ECE)**
- Measures prediction reliability
- Are predicted probabilities accurate?
- Lower is better (ECE < 0.05 = well-calibrated)

#### 9. **Brier Score**
- Mean squared difference between predictions and outcomes
- Combines discrimination and calibration
- Lower is better

### Evaluation Script

```bash
# Evaluate all models and generate comprehensive report
python src/evaluation/evaluate.py \
  --checkpoint experiments/full_model_stage2/best_model.pth \
  --data-path processed/phase1_output \
  --split test \
  --output-dir results/evaluation \
  --gpu-id 0
```

**Output:**
```
results/evaluation/
├── metrics_summary.csv         # All metrics per abnormality
├── confusion_matrices/         # Multi-label confusion matrices
├── roc_curves/                 # ROC curves per abnormality
├── pr_curves/                  # Precision-recall curves
├── calibration_plots/          # Reliability diagrams
├── correlation_analysis/       # Clinical features ↔ abnormalities
├── attention_maps/             # Visualized attention weights
├── shap_analysis/              # Feature importance
├── error_analysis/             # Failure mode analysis
└── report.html                 # Interactive HTML report
```

## Confusion Matrices

### Multi-Label Confusion Matrices

For multi-label classification, we generate a 2×2 confusion matrix for **each abnormality**:

```
                Predicted Negative  Predicted Positive
Actual Negative        TN                  FP
Actual Positive        FN                  TP
```

**Script:**
```bash
python src/evaluation/plot_confusion_matrices.py \
  --checkpoint experiments/full_model_stage2/best_model.pth \
  --data-path processed/phase1_output \
  --output-dir results/confusion_matrices
```

**Generated Visualizations:**

1. **Individual Confusion Matrices** (14 separate plots)
   - One heatmap per abnormality
   - Shows TP, TN, FP, FN counts
   - Annotated with percentages

2. **Combined Grid View**
   - All 14 confusion matrices in single figure (4×4 grid)
   - Quick overview of all classes

3. **Normalized Confusion Matrices**
   - Row-normalized (by actual class)
   - Shows sensitivity and specificity directly

### Example Output

```python
# Confusion Matrix for Pneumonia
#
#                 Predicted
#                 Neg    Pos
# Actual  Neg    1245    83   (93.7% specificity)
#         Pos      47   125   (72.7% sensitivity)
#
# Accuracy: 91.3%
# F1-Score: 0.658
```

## ROC & Precision-Recall Curves

### ROC Curves (Receiver Operating Characteristic)

**Script:**
```bash
python src/evaluation/plot_roc_curves.py \
  --checkpoint experiments/full_model_stage2/best_model.pth \
  --data-path processed/phase1_output \
  --output-dir results/roc_curves \
  --plot-ci  # Include 95% confidence intervals
```

**Generated Plots:**

1. **Individual ROC Curves** (14 plots)
   - One curve per abnormality
   - AUROC value displayed
   - Optimal threshold marked
   - 95% CI shaded region (via bootstrapping)

2. **Combined ROC Plot**
   - All 14 curves on same axes
   - Color-coded by abnormality
   - Mean AUROC shown

3. **Grouped by Prevalence**
   - Common (prevalence > 10%): Atelectasis, Cardiomegaly, Edema, etc.
   - Rare (prevalence < 5%): Fracture, Lung Lesion, Pleural Other

### Precision-Recall Curves

Critical for imbalanced classes:

```bash
python src/evaluation/plot_pr_curves.py \
  --checkpoint experiments/full_model_stage2/best_model.pth \
  --data-path processed/phase1_output \
  --output-dir results/pr_curves
```

**Interpretation:**
- High AUPRC: Model maintains precision even at high recall
- Important for rare classes where AUROC can be misleading

## Correlation Analysis

### Clinical Features ↔ Abnormalities

Analyze relationships between clinical measurements and CXR findings:

**Script:**
```bash
python src/evaluation/correlation_analysis.py \
  --checkpoint experiments/full_model_stage2/best_model.pth \
  --data-path processed/phase1_output \
  --output-dir results/correlation_analysis
```

### Analyses Performed

#### 1. **Pearson Correlation**
- Linear relationship between continuous clinical features and abnormality probabilities
- Example: Temperature vs Pneumonia
- Heatmap: Clinical features (rows) × Abnormalities (columns)

#### 2. **Spearman Correlation**
- Monotonic (but not necessarily linear) relationships
- More robust to outliers
- Better for non-normal distributions

#### 3. **Point-Biserial Correlation**
- For binary clinical features (e.g., gender) vs abnormality predictions
- Example: Gender vs Cardiomegaly

### Clinical Features Analyzed

| Feature | Type | Clinical Significance |
|---------|------|---------------------|
| Age | Continuous | Risk factor for many conditions |
| Gender | Binary | Sex-specific disease patterns |
| Heart Rate | Continuous | Cardiac and pulmonary stress |
| Respiratory Rate | Continuous | Respiratory distress indicator |
| Temperature | Continuous | Infection/inflammation marker |
| O2 Saturation | Continuous | Hypoxemia indicator |
| Systolic BP | Continuous | Cardiovascular status |
| Diastolic BP | Continuous | Cardiovascular status |
| Acuity Level | Ordinal | Illness severity (1-5) |
| Pain Score | Ordinal | Subjective distress (0-10) |

### Expected Correlations

**Strong Positive Correlations:**
- Temperature ↔ Pneumonia (r ≈ 0.35)
- Respiratory Rate ↔ Pulmonary Edema (r ≈ 0.42)
- Low O2 Sat ↔ Multiple abnormalities (r ≈ -0.38)
- Acuity Level ↔ Severe findings (r ≈ 0.31)

**Demographic Patterns:**
- Age ↔ Cardiomegaly (r ≈ 0.28)
- Age ↔ Atelectasis (r ≈ 0.22)

### Visualization Outputs

1. **Correlation Heatmap**
   - 10 clinical features × 14 abnormalities grid
   - Color-coded by correlation strength
   - Significant correlations (p < 0.05) marked with asterisks

2. **Top Correlations Bar Chart**
   - 20 strongest feature-abnormality pairs
   - Sorted by absolute correlation coefficient

3. **Feature Importance per Abnormality**
   - For each abnormality, rank clinical features by correlation
   - Helps interpret which vital signs matter most

## Attention Visualization

### Cross-Modal Attention Maps

Visualize which modalities the model attends to:

**Script:**
```bash
python src/evaluation/visualize_attention.py \
  --checkpoint experiments/full_model_stage2/best_model.pth \
  --data-path processed/phase1_output \
  --num-samples 100 \
  --output-dir results/attention_maps
```

### Attention Matrices

The fusion layer produces attention weights showing interactions between modalities:

```
                Vision    Text    Clinical
Vision           0.45     0.35      0.20
Text             0.30     0.50      0.20
Clinical         0.25     0.30      0.45
```

**Interpretation:**
- Diagonal values: Self-attention (how much a modality attends to itself)
- Off-diagonal: Cross-modal attention
- Example: Vision→Text = 0.35 means vision features attend 35% to text context

### Visualizations

#### 1. **Attention Heatmaps**
- 3×3 matrix per sample
- Averaged across attention heads
- Color-coded intensity

#### 2. **Aggregated Attention Statistics**
- Mean attention weights across test set
- Per-abnormality attention patterns
- Identify which modalities matter most for each finding

#### 3. **Grad-CAM on Images** (Vision Attention)
- Overlay heatmap on X-ray showing attended regions
- Highlights important anatomical areas
- Separate Grad-CAM for each abnormality class

**Example:**
```
Sample: Patient 10000123, Pneumonia (predicted: 0.87, actual: 1)

Attention Weights:
  Vision → Text:     0.42  (high - text provides diagnostic context)
  Vision → Clinical: 0.18  (moderate - vitals suggest infection)
  Text → Vision:     0.38  (high - text directs visual attention)

Grad-CAM Regions:
  - Right lower lobe (high activation)
  - Costophrenic angle (moderate activation)
  → Consistent with right lower lobe pneumonia
```

## SHAP Feature Importance

### SHAP (SHapley Additive exPlanations)

Explain individual predictions using game-theoretic approach:

**Script:**
```bash
python src/evaluation/shap_analysis.py \
  --checkpoint experiments/full_model_stage2/best_model.pth \
  --data-path processed/phase1_output \
  --num-background 100 \
  --num-explain 500 \
  --output-dir results/shap_analysis
```

### SHAP Values

For each clinical feature, SHAP value indicates:
- **Positive SHAP**: Feature pushes prediction toward positive class
- **Negative SHAP**: Feature pushes prediction toward negative class
- **Magnitude**: Strength of influence

### Visualizations

#### 1. **SHAP Summary Plot**
- Beeswarm plot showing feature importance
- X-axis: SHAP value (impact on prediction)
- Y-axis: Features (ranked by importance)
- Color: Feature value (red = high, blue = low)

#### 2. **SHAP Dependence Plots**
- Scatter plot: Feature value vs SHAP value
- Shows how feature affects predictions
- Example: "Temperature vs SHAP for Pneumonia"
  - High temperature → positive SHAP (increases pneumonia probability)

#### 3. **SHAP Force Plots**
- Individual prediction explanation
- Shows each feature pushing prediction higher/lower
- Waterfall visualization

#### 4. **SHAP Interaction Values**
- 2D interaction effects
- Example: Age × Acuity interaction for Cardiomegaly

### Expected Insights

**High SHAP Features (common across abnormalities):**
- Age (baseline risk factor)
- Acuity level (illness severity)
- O2 saturation (respiratory compromise)

**Abnormality-Specific Features:**
- Pneumonia: Temperature (+), Respiratory rate (+)
- Pulmonary Edema: Heart rate (+), Respiratory rate (+)
- Pneumothorax: O2 saturation (-), Respiratory rate (+)

## Error Analysis

### Failure Mode Identification

**Script:**
```bash
python src/evaluation/error_analysis.py \
  --checkpoint experiments/full_model_stage2/best_model.pth \
  --data-path processed/phase1_output \
  --output-dir results/error_analysis
```

### Error Categories

#### 1. **False Positives** (Type I Errors)
- Model predicts abnormality, but ground truth is negative
- Clinical impact: Unnecessary follow-up, patient anxiety

**Analysis:**
- Which abnormalities have highest FP rate?
- Are FPs concentrated in certain patient subgroups?
- Do FPs correlate with ambiguous imaging findings?

#### 2. **False Negatives** (Type II Errors)
- Model misses actual abnormality
- Clinical impact: **Critical** - missed diagnoses

**Analysis:**
- Which abnormalities are most often missed?
- Are FNs associated with subtle findings?
- Do FNs occur more in rare classes?

#### 3. **High-Confidence Errors**
- Errors where model was very confident (p > 0.9)
- Most concerning errors for clinical deployment

### Error Reports

#### A. **Error Rate by Abnormality**
```
Abnormality           FP Rate   FN Rate   High-Conf Errors
Pneumonia             4.2%      3.8%      12
Pneumothorax          2.1%      5.5%      8
Lung Lesion           8.7%      12.3%     22  ← High error rate
```

#### B. **Error Case Studies**
- Sample 10 worst errors per abnormality
- Display:
  - X-ray image
  - Predicted probability
  - Ground truth label
  - Clinical features
  - Pseudo-note
  - Attention weights
  - Possible explanation

#### C. **Error Clustering**
- Group errors by similarity
- Identify common patterns:
  - "Borderline findings (ground truth uncertain)"
  - "Atypical presentations"
  - "Poor image quality"
  - "Overlapping pathologies"

### Actionable Insights

**From Error Analysis:**
1. Identify classes needing more training data (high FN rate)
2. Detect systematic biases (errors concentrated in subgroups)
3. Flag ambiguous cases for human review
4. Guide future model improvements

## Subgroup Analysis

### Performance Stratification

Ensure fair performance across patient demographics:

**Script:**
```bash
python src/evaluation/subgroup_analysis.py \
  --checkpoint experiments/full_model_stage2/best_model.pth \
  --data-path processed/phase1_output \
  --output-dir results/subgroup_analysis
```

### Subgroups Analyzed

#### 1. **Age Groups**
- Young (< 40 years)
- Middle-aged (40-65 years)
- Elderly (> 65 years)

**Expected Pattern:**
- Elderly: Higher prevalence of most abnormalities
- Model should maintain performance across age groups

#### 2. **Gender**
- Male vs Female
- Check for sex-based disparities

#### 3. **View Position**
- PA (Posterior-Anterior) - standard view
- AP (Anterior-Posterior) - portable/supine

**Expected:**
- AP views often lower quality → potentially lower AUROC
- Model should handle both view types

#### 4. **Acuity Level**
- Critical (Level 1-2)
- Urgent (Level 3)
- Non-urgent (Level 4-5)

**Expected:**
- Higher acuity → more abnormalities
- Performance should be consistent across acuity

#### 5. **Rare vs Common Findings**
- Common (prevalence > 10%): Atelectasis, Cardiomegaly, Edema, Support Devices
- Rare (prevalence < 5%): Fracture, Lung Lesion, Pleural Other

**Goal:**
- Rare findings should achieve AUROC ≥ 0.75 (harder benchmark)
- Common findings should achieve AUROC ≥ 0.85

### Fairness Metrics

#### Performance Parity
- AUROC difference between subgroups < 0.05
- Example: |AUROC_male - AUROC_female| < 0.05

#### Calibration Parity
- ECE similar across subgroups
- Predictions equally reliable for all patients

### Reports Generated

1. **Subgroup Performance Table**
   ```
   Abnormality    Overall  Male   Female  Age<40  Age40-65  Age>65
   Pneumonia      0.862    0.859  0.865   0.831   0.869     0.871
   Atelectasis    0.834    0.828  0.841   0.821   0.838     0.839
   ```

2. **Disparity Heatmap**
   - Subgroups (rows) × Abnormalities (columns)
   - Color indicates AUROC relative to overall
   - Highlight cells with disparities > 0.05

3. **Calibration Plots per Subgroup**
   - Separate reliability diagrams
   - Verify predictions are well-calibrated for all groups

## Modality Ablation Studies

### Per-Modality Contribution

Quantify each modality's importance:

**Script:**
```bash
python src/evaluation/ablation_study.py \
  --checkpoints \
    experiments/vision_only/best_model.pth \
    experiments/text_only/best_model.pth \
    experiments/clinical_only/best_model.pth \
    experiments/vision_text/best_model.pth \
    experiments/vision_clinical/best_model.pth \
    experiments/text_clinical/best_model.pth \
    experiments/full_model/best_model.pth \
  --data-path processed/phase1_output \
  --output-dir results/ablation_study
```

### Models Compared

| Model | Modalities | Parameters | Expected AUROC |
|-------|------------|------------|----------------|
| Vision-only | V | 87M | 0.810 |
| Text-only | T | 149M | 0.795 |
| Clinical-only | C | 525K | 0.720 |
| Vision + Text | V+T | 239M | 0.845 |
| Vision + Clinical | V+C | 89M | 0.825 |
| Text + Clinical | T+C | 149M | 0.810 |
| **Full Model** | **V+T+C** | **239M** | **0.860+** |

### Analysis

#### 1. **Marginal Contribution**
- Improvement from adding each modality
- Example:
  - Vision alone: 0.810
  - Vision + Text: 0.845 → **Text contributes +0.035**
  - Vision + Text + Clinical: 0.860 → **Clinical contributes +0.015**

#### 2. **Per-Abnormality Modality Importance**
- Which modality matters most for each finding?

**Expected Patterns:**
- **Pneumonia**: Text >> Vision (fever, cough in clinical note)
- **Pneumothorax**: Vision >> Text (visual finding, may be asymptomatic)
- **Cardiomegaly**: Vision > Clinical (imaging + vital signs)
- **Fracture**: Vision >>> others (purely visual)

#### 3. **Synergy Analysis**
- Is full model > sum of parts?
- Synergy = AUROC(V+T+C) - max(AUROC(V), AUROC(T), AUROC(C))
- Positive synergy indicates beneficial fusion

### Visualizations

1. **Ablation Bar Chart**
   - Mean AUROC for each model configuration
   - Error bars (95% CI)

2. **Per-Abnormality Comparison**
   - Heatmap: Models (rows) × Abnormalities (columns)
   - Shows which modalities excel for each class

3. **Marginal Contribution Waterfall**
   - Starting from vision-only baseline
   - Show incremental gains from adding modalities

## Statistical Significance Testing

### DeLong Test for AUROC Comparison

Test if performance differences are statistically significant:

**Script:**
```bash
python src/evaluation/statistical_tests.py \
  --checkpoint1 experiments/vision_only/best_model.pth \
  --checkpoint2 experiments/full_model/best_model.pth \
  --data-path processed/phase1_output \
  --alpha 0.05
```

### Hypotheses

**Null Hypothesis (H0):** AUROC_full = AUROC_baseline
**Alternative (H1):** AUROC_full > AUROC_baseline

**Result:**
```
DeLong Test: Full Model vs Vision-Only
Abnormality      AUROC_vision  AUROC_full  Difference  p-value  Significant
Pneumonia        0.768         0.821       +0.053      0.002    ***
Atelectasis      0.792         0.831       +0.039      0.018    *
Cardiomegaly     0.881         0.912       +0.031      0.045    *
...

*** p < 0.001, ** p < 0.01, * p < 0.05
```

### Bootstrap Confidence Intervals

Compute 95% CI for all metrics:

```bash
python src/evaluation/bootstrap_ci.py \
  --checkpoint experiments/full_model/best_model.pth \
  --data-path processed/phase1_output \
  --n-bootstrap 1000 \
  --output-dir results/bootstrap_ci
```

**Output:**
```
Abnormality      AUROC    95% CI
Pneumonia        0.821    [0.812, 0.830]
Atelectasis      0.831    [0.821, 0.841]
...
```

## Complete Evaluation Pipeline

### Run All Evaluations

```bash
# Full evaluation suite
bash scripts/evaluate_all.sh \
  --checkpoint experiments/full_model_stage2/best_model.pth \
  --data-path processed/phase1_output \
  --output-dir results/full_evaluation
```

**Estimated Runtime:**
- Metrics computation: ~5 minutes
- Confusion matrices: ~2 minutes
- ROC/PR curves: ~5 minutes
- Correlation analysis: ~3 minutes
- Attention visualization (100 samples): ~10 minutes
- SHAP analysis (500 samples): ~30 minutes
- Error analysis: ~10 minutes
- Subgroup analysis: ~10 minutes
- **Total**: ~75 minutes on single GPU

### Generated Report

```
results/full_evaluation/
├── index.html                  # Interactive dashboard
├── metrics/
│   ├── summary.csv
│   ├── per_abnormality.csv
│   └── calibration.csv
├── confusion_matrices/
│   ├── individual/             # 14 separate matrices
│   ├── combined_grid.png
│   └── normalized/
├── curves/
│   ├── roc/                    # ROC curves
│   ├── pr/                     # Precision-Recall curves
│   └── calibration/            # Reliability diagrams
├── correlation/
│   ├── heatmap.png
│   ├── top_correlations.png
│   └── correlation_matrix.csv
├── attention/
│   ├── heatmaps/               # 100 sample attention maps
│   ├── aggregated_stats.csv
│   └── gradcam/                # Vision attention overlays
├── shap/
│   ├── summary_plot.png
│   ├── dependence_plots/
│   ├── force_plots/
│   └── shap_values.csv
├── errors/
│   ├── error_summary.csv
│   ├── case_studies/
│   └── error_clusters.png
├── subgroups/
│   ├── performance_table.csv
│   ├── disparity_heatmap.png
│   └── calibration_by_subgroup/
├── ablation/
│   ├── model_comparison.csv
│   ├── modality_importance.png
│   └── synergy_analysis.csv
└── statistical_tests/
    ├── delong_tests.csv
    ├── bootstrap_ci.csv
    └── significance_summary.txt
```

## Summary

This evaluation guide provides:
- ✅ **Comprehensive metrics** (AUROC, AUPRC, F1, Sensitivity, Specificity, ECE, Brier)
- ✅ **Multi-label confusion matrices** (per-abnormality and combined)
- ✅ **ROC & PR curves** with confidence intervals
- ✅ **Correlation analysis** (clinical features ↔ abnormalities)
- ✅ **Attention visualization** (cross-modal attention + Grad-CAM)
- ✅ **SHAP feature importance** (explainable AI)
- ✅ **Error analysis** (failure modes and case studies)
- ✅ **Subgroup analysis** (fairness across demographics)
- ✅ **Modality ablation** (per-modality contribution)
- ✅ **Statistical tests** (DeLong, bootstrap CIs)

**Complete evaluation**: ~75 minutes
**Output**: Interactive HTML report + all visualizations

## Next Steps

1. Review evaluation results
2. Compare against MDF-NET benchmarks
3. Identify areas for improvement
4. Iterate on model architecture/training
5. Document findings for publication

## Additional Resources

- **Training Guide**: [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- **Model Architecture**: [README.md](../README.md#model-architecture)
- **Phase 3 Integration**: [PHASE3_INTEGRATION.md](PHASE3_INTEGRATION.md)
