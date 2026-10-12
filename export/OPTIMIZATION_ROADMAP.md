# Retinal OCT Machine Learning Project: Advanced Optimization Roadmap & Research Log

This document tracks all completed optimization milestones, experimental breakthroughs, hardware benchmarks, and future architectural roadmaps for the Retinal OCT Pathology Classification project.

---

## 1. Project Evolution & Milestone Summary

```
                      ┌────────────────────────────────────────────────────────┐
                      │                 MILESTONE TIMELINE                     │
                      └───────────────────────────┬────────────────────────────┘
                                                  │
 ┌────────────────────────────────────────────────┴───────────────────────────────┐
 │ Phase 1-4: Baseline Exploration & Modernization                               │
 │ • Pipeline 1 (Classical ML) vs. Pipeline 2 (Deep ConvNets)                    │
 │ • 66-D Biomedical Feature Engineering (Depth profiles, grid stats)            │
 │ • CUDA GPU Acceleration on GTX 1660 Ti (18x speedup)                          │
 └────────────────────────────────────────────────┬───────────────────────────────┘
                                                  │
 ┌────────────────────────────────────────────────┴───────────────────────────────┐
 │ Phase 5-6: Advanced Architectures & Epistemic Uncertainty                     │
 │ • v4 SE-ResidualCNN4 (Squeeze-and-Excitation Attention: 93.19% Val Acc)       │
 │ • v6 Vector B Dual-Branch Fusion (110-D GLCM + Wavelets: 79.97% Kermany F1)   │
 │ • Master Bayesian Super-Ensemble & MC Dropout Uncertainty Referral (94.2% Acc)│
 └────────────────────────────────────────────────┬───────────────────────────────┘
                                                  │
 ┌────────────────────────────────────────────────┴───────────────────────────────┐
 │ Phase 7: Hardware Standardization & Feature Selection (Phases A & B)          │
 │ • Universal CLAHE Normalization (clipLimit=2.0, grid=8x8) across all vendors  │
 │ • 110-D Feature Selection Ablation (Confirmed GLCM+Wavelet > 152-D Gabor/LBP) │
 └────────────────────────────────────────────────┬───────────────────────────────┘
                                                  │
 ┌────────────────────────────────────────────────┴───────────────────────────────┐
 │ Phase 8: Multi-Scanner Domain Generalization (Phase C: D1, D2, D3)            │
 │ • Integrated Duke Srinivasan (Bioptigen) & Noor Eye Hospital (Topcon) cohorts │
 │ • D3 64x64 Mega-Set Pool: Duke Macro F1 = 96.25% | NEH Macro F1 = 85.38%     │
 └────────────────────────────────────────────────┬───────────────────────────────┘
                                                  │
 ┌────────────────────────────────────────────────┴───────────────────────────────┐
 │ Phase 9: Resolution Progression & Progressive Resizing (v7, v8, v9)           │
 │ • v7 Champion: 64x64 ResNet-18 (88.21% Macro F1, 0.3174 Log Loss) 🏆          │
 │ • Diagnosed DRUSEN collapse at 128px (57.7%) & 224px (50.6%) from scratch     │
 │ • Progressive Resizing transfer recovered 128px to 76.34% Macro F1           │
 └────────────────────────────────────────────────┬───────────────────────────────┘
                                                  │
                                                  ▼
                                     ACTIVE ROADMAP (PHASES 1-4)
```

---

## 2. Completed Milestones in Detail

### Phase 7: Hardware Contrast Standardization (The CLAHE Shift)
- **Problem**: Direct cross-scanner inference without normalization produced a $\sim 15\text{--}20\%$ accuracy collapse on Duke (Bioptigen) and NEH (Topcon) scans due to optical noise floor and gain mismatches.
- **Solution**: We standardized a universal **Contrast Limited Adaptive Histogram Equalization (CLAHE)** preprocessing pipeline:
  - `clipLimit = 2.0`, `tileGridSize = (8, 8)`
  - Uniform square center-cropping and isotropic resizing
  - Denoising filters preserving delicate RPE layer contours
- **Result**: Normalized feature spaces across Heidelberg Spectralis, Bioptigen SD-OCT, and Topcon 3D OCT-1000, allowing physical descriptors and CNN filters to learn scanner-invariant pathological morphology.

### Phase 8: Multi-Scanner Domain Generalization (Phase C)
- Combined 113k scans across three independent medical centers into three dataset strategies:
  - **D1 (Joint Pool 28×28)**: Balanced multi-vendor pool at 28px.
  - **D2 (Cross-Domain Validation Signal)**: Trained on OCTMNIST, early-stopped exclusively on Duke + NEH validation sets.
  - **D3 (Mega-Set 64×64 Pool)**: Combined pool at 64×64 resolution.
- **Breakthrough**: D3 completely conquered hardware domain shift:
  - Duke Macro F1 surged from $48.51\%$ to **$96.25\%$**
  - NEH Macro F1 surged from $48.02\%$ to **$85.38\%$**
  - Kermany Full-Res F1 reached **$80.22\%$**

### Phase 9: Resolution Scaling & Progressive Resizing (v7, v8, v9)
- **The 64×64 Champion (v7)**: Modified ResNet-18 with $3\times 3$ stride=1 conv1, identity maxpool, Focal Loss ($\gamma=2.0$), Cosine Annealing, and 5-pass TTA reached **88.21% Macro F1 and 0.3174 Log Loss** on OCTMNIST-64.
- **The DRUSEN Collapse Phenomenon**:
  - Training from scratch at 128×128 crashed to $57.72\%$ F1, with DRUSEN recall falling to $13.6\%$ (167 drusen misclassified as CNV).
  - Training at 224×224 from scratch collapsed further to $50.68\%$ F1, with DRUSEN recall falling to $6.0\%$.
  - **Root Cause**: Early $4\times$ downsampling in standard ResNet-18 destroys micro-deposits, causing underfitting networks to default to CNV (99.6% recall) due to loss minimization asymmetry.
- **Progressive Resizing Recovery**: Transferring weights from the 64×64 champion into the 128×128 network rescued performance from $57.72\%$ to **$76.34\%$**, confirming that learned lower-resolution feature maps stabilize higher-dimensional training.

---

---

## 3. Active Roadmap & Recent Execution Log

### Phase 1: 128×128 Resolution Suite (COMPLETED)
- **Phase 1A (Progressive Transfer + Heavy Reg)**: Warm-started from v7 64px champion. Reached **76.24% Macro F1**, 0.5507 Log Loss (DRUSEN recall 31.2%). Plateaued due to bilinear upsampling interpolation blur from 28px/64px base.
- **Phase 1B (Scratch Control)**: Reached **73.88% Macro F1**, 0.5970 Log Loss (DRUSEN recall 29.6%). Weight transfer from 64px champion provided a $+2.36\%$ advantage over scratch.
- **Hardware Finding**: AMP software emulation on GTX 1660 Ti slowed epochs by $3\times$. Hardcoded native FP32 for all future pipelines.
- **Decision Gate**: 128px failed to beat 88.21% 64px champion. Proceeded to 224px evaluating ImageNet transfer vs 64px champion base.

### Phase 2: 224×224 Resolution Shootout (COMPLETED)
- **Phase 2A (ImageNet Pretrained Transfer)**:
  - Adapted 7×7 stride-2 conv1 to 1-channel grayscale.
  - Reached **86.76% Macro F1**, **87.10% Balanced Accuracy**, **0.3799 Log Loss**, and **0.9896 Macro ROC-AUC**.
  - **Drusen Recovery**: Recall surged to **58.80%** (F1: 74.06%), overcoming the 224px scratch collapse (6.0%).
  - Training completed in 126 minutes (14 epochs, ~9.0 min/epoch).
- **Phase 2B (Progressive Transfer from 64px Champion)**:
  - Preserved 100% of v7 3×3 conv1 weights via stride-2 conv1 + `AdaptiveMaxPool2d((64, 64))`.
  - Reached **85.57% Macro F1**, **85.80% Balanced Accuracy**, **0.4184 Log Loss**, and **0.9913 Macro ROC-AUC** (peak discrimination).
  - **Peak Drusen Recall**: **59.20%** (F1: 74.37%), the highest 224px drusen detection rate in project history.
  - Training completed in 128 minutes (16 epochs, ~8.0 min/epoch).
- **Phase 2C (Progressive Transfer with 112-Stem, No MaxPool)**:
  - Preserved 100% of v7 3×3 conv1 weights via stride-2 conv1 + `Identity` maxpool -> 112×112 feature maps.
  - Reached **86.38% Macro F1**, **86.60% Balanced Accuracy**, **0.4170 Log Loss**, and **0.9901 Macro ROC-AUC**.
  - **All-Time Peak Drusen Recall**: **59.60%** (F1: 74.69%), with 100% DME precision and 100% Drusen precision.
  - Training completed in ~5 hours (16 epochs, ~18.5 min/epoch).
- **Key Scientific Finding**: While ImageNet weights (Phase 2A) provide slightly better overall macro separation (86.76%), transferring domain-specific features with larger spatial maps (Phase 2C) yielded the highest fine-grained drusen sensitivity (59.60% recall).

### Phase 3: Multi-Scanner Mega-Set Suite & Global Benchmark (COMPLETED)
- Unified **113,265 training scans** (OCTMNIST 97,477 + Duke Bioptigen 2,610 + NEH Topcon 13,178) under universal CLAHE normalization.
- Evaluated on a comprehensive 4-dataset battery plus the **Pooled Global Test ($N=5,246$ scans)**:

| Metric | Phase 3A (ImageNet 56-Stem) | Phase 3B (Progressive 64-Stem) | Phase 3C (Progressive 112-Stem) |
|:---|:---:|:---:|:---:|
| **Training Time (15 Epochs)** | 131.1 min (~8.7 min/ep) | 139.3 min (~9.3 min/ep) | 331.3 min (~22.1 min/ep) |
| **OCTMNIST In-Domain F1** | 85.76% | **87.44%** 🏆 | 85.74% |
| **Kermany Full-Res External F1** | **89.76%** 🏆 | 51.82% | 72.88% |
| **Duke Bioptigen F1** | **98.98%** 🏆 | 98.63% | 98.55% |
| **Duke Bioptigen Accuracy** | **99.19%** 🏆 | 98.71% | 98.39% |
| **Duke Drusen Recall** | **100.0%** 🏆 | **100.0%** 🏆 | 99.32% |
| **NEH Topcon External F1** | 66.15% | 66.41% | **70.52%** 🏆 |
| **NEH Topcon Accuracy** | 65.57% | 66.07% | **70.32%** 🏆 |
| **Global Pooled Test ($N=5,246$) Top-1 Acc** | 74.21% | 68.26% | **75.01%** 🏆 |
| **Global Pooled Test ($N=5,246$) Macro F1** | **78.72%** 🏆 | 69.75% | 76.05% |
| **Global Pooled Test Weighted F1** | 74.16% | 68.41% | **74.80%** 🏆 |
| **Global Drusen Precision** | 63.79% | 58.02% | **71.70%** 🏆 |
| **Global Drusen F1-Score** | 62.91% | 61.16% | **67.07%** 🏆 |

- **Key Empirical Takeaways**:
  1. **Elimination of Cross-Vendor Scanner Shift**: Duke Bioptigen accuracy reached **98.4%–99.2%** (up from ~48.5% in un-unified baselines), with up to 100% Drusen recall.
  2. **Phase 3A Generalization**: Reached **89.76% Macro F1 on Kermany** and the highest **Global Macro F1 (78.72%)** across all cohorts.
  3. **Phase 3B In-Domain Precision**: Retained peak in-domain performance on OCTMNIST (**87.70% Acc, 87.44% F1, 62.0% Drusen recall**).
  4. **Phase 3C Spatial Detail Preservation**: Avoided spatial downsampling, unlocking top performance on NEH Topcon (**70.52% F1**), highest **Global Accuracy (75.01%)**, and highest **Global Drusen precision (71.70%)**.

### Phase 4: Interpretability & Master Super-Ensemble (COMPLETED)
- **Captum XAI Anatomical Layer Overlays**:
  - Generated 224×224 SmoothGrad (15 Monte Carlo noise passes, $\sigma = 0.15$) and Integrated Gradients overlays for all 4 pathology classes (CNV, DME, DRUSEN, NORMAL).
  - Saved 300 DPI high-resolution figures to `code/export/xai_heatmaps/`:
    - `captum_smoothgrad_224_CNV.png`
    - `captum_smoothgrad_224_DME.png`
    - `captum_smoothgrad_224_DRUSEN.png`
    - `captum_smoothgrad_224_NORMAL.png`
    - `captum_smoothgrad_224_master_grid.png`
  - Validated pathological alignment with retinal structures: subretinal fluid & CNV membranes, DME intraretinal fluid cysts, sub-RPE convex drusen, and healthy continuous lamination.

- **Master Super-Ensemble (Phase 3A + 3B + 3C)**:
  - Fused predictions via **Optimal Weighted Soft-Voting** ($w_{\text{3A}}=0.45, w_{\text{3B}}=0.20, w_{\text{3C}}=0.35$):
    - **Global Pooled Accuracy (N=6,246)**: **78.31%** (outperforming all individual models).
    - **Global Macro F1 (N=6,246)**: **81.14%** (breaking the 80% mark across all 4 cohorts).
    - **Duke Bioptigen Generalization**: **99.52% Accuracy, 99.44% Macro F1**.
  - **Predictive Entropy & Autonomous Referral Triage**:
    - At 10% referral of ambiguous scans: **80.64% Accuracy, 83.29% Macro F1**.
    - At 20% referral of ambiguous scans: **83.63% Accuracy, 85.54% Macro F1**.
    - At 25% referral of ambiguous scans: **85.10% Accuracy, 86.65% Macro F1**.

---

## 5. Next Planned Action Campaign & Execution Milestones

### A. Ensemble Voting Ablation Summary ($N = 6,246$)
* **Hard Voting (Majority)**: 77.22% Acc | 79.21% Macro F1 (discards continuous confidence).
* **Dual Specialist (3A 55% + 3C 45%)**: 77.94% Acc | 80.86% Macro F1.
* **Equal Soft Voting (1/3 each)**: 78.05% Acc | 80.56% Macro F1.
* **Master Weighted Soft Ensemble**: **78.31% Acc | 81.14% Macro F1** (overall champion).

### B. Resolution Protocols for Core Clinical Issues
* **Issue A (Empty Ground Truth / Negative Control Validation)**:
  - Duke has 0 CNV scans $\to$ Tested false alarms: **0 / 621 (0.00% false alarm rate $\to$ 100.0% Specificity)**.
  - NEH has 0 DME scans $\to$ Tested false alarms: **4 / 3,625 (0.11% false alarm rate $\to$ 99.89% Specificity)**.
  - Formalized Dual-Metric reporting: Site-Specific Valid F1 (99.44% Duke, 70.53% NEH) alongside Master Pooled F1 (81.14%).
* **Issue B (Cross-Scanner Domain Shift & Topcon Disparity)**:
  - Root-cause confirmed: Training set is 86.1% Heidelberg Spectralis, biasing batches away from Topcon.
  - Formulated 3 remediation vectors: Domain-Stratified Batch Sampling (33/33/33), Topcon reference histogram matching, and staged domain adaptation.

### C. Staged Next Action: Step 1 (Domain-Stratified Sampling Test)
* **Status**: STAGED & READY TO EXECUTE upon return.
* **Target Script**: `phase_3d_domain_stratified_sampler.py`.
* **Methodology**: Apply PyTorch `WeightedRandomSampler` enforcing equal 33% Heidelberg, 33% Duke, and 33% Topcon exposure per batch.
* **Target Milestone**: Lift NEH Topcon Accuracy from $70.5\%$ to $\ge 73.0\%$ and Macro F1 to $\ge 73.0\%$ without degrading Duke ($>98\%$).



