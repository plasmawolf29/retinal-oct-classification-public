# Multi-Scanner Retinal OCT Classification: Consolidated Experimental Synthesis

This document presents the complete diagnostic and comparative benchmarks for the UConn BME Machine Learning Final Project. All models were trained on high-performance ResNet-18 variants with grayscale single-channel input adaptations and evaluated on **$N = 6,246$ held-out clinical B-scans** across four distinct cohorts (Duke Bioptigen, NEH Topcon, Kermany Cell 2018, and OCTMNIST Heidelberg).

---

## Table 1: Complete Progression of Individual Architectures (Phases 1A to 3E)

| Phase | Architecture & Input Resolution | Training Strategy / Sampler | Training Scans | GPU Time (min) | Global Pooled Acc (%) | Global Pooled Macro F1 (%) | Kermany External F1 (%) | Duke Bioptigen Valid F1 (%) | NEH Topcon Valid F1 (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 1A** | ResNet-18 ($128 \times 128$) | Progressive Resolution (Frozen Stem) | 97,477 | 45.2 | 70.84% | 72.41% | 82.10% | 94.20% | 58.40% |
| **Phase 1B** | ResNet-18 ($128 \times 128$) | Scratch Control Baseline | 97,477 | 51.8 | 69.12% | 70.55% | 79.80% | 92.10% | 56.10% |
| **Phase 2A** | ResNet-18 ImageNet ($224 \times 224$) | ImageNet Transfer (56-stem) | 97,477 | 126.4 | 74.50% | 77.10% | 88.50% | 97.40% | 63.80% |
| **Phase 2B** | ResNet-18 ($224 \times 224$) | Progressive 64-stem ($64 \times 64$ maps) | 97,477 | 134.1 | 72.10% | 74.80% | 86.20% | 96.80% | 62.10% |
| **Phase 2C** | ResNet-18 ($224 \times 224$) | Progressive 112-stem ($112 \times 112$ maps) | 97,477 | 296.8 | 74.80% | 76.90% | 85.90% | 96.50% | 65.40% |
| **Phase 3A** | ResNet-18 ImageNet ($224 \times 224$) | Multi-Scanner Mega-Pool (Natural: 86/12/2%) | 113,265 | 131.1 | 76.19% | 79.49% | 90.47% | 98.68% | 66.12% |
| **Phase 3B** | ResNet-18 ($224 \times 224$) | Progressive 64-stem Mega-Pool | 113,265 | 139.3 | 71.53% | 73.67% | 56.48% | 99.14% | 65.82% |
| **Phase 3C** | ResNet-18 ($224 \times 224$) | Progressive 112-stem Mega-Pool | 113,265 | 331.3 | 76.87% | 78.17% | 76.18% | 98.06% | 70.53% |
| **Phase 3D** | ResNet-18 ImageNet ($224 \times 224$) | Domain Parity (33% Hdbg / 33% Topcon / 33% Duke) | 113,265 | 171.5 | 76.74% | 80.08% | 91.84% | 99.20% | 67.09% |
| **Phase 3E** | ResNet-18 ImageNet ($224 \times 224$) | Proportional Dampening (**50% Hdbg / 40% Topcon / 10% Duke**) 🏆 | 113,265 | 165.2 | **77.33%** | **80.47%** | **92.76%** | **99.85%** | **67.28%** |

---

## Table 2: Ensemble Combinatorics, Voting Strategies, and Parameter Sweeps

| Ensemble Strategy | Formulation / Weights | Models Included | Global Top-1 Acc (%) | Global Macro F1 (%) | Kermany External F1 (%) | Duke Bioptigen Valid F1 (%) | NEH Topcon Valid F1 (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Prior 3-Model Weighted** | $0.45 \cdot 3A + 0.20 \cdot 3B + 0.35 \cdot 3C$ | 3A, 3B, 3C | 78.31% | 81.14% | 89.93% | 99.44% | 69.41% |
| **Equal 4-Model Soft** | Uniform ($0.25$ each) | 3A, 3B, 3C, 3D | 78.74% | 81.51% | 90.48% | 99.65% | 70.06% |
| **Equal 5-Model Pentad** | Uniform ($0.20$ each) | 3A, 3B, 3C, 3D, 3E | 78.82% | 81.72% | 91.58% | 99.65% | 69.90% |
| **Dual Specialist (Equal)** | $0.50 \cdot 3C + 0.50 \cdot 3E$ | 3C, 3E | 79.09% | 81.59% | 89.39% | 100.00% | 70.79% |
| **Dual Specialist (Optimal Sweep) 🥈** | $\mathbf{0.38 \cdot 3C + 0.62 \cdot 3E}$ | 3C, 3E | **79.07%** | **81.81%** | **90.96%** | **100.00%** | **70.61%** |
| **Weighted Champion 🏆** | $\mathbf{0.50 \cdot 3E + 0.33 \cdot 3C + 0.17 \cdot 3D}$ | 3C, 3D, 3E | **79.04%** | **81.87%** | **91.95%** | **99.85%** | **70.61%** |

---

## Table 3: Negative Control Specificity Analysis (Addressing Incomplete Cohorts)

*Clinical rationale: Duke cohort contains 0 true CNV scans; NEH cohort contains 0 true DME scans. Rather than penalizing models for zero true positives, evaluation audits verify negative-control false-alarm suppression.*

| Evaluated Model / Configuration | Duke CNV False Alarms ($N = 621$) | Duke True Specificity (%) | NEH DME False Alarms ($N = 3,625$) | NEH True Specificity (%) | Total Negative Control Specificity (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Phase 3A** | 0 / 621 | 100.00% | 4 / 3,625 | 99.89% | 99.91% |
| **Phase 3D (33/33/33)** | 0 / 621 | 100.00% | 1 / 3,625 | 99.97% | 99.98% |
| **Phase 3E (50/40/10)** | **0 / 621** | **100.00%** | **0 / 3,625** | **100.00% 🎯** | **100.00% 🎯** |
| **Dual Specialist (38% 3C + 62% 3E)** | **0 / 621** | **100.00%** | **0 / 3,625** | **100.00%** | **100.00%** |
| **Weighted Champion (50% 3E + 33% 3C + 17% 3D)** | **0 / 621** | **100.00%** | **0 / 3,625** | **100.00%** | **100.00%** |

---

## Table 4: Clinical Autonomous Referral Operating Characteristic (ROC-Triage)

*Predictive epistemic uncertainty thresholding computed via Shannon predictive entropy $H(x) = -\sum p_i \log_2 p_i$ on the Weighted Champion ensemble across all $N = 6,246$ scans.*

| Ophthalmologist Referral Rate (%) | Scans Retained for Autonomous AI Diagnosis | Autonomous Top-1 Accuracy (%) | Autonomous Macro F1-Score (%) | Autonomous Weighted F1-Score (%) | Clinical Operational Interpretation |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.0%** | 6,246 | 79.04% | 81.87% | 78.93% | Full autonomous baseline deployment |
| **5.0%** | 5,934 | 80.62% | 83.17% | 80.52% | Filters borderline / severe optical artifacts |
| **10.0%** | 5,622 | 82.14% | 84.38% | 82.04% | Standard clinical screening triage |
| **15.0%** | 5,310 | 83.50% | 85.39% | 83.41% | High-confidence diagnostic grade |
| **20.0%** | 4,997 | 85.15% | 86.67% | 85.08% | Specialist-verified tier |
| **25.0%** | 4,685 | 86.28% | 87.39% | 86.23% | High sensitivity screening |
| **30.0%** | 4,373 | 87.72% | 88.44% | 87.69% | Critical referral mode (difficult cases escalated) |
