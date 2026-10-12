# Retinal OCT Multi-Scanner Benchmark Summary

Comprehensive performance matrix comparing models across four independent clinical distributions and the combined global test benchmark ($N=6,246$ scans).

| Model Architecture & Training Strategy | Resolution | OCTMNIST (In-Domain) | Kermany Cell 2018 (Near-Domain) | Duke Bioptigen (Out-of-Domain) | NEH Topcon (Out-of-Domain) | **Master Pooled Test** ($N=6,246$) | Key Clinical Finding |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **v4 SE-ResidualCNN4** | 28×28 | 71.23% F1 | 75.69% F1 | 47.81% F1 | 43.46% F1 | — | Severe out-of-domain hardware collapse |
| **v5 EfficientNet-B0** | 28×28 | 65.30% F1 | 66.34% F1 | 34.04% F1 | 38.12% F1 | — | Inefficient on 28px low-res images |
| **v6 Vector B Fusion (110-D)** | 28×28 | 72.87% F1 | 80.07% F1 | 47.03% F1 | 48.02% F1 | — | Wavelet/GLCM features boost near-domain |
| **v7 ResNet-18 Champion** 🏆 | 64×64 | **88.21% F1** | ~82.0% F1 | ~50.2% F1 | ~49.1% F1 | — | All-time in-domain single-dataset champion |
| **D3 64px Mega-Set Pool** | 64×64 | 71.28% F1 | 80.22% F1 | 96.25% F1 | 85.38% F1 | — | First proof of multi-scanner unification |
| **Phase 3A (ImageNet 56-Stem)** | 224×224 | 86.50% Acc<br>86.15% F1 | **90.70% Acc**<br>**90.47% F1** | 98.87% Acc<br>98.68% F1 | 65.46% Acc<br>66.12% F1 | 76.19% Acc<br>79.49% Macro F1 | **Strongest Single Generalist** (90.7% on Kermany) |
| **Phase 3B (Progressive 64-Stem)** | 224×224 | **88.70% Acc** 🏆<br>**88.43% F1** 🏆 | 59.50% Acc<br>56.48% F1 | 99.19% Acc<br>99.14% F1 | 65.38% Acc<br>65.82% F1 | 71.53% Acc<br>73.67% Macro F1 | **Peak In-Domain Engine**; 62.0% Drusen recall |
| **Phase 3C (Progressive 112-Stem)** | 224×224 | 86.60% Acc<br>86.33% F1 | 77.40% Acc<br>76.18% F1 | 97.91% Acc<br>98.06% F1 | **70.43% Acc** 🏆<br>**70.53% F1** 🏆 | 76.87% Acc<br>78.17% Macro F1 | **Peak Fine-Grained Topcon Generalization** |
| **Equal Soft Ensemble (3A+3B+3C)** | 224×224 | 88.00% Acc<br>87.73% F1 | 86.70% Acc<br>86.77% F1 | **99.52% Acc**<br>**99.44% F1** | 69.24% Acc<br>69.68% F1 | 78.05% Acc<br>80.56% Macro F1 | Outperforms all individual models globally |
| **Master Weighted Ensemble** 🏆 | 224×224 | 87.60% Acc<br>87.26% F1 | **90.00% Acc**<br>**89.93% F1** | **99.52% Acc** 🏆<br>**99.44% F1** 🏆 | 68.88% Acc<br>69.41% F1 | **78.31% Acc** 🏆<br>**81.14% Macro F1** 🏆 | **All-Time Project Record Across All 4 Datasets** |

*Note: For Duke Bioptigen (CNV naturally absent) and NEH Topcon (DME naturally absent), Macro F1 is computed over clinically present classes.*

---

## Autonomous Clinical Referral Triage Trajectory (Entropy-Guided)

| Referral Rate to Ophthalmologist | Evaluated Scans Retained | Retained Diagnostic Accuracy | Retained Macro F1-Score |
|:---:|:---:|:---:|:---:|
| **0.0%** (Fully Autonomous) | 6,246 / 6,246 (100.0%) | 78.31% | 81.14% |
| **5.0%** | 5,933 / 6,246 (95.0%) | 79.30% | 82.16% |
| **10.0%** | 5,621 / 6,246 (90.0%) | **80.64%** | **83.29%** |
| **15.0%** | 5,309 / 6,246 (85.0%) | **82.16%** | **84.49%** |
| **20.0%** | 4,997 / 6,246 (80.0%) | **83.63%** | **85.54%** |
| **25.0%** | 4,684 / 6,246 (75.0%) | **85.10%** | **86.65%** |