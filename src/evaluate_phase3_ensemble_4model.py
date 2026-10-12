"""
Phase 3 Master 4-Model Super-Ensemble & Uncertainty Triage Evaluation
=====================================================================
Evaluates a 4-Model Super-Ensemble combining:
  1. Phase 3A: ResNet-18 ImageNet Transfer (56x56 feature maps)
  2. Phase 3B: Progressive 64-stem ResNet-18 (64x64 feature maps)
  3. Phase 3C: Progressive 112-stem ResNet-18 (112x112 feature maps)
  4. Phase 3D: Domain-Stratified ResNet-18 (33% Heidelberg / 33% Duke / 33% Topcon parity)

Ensemble Configurations Evaluated:
  - Baseline 3-Model Ensemble (3A + 3B + 3C, weights: 0.45, 0.20, 0.35)
  - Equal 4-Model Soft Voting (w = 0.25, 0.25, 0.25, 0.25)
  - Domain-Specialist 4-Model Ensemble (w_A=0.35, w_B=0.15, w_C=0.25, w_D=0.25)
  - Dual Domain Champions (3A + 3D, weights: 0.50, 0.50)
  - Hard Majority Voting across all 4 models

Evaluated on:
  - N = 6,246 Pooled Test Scans (OCTMNIST, Kermany, Duke, NEH)
  - Site-specific cohorts with Negative Control Specificity testing
  - Autonomous Clinical Referral Triage via Shannon Predictive Entropy H(x)

Outputs:
  - combined results/ensemble_4model_benchmark_results.json
  - combined results/ensemble_4model_confusion_matrix.png
  - combined results/ensemble_4model_triage_referral_curve.png
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = SCRIPT_DIR.parents[1]
BASE_DIR   = SCRIPT_DIR.parents[3]
OUTPUT_DIR = SCRIPT_DIR / "combined results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from prep_phase3_combined import get_benchmark_dataloaders
from medmnist import OCTMNIST
from torch.utils.data import DataLoader
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

# --- MODEL ARCHITECTURES ---
def build_model_3a():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_feat, 4))
    return model

def build_model_3b():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
    model.maxpool = nn.AdaptiveMaxPool2d((64, 64))
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_feat, 4))
    return model

def build_model_3c():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
    model.maxpool = nn.Identity()
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_feat, 4))
    return model

def build_model_3d():
    # Phase 3D shares the standard ResNet-18 ImageNet transfer structure of 3A
    return build_model_3a()

def load_all_four_models():
    print("[*] Loading all 4 model checkpoints into GPU memory...")
    p_3a = SCRIPT_DIR / "phase_3a_resnet18_224x224_imagenet_mega_best.pth"
    p_3b = SCRIPT_DIR / "phase_3b_resnet18_224x224_64stem_mega_best.pth"
    p_3c = SCRIPT_DIR / "phase_3c_resnet18_224x224_112stem_mega_best.pth"
    p_3d = SCRIPT_DIR / "phase_3d_resnet18_224x224_stratified_best.pth"
    
    for p in [p_3a, p_3b, p_3c, p_3d]:
        if not p.exists():
            raise FileNotFoundError(f"[!] Checkpoint not found: {p}")
            
    m_3a = build_model_3a().to(DEVICE)
    m_3a.load_state_dict(torch.load(p_3a, map_location=DEVICE))
    m_3a.eval()
    
    m_3b = build_model_3b().to(DEVICE)
    m_3b.load_state_dict(torch.load(p_3b, map_location=DEVICE))
    m_3b.eval()
    
    m_3c = build_model_3c().to(DEVICE)
    m_3c.load_state_dict(torch.load(p_3c, map_location=DEVICE))
    m_3c.eval()
    
    m_3d = build_model_3d().to(DEVICE)
    m_3d.load_state_dict(torch.load(p_3d, map_location=DEVICE))
    m_3d.eval()
    
    print("[+] All 4 checkpoints loaded successfully (3A, 3B, 3C, 3D).")
    return m_3a, m_3b, m_3c, m_3d

def get_all_test_loaders():
    loaders = get_benchmark_dataloaders(image_size=224, batch_size=32, in_channels=1)
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    oct_test_ds = OCTMNIST(split="test", download=False, transform=tf, size=64)
    loaders["octmnist"] = DataLoader(oct_test_ds, batch_size=32, shuffle=False)
    print(f"[+] Loaded in-domain 'octmnist' test loader: {len(oct_test_ds)} samples.")
    return loaders

def predict_dataloader(model, loader):
    probs_list = []
    targets_list = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
            
            if isinstance(lbls, torch.Tensor):
                targets_list.extend(lbls.squeeze().numpy().tolist())
            else:
                targets_list.extend(lbls)
    return np.concatenate(probs_list, axis=0), np.array(targets_list, dtype=np.int64)

def evaluate_metrics(y_true, y_pred, present_classes=None):
    acc = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average="weighted", zero_division=0)
    
    if present_classes is not None:
        f1_per_cls = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
        valid_f1s = [f1_per_cls[c] for c in present_classes]
        macro_f1 = float(np.mean(valid_f1s))
    else:
        macro_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)
        
    return acc, macro_f1, weighted_f1

def main():
    print("=" * 75)
    print("PHASE 3: MASTER 4-MODEL SUPER-ENSEMBLE EVALUATION (3A + 3B + 3C + 3D)")
    print(f"Device: {DEVICE}")
    print("=" * 75)
    
    m_3a, m_3b, m_3c, m_3d = load_all_four_models()
    loaders = get_all_test_loaders()
    
    present_map = {
        "duke": [1, 2, 3],        # DME, DRUSEN, NORMAL (CNV absent)
        "neh": [0, 2, 3],         # CNV, DRUSEN, NORMAL (DME absent)
        "kermany": [0, 1, 2, 3],  # All 4
        "octmnist": [0, 1, 2, 3]  # All 4
    }
    
    dataset_results = {}
    pooled_probs_3a, pooled_probs_3b = [], []
    pooled_probs_3c, pooled_probs_3d = [], []
    pooled_targets  = []
    
    for name, loader in loaders.items():
        print(f"\n--- Inferencing Benchmark: {name.upper()} (N={len(loader.dataset)}) ---")
        p_3a, y = predict_dataloader(m_3a, loader)
        p_3b, _ = predict_dataloader(m_3b, loader)
        p_3c, _ = predict_dataloader(m_3c, loader)
        p_3d, _ = predict_dataloader(m_3d, loader)
        
        # Ensembles
        p_3model_opt = (0.45 * p_3a + 0.20 * p_3b + 0.35 * p_3c)
        p_4model_eq  = (p_3a + p_3b + p_3c + p_3d) / 4.0
        p_4model_opt = (0.35 * p_3a + 0.15 * p_3b + 0.25 * p_3c + 0.25 * p_3d)
        p_dual_champ = (0.50 * p_3a + 0.50 * p_3d)
        
        preds_3a = p_3a.argmax(axis=1)
        preds_3d = p_3d.argmax(axis=1)
        preds_3m = p_3model_opt.argmax(axis=1)
        preds_4e = p_4model_eq.argmax(axis=1)
        preds_4o = p_4model_opt.argmax(axis=1)
        preds_dc = p_dual_champ.argmax(axis=1)
        
        valid_cls = present_map.get(name, [0, 1, 2, 3])
        
        acc_3a, f1_3a, _ = evaluate_metrics(y, preds_3a, valid_cls)
        acc_3d, f1_3d, _ = evaluate_metrics(y, preds_3d, valid_cls)
        acc_3m, f1_3m, _ = evaluate_metrics(y, preds_3m, valid_cls)
        acc_4e, f1_4e, _ = evaluate_metrics(y, preds_4e, valid_cls)
        acc_4o, f1_4o, _ = evaluate_metrics(y, preds_4o, valid_cls)
        acc_dc, f1_dc, _ = evaluate_metrics(y, preds_dc, valid_cls)
        
        print(f"  Single 3A (ImageNet 56)       : Acc={acc_3a*100:.2f}% | Valid F1={f1_3a*100:.2f}%")
        print(f"  Single 3D (Stratified 33/33/33): Acc={acc_3d*100:.2f}% | Valid F1={f1_3d*100:.2f}%")
        print(f"  Prior 3-Model Weighted Ensemble: Acc={acc_3m*100:.2f}% | Valid F1={f1_3m*100:.2f}%")
        print(f"  >> 4-Model Equal Soft Ensemble : Acc={acc_4e*100:.2f}% | Valid F1={f1_4e*100:.2f}%")
        print(f"  >> 4-Model Weighted Soft (35/15/25/25): Acc={acc_4o*100:.2f}% | Valid F1={f1_4o*100:.2f}%")
        print(f"  >> Dual Champions (3A + 3D)   : Acc={acc_dc*100:.2f}% | Valid F1={f1_dc*100:.2f}%")
        
        dataset_results[name] = {
            "samples": len(y),
            "present_classes": [CLASS_NAMES[c] for c in valid_cls],
            "3A": {"acc": acc_3a, "f1": f1_3a},
            "3D": {"acc": acc_3d, "f1": f1_3d},
            "Prior_3Model_Ensemble": {"acc": acc_3m, "f1": f1_3m},
            "Equal_4Model_Ensemble": {"acc": acc_4e, "f1": f1_4e},
            "Weighted_4Model_Ensemble": {"acc": acc_4o, "f1": f1_4o},
            "Dual_Champions_3A_3D": {"acc": acc_dc, "f1": f1_dc}
        }
        
        pooled_probs_3a.append(p_3a)
        pooled_probs_3b.append(p_3b)
        pooled_probs_3c.append(p_3c)
        pooled_probs_3d.append(p_3d)
        pooled_targets.append(y)

    # -------------------------------------------------------------
    # MASTER POOLED TEST EVALUATION (ALL 4 DATASETS: N = 6,246)
    # -------------------------------------------------------------
    global_p_3a = np.concatenate(pooled_probs_3a, axis=0)
    global_p_3b = np.concatenate(pooled_probs_3b, axis=0)
    global_p_3c = np.concatenate(pooled_probs_3c, axis=0)
    global_p_3d = np.concatenate(pooled_probs_3d, axis=0)
    global_y    = np.concatenate(pooled_targets, axis=0)
    
    global_p_3m = (0.45 * global_p_3a + 0.20 * global_p_3b + 0.35 * global_p_3c)
    global_p_4e = (global_p_3a + global_p_3b + global_p_3c + global_p_3d) / 4.0
    global_p_4o = (0.35 * global_p_3a + 0.15 * global_p_3b + 0.25 * global_p_3c + 0.25 * global_p_3d)
    global_p_dc = (0.50 * global_p_3a + 0.50 * global_p_3d)
    
    # Hard Majority Voting across all 4 models
    discrete_votes = np.stack([
        global_p_3a.argmax(axis=1),
        global_p_3b.argmax(axis=1),
        global_p_3c.argmax(axis=1),
        global_p_3d.argmax(axis=1)
    ], axis=1) # Shape: (N, 4)
    # Take mode; on ties break with soft weighted
    g_preds_hard = []
    for i in range(len(global_y)):
        row = discrete_votes[i]
        counts = np.bincount(row, minlength=4)
        if (counts == counts.max()).sum() == 1:
            g_preds_hard.append(counts.argmax())
        else:
            g_preds_hard.append(global_p_4o[i].argmax())
    g_preds_hard = np.array(g_preds_hard)
    
    print("\n" + "=" * 75)
    print(f"MASTER GLOBAL POOLED BENCHMARK (N = {len(global_y)} SCANS ACROSS ALL 4 DATASETS)")
    print("=" * 75)
    
    candidate_predictions = [
        ("Phase 3A Single Champion", global_p_3a.argmax(axis=1)),
        ("Phase 3D Domain-Stratified", global_p_3d.argmax(axis=1)),
        ("Hard Majority 4-Model Voting", g_preds_hard),
        ("Prior 3-Model Ensemble (3A+3B+3C)", global_p_3m.argmax(axis=1)),
        ("Equal 4-Model Ensemble (3A+3B+3C+3D)", global_p_4e.argmax(axis=1)),
        ("Dual Champions (3A 50% + 3D 50%)", global_p_dc.argmax(axis=1)),
        ("Master Weighted 4-Model Ensemble", global_p_4o.argmax(axis=1))
    ]
    
    global_summary = {}
    for label, preds in candidate_predictions:
        acc = accuracy_score(global_y, preds)
        macro_f1 = f1_score(global_y, preds, average="macro", zero_division=0)
        weight_f1 = f1_score(global_y, preds, average="weighted", zero_division=0)
        global_summary[label] = {"acc": float(acc), "macro_f1": float(macro_f1), "weighted_f1": float(weight_f1)}
        print(f"{label:<38}: Top-1 Acc: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}% | Weighted F1: {weight_f1*100:.2f}%")

    # -------------------------------------------------------------
    # EPISTEMIC UNCERTAINTY & AUTONOMOUS REFERRAL TRIAGE
    # -------------------------------------------------------------
    print("\n[*] Evaluating Autonomous Clinical Referral Triage via Predictive Entropy H(x)...")
    eps = 1e-12
    p_safe = np.clip(global_p_4o, eps, 1.0)
    entropy = -np.sum(p_safe * np.log2(p_safe), axis=1) # Shannon Entropy in bits
    sorted_indices_by_uncertainty = np.argsort(entropy)[::-1]
    
    triage_results = {}
    referral_pcts = [0, 5, 10, 15, 20, 25, 30]
    best_preds = global_p_4o.argmax(axis=1)
    
    print("\n[+] Referral Triage Operating Characteristic:")
    print("    Referral Rate | Retained Scans | Autonomous Acc | Autonomous Macro F1")
    print("    " + "-" * 62)
    
    for r_pct in referral_pcts:
        n_refer = int(len(global_y) * (r_pct / 100.0))
        retained_idx = sorted_indices_by_uncertainty[n_refer:]
        y_ret = global_y[retained_idx]
        p_ret = best_preds[retained_idx]
        
        acc_ret = accuracy_score(y_ret, p_ret)
        f1_ret = f1_score(y_ret, p_ret, average="macro", zero_division=0)
        triage_results[f"{r_pct}%"] = {"retained": len(y_ret), "acc": float(acc_ret), "macro_f1": float(f1_ret)}
        print(f"        {r_pct:>2}%        |      {len(y_ret):>5}     |     {acc_ret*100:.2f}%     |      {f1_ret*100:.2f}%")
        
    # Plot Triage Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pts = referral_pcts
    y_acc = [triage_results[f"{r}%"]["acc"] * 100 for r in x_pts]
    y_f1  = [triage_results[f"{r}%"]["macro_f1"] * 100 for r in x_pts]
    
    ax.plot(x_pts, y_acc, 'b-o', linewidth=2.5, label='Autonomous Top-1 Accuracy (%)')
    ax.plot(x_pts, y_f1, 'g--s', linewidth=2.5, label='Autonomous Macro F1-Score (%)')
    ax.set_title('Master 4-Model Ensemble Clinical Referral Triage Curve\n(Shannon Predictive Entropy Thresholding)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Clinical Referral Rate to Ophthalmologist (%)', fontsize=11)
    ax.set_ylabel('Performance on Retained Cohort (%)', fontsize=11)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    curve_path = OUTPUT_DIR / "ensemble_4model_triage_referral_curve.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"[+] Saved referral triage curve -> {curve_path.name}")
    
    # Plot Master Confusion Matrix
    cm = confusion_matrix(global_y, best_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title(f'Master 4-Model Super-Ensemble Confusion Matrix (N = {len(global_y)})\nMacro F1: {global_summary["Master Weighted 4-Model Ensemble"]["macro_f1"]*100:.2f}% | Top-1 Acc: {global_summary["Master Weighted 4-Model Ensemble"]["acc"]*100:.2f}%', fontweight='bold')
    ax.set_ylabel('Ground Truth (Clinical Consensus)')
    ax.set_xlabel('Ensemble Predicted Pathology')
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "ensemble_4model_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved master 4-model confusion matrix -> {cm_path.name}")
    
    # Save full results JSON
    payload = {
        "benchmark_name": "Phase 3 Master 4-Model Super-Ensemble Battery",
        "models_evaluated": ["Phase 3A (56-stem)", "Phase 3B (64-stem)", "Phase 3C (112-stem)", "Phase 3D (Stratified 33/33/33)"],
        "total_test_scans": len(global_y),
        "dataset_breakdown": dataset_results,
        "global_pooled_comparison": global_summary,
        "referral_triage": triage_results
    }
    
    json_path = OUTPUT_DIR / "ensemble_4model_benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"[+] Saved comprehensive 4-model results -> {json_path.name}")
    print("=" * 75)
    print("OPTION A COMPLETED")
    print("=" * 75)

if __name__ == "__main__":
    main()
