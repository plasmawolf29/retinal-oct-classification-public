"""
Phase 3 Master Super-Ensemble & Uncertainty Triage Evaluation
=============================================================
1. Loads Phase 3A (56-stem), Phase 3B (64-stem), and Phase 3C (112-stem) models.
2. Evaluates:
   - Individual Models (3A, 3B, 3C)
   - Equal Soft-Voting Ensemble (w = 1/3, 1/3, 1/3)
   - Optimal Weighted Soft-Voting Ensemble (w_A=0.45, w_B=0.20, w_C=0.35)
3. Addresses Single-Dataset Class Mismatch:
   - Duke Test (N=621): Computes both raw 4-class and valid 3-class (DME, DRUSEN, NORMAL) metrics.
   - NEH Test (N=3,625): Computes both raw 4-class and valid 3-class (CNV, DRUSEN, NORMAL) metrics.
4. Epistemic Uncertainty & Autonomous Clinical Referral Triage:
   - Computes predictive entropy H(x).
   - Generates referral curve (accuracy at 0%, 5%, 10%, 15%, 20% referral).
5. Saves results to code/export/scratch/next-scripts/combined results/ensemble_benchmark_results.json.
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

# --- MODEL CONSTRUCTORS ---
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

def load_models():
    print("[*] Loading checkpoints into GPU memory...")
    p_3a = SCRIPT_DIR / "phase_3a_resnet18_224x224_imagenet_mega_best.pth"
    p_3b = SCRIPT_DIR / "phase_3b_resnet18_224x224_64stem_mega_best.pth"
    p_3c = SCRIPT_DIR / "phase_3c_resnet18_224x224_112stem_mega_best.pth"
    
    m_3a = build_model_3a().to(DEVICE)
    m_3a.load_state_dict(torch.load(p_3a, map_location=DEVICE))
    m_3a.eval()
    
    m_3b = build_model_3b().to(DEVICE)
    m_3b.load_state_dict(torch.load(p_3b, map_location=DEVICE))
    m_3b.eval()
    
    m_3c = build_model_3c().to(DEVICE)
    m_3c.load_state_dict(torch.load(p_3c, map_location=DEVICE))
    m_3c.eval()
    
    print("[+] All 3 champion models loaded successfully.")
    return m_3a, m_3b, m_3c

def get_all_test_loaders():
    # 1. External loaders (Kermany, Duke, NEH)
    loaders = get_benchmark_dataloaders(image_size=224, batch_size=32, in_channels=1)
    
    # 2. Add OCTMNIST In-Domain Test Loader
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
    print("=" * 70)
    print("PHASE 3 ENSEMBLE, DATASET AUDIT & CLINICAL REFERRAL BENCHMARK")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    m_3a, m_3b, m_3c = load_models()
    loaders = get_all_test_loaders()
    
    # Present classes per dataset
    present_map = {
        "duke": [1, 2, 3],        # DME, DRUSEN, NORMAL (CNV is absent)
        "neh": [0, 2, 3],         # CNV, DRUSEN, NORMAL (DME is absent)
        "kermany": [0, 1, 2, 3],  # All 4 classes
        "octmnist": [0, 1, 2, 3]  # All 4 classes
    }
    
    dataset_results = {}
    
    # Storage for pooled predictions
    pooled_probs_3a = []
    pooled_probs_3b = []
    pooled_probs_3c = []
    pooled_targets  = []
    
    for name, loader in loaders.items():
        print(f"\n--- Inferencing Benchmark: {name.upper()} (N={len(loader.dataset)}) ---")
        p_3a, y = predict_dataloader(m_3a, loader)
        p_3b, _ = predict_dataloader(m_3b, loader)
        p_3c, _ = predict_dataloader(m_3c, loader)
        
        # Ensembles
        p_equal = (p_3a + p_3b + p_3c) / 3.0
        p_opt   = (0.45 * p_3a + 0.20 * p_3b + 0.35 * p_3c)
        
        preds_3a = p_3a.argmax(axis=1)
        preds_3b = p_3b.argmax(axis=1)
        preds_3c = p_3c.argmax(axis=1)
        preds_eq = p_equal.argmax(axis=1)
        preds_op = p_opt.argmax(axis=1)
        
        valid_cls = present_map.get(name, [0, 1, 2, 3])
        
        # Evaluate valid-class macro F1
        acc_3a, f1_3a, _ = evaluate_metrics(y, preds_3a, valid_cls)
        acc_3b, f1_3b, _ = evaluate_metrics(y, preds_3b, valid_cls)
        acc_3c, f1_3c, _ = evaluate_metrics(y, preds_3c, valid_cls)
        acc_eq, f1_eq, _ = evaluate_metrics(y, preds_eq, valid_cls)
        acc_op, f1_op, _ = evaluate_metrics(y, preds_op, valid_cls)
        
        print(f"  3A (ImageNet 56) : Acc={acc_3a*100:.2f}% | Valid-Class Macro F1={f1_3a*100:.2f}%")
        print(f"  3B (Prog 64)     : Acc={acc_3b*100:.2f}% | Valid-Class Macro F1={f1_3b*100:.2f}%")
        print(f"  3C (Prog 112)    : Acc={acc_3c*100:.2f}% | Valid-Class Macro F1={f1_3c*100:.2f}%")
        print(f"  >> Equal Ensemble: Acc={acc_eq*100:.2f}% | Valid-Class Macro F1={f1_eq*100:.2f}%")
        print(f"  >> Weighted Soft : Acc={acc_op*100:.2f}% | Valid-Class Macro F1={f1_op*100:.2f}%")
        
        dataset_results[name] = {
            "samples": len(y),
            "present_classes": [CLASS_NAMES[c] for c in valid_cls],
            "3A": {"acc": acc_3a, "f1": f1_3a},
            "3B": {"acc": acc_3b, "f1": f1_3b},
            "3C": {"acc": acc_3c, "f1": f1_3c},
            "Equal_Ensemble": {"acc": acc_eq, "f1": f1_eq},
            "Weighted_Ensemble": {"acc": acc_op, "f1": f1_op}
        }
        
        # Collect for global pool
        pooled_probs_3a.append(p_3a)
        pooled_probs_3b.append(p_3b)
        pooled_probs_3c.append(p_3c)
        pooled_targets.append(y)

    # -------------------------------------------------------------
    # MASTER POOLED TEST EVALUATION (ALL 4 DATASETS: N = 6,246)
    # -------------------------------------------------------------
    global_p_3a = np.concatenate(pooled_probs_3a, axis=0)
    global_p_3b = np.concatenate(pooled_probs_3b, axis=0)
    global_p_3c = np.concatenate(pooled_probs_3c, axis=0)
    global_y    = np.concatenate(pooled_targets, axis=0)
    
    global_p_equal = (global_p_3a + global_p_3b + global_p_3c) / 3.0
    global_p_opt   = (0.45 * global_p_3a + 0.20 * global_p_3b + 0.35 * global_p_3c)
    
    g_preds_3a = global_p_3a.argmax(axis=1)
    g_preds_3b = global_p_3b.argmax(axis=1)
    g_preds_3c = global_p_3c.argmax(axis=1)
    g_preds_eq = global_p_equal.argmax(axis=1)
    g_preds_op = global_p_opt.argmax(axis=1)
    
    print("\n" + "=" * 70)
    print(f"MASTER GLOBAL POOLED BENCHMARK (N = {len(global_y)} SCANS ACROSS ALL 4 DATASETS)")
    print("=" * 70)
    
    for label, preds in [
        ("Phase 3A (ImageNet 56)", g_preds_3a),
        ("Phase 3B (Prog 64)", g_preds_3b),
        ("Phase 3C (Prog 112)", g_preds_3c),
        ("Equal Soft Ensemble (3A+3B+3C)", g_preds_eq),
        ("Optimal Weighted Soft Ensemble", g_preds_op),
    ]:
        acc = accuracy_score(global_y, preds)
        macro_f1 = f1_score(global_y, preds, average="macro", zero_division=0)
        weight_f1 = f1_score(global_y, preds, average="weighted", zero_division=0)
        print(f"{label:<32}: Top-1 Acc: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}% | Weighted F1: {weight_f1*100:.2f}%")

    # -------------------------------------------------------------
    # EPISTEMIC UNCERTAINTY & AUTONOMOUS REFERRAL TRIAGE
    # -------------------------------------------------------------
    print("\n[*] Evaluating Autonomous Clinical Referral Triage via Predictive Entropy H(x)...")
    # Predictive Entropy in bits
    eps = 1e-12
    p_safe = np.clip(global_p_opt, eps, 1.0)
    entropy = -np.sum(p_safe * np.log2(p_safe), axis=1)
    
    referral_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    triage_results = []
    
    for rate in referral_rates:
        n_refer = int(len(entropy) * rate)
        if n_refer == 0:
            retained_idx = np.arange(len(entropy))
        else:
            cutoff = np.quantile(entropy, 1.0 - rate)
            retained_idx = np.where(entropy <= cutoff)[0]
            
        ret_acc = accuracy_score(global_y[retained_idx], g_preds_op[retained_idx])
        ret_f1  = f1_score(global_y[retained_idx], g_preds_op[retained_idx], average="macro", zero_division=0)
        triage_results.append({
            "referral_rate": rate,
            "retained_samples": len(retained_idx),
            "retained_accuracy": ret_acc,
            "retained_macro_f1": ret_f1
        })
        print(f"  Referral: {rate*100:4.1f}% | Retained: {len(retained_idx):4d}/{len(entropy)} | Retained Acc: {ret_acc*100:.2f}% | Retained Macro F1: {ret_f1*100:.2f}%")

    # Plot Referral Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    rates_pct = [r["referral_rate"] * 100 for r in triage_results]
    accs_pct  = [r["retained_accuracy"] * 100 for r in triage_results]
    f1s_pct   = [r["retained_macro_f1"] * 100 for r in triage_results]
    
    ax.plot(rates_pct, accs_pct, marker="o", color="#2b5c8f", lw=2.5, label="Retained Accuracy (%)")
    ax.plot(rates_pct, f1s_pct, marker="s", color="#348a48", lw=2.5, label="Retained Macro F1 (%)")
    ax.set_xlabel("Referral to Human Ophthalmologist (%)", fontsize=12)
    ax.set_ylabel("Diagnostic Score on Retained Scans (%)", fontsize=12)
    ax.set_title("Autonomous Clinical Triage: Performance vs. Referral Rate", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    triage_plot_path = OUTPUT_DIR / "ensemble_triage_referral_curve.png"
    plt.savefig(triage_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved referral curve plot: {triage_plot_path.name}")

    # Plot Confusion Matrix of Optimal Ensemble
    cm = confusion_matrix(global_y, g_preds_op)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Optimal Soft Ensemble Confusion Matrix (N={len(global_y)})", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Condition")
    plt.ylabel("True Pathology")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "ensemble_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[+] Saved ensemble confusion matrix: {cm_path.name}")

    # Save summary json
    final_output = {
        "dataset_results": dataset_results,
        "global_eval": {
            "total_samples": len(global_y),
            "3A": {"acc": float(accuracy_score(global_y, g_preds_3a)), "macro_f1": float(f1_score(global_y, g_preds_3a, average="macro"))},
            "3B": {"acc": float(accuracy_score(global_y, g_preds_3b)), "macro_f1": float(f1_score(global_y, g_preds_3b, average="macro"))},
            "3C": {"acc": float(accuracy_score(global_y, g_preds_3c)), "macro_f1": float(f1_score(global_y, g_preds_3c, average="macro"))},
            "equal_ensemble": {"acc": float(accuracy_score(global_y, g_preds_eq)), "macro_f1": float(f1_score(global_y, g_preds_eq, average="macro"))},
            "weighted_ensemble": {"acc": float(accuracy_score(global_y, g_preds_op)), "macro_f1": float(f1_score(global_y, g_preds_op, average="macro"))}
        },
        "triage_results": triage_results
    }
    
    with open(OUTPUT_DIR / "ensemble_benchmark_results.json", "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"[+] Saved ensemble benchmark results to JSON.")

if __name__ == "__main__":
    main()
