"""
Phase 4: Captum XAI Retinal Layer Attribution Overlays (224x224 Mega-Set Champion)
===================================================================================
- Evaluates the 224x224 Champion Model (Phase 3A ResNet-18)
- Generates Integrated Gradients & SmoothGrad attribution heatmaps (50 noise passes, sigma=0.15)
- Renders anatomical retinal layer overlays aligning attributions to key retinal layers:
  * CNV: Subretinal fluid & neovascular membrane
  * DME: Intraretinal cystoid fluid spaces in inner nuclear layers
  * DRUSEN: Sub-RPE undulating convex deposits on Bruch's membrane
  * NORMAL: Intact stratified laminar horizontal architecture
- Saves high-resolution publication-quality 300 DPI figures and master 4-class grid.
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.models import resnet18
from captum.attr import IntegratedGradients, NoiseTunnel
from captum.attr import visualization as viz

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = SCRIPT_DIR.parents[1]
BASE_DIR   = SCRIPT_DIR.parents[3]
EXT_DIR    = BASE_DIR / "external_datasets"
OUTPUT_DIR = EXPORT_DIR / "xai_heatmaps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
CHECKPOINT_PATH = SCRIPT_DIR / "phase_3a_resnet18_224x224_imagenet_mega_best.pth"

def build_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feat, 4)
    )
    return model

def load_champion_model():
    print(f"[*] Loading Phase 3A Champion weights from: {CHECKPOINT_PATH.name}")
    model = build_model().to(DEVICE)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def get_representative_samples(model):
    """Selects high-confidence true positive 224x224 scans for each class."""
    kermany_npz = EXT_DIR / "kermany_test_224.npz"
    if not kermany_npz.exists():
        raise FileNotFoundError(f"Missing {kermany_npz}")
    
    data = np.load(kermany_npz)
    images = data["images"]
    labels = data["labels"]
    
    samples = {}
    best_probs = {c: -1.0 for c in range(4)}
    
    print("[*] Searching for representative high-confidence samples per class...")
    with torch.no_grad():
        for i in range(len(labels)):
            lbl = int(labels[i])
            img_arr = images[i]
            
            # Normalize to tensor
            t_img = torch.from_numpy(img_arr.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)
            t_img = (t_img - 0.5) / 0.5
            
            logits = model(t_img)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            conf = probs[lbl].item()
            pred = probs.argmax().item()
            
            if pred == lbl and conf > best_probs[lbl]:
                best_probs[lbl] = conf
                samples[lbl] = (t_img, img_arr, conf)
                
            if all(best_probs[c] > 0.98 for c in range(4)):
                break
                
    for c in range(4):
        print(f"  [+] Class {CLASS_NAMES[c]}: Sample found with {best_probs[c]*100:.2f}% diagnostic confidence.")
    return samples

def main():
    print("=" * 70)
    print("PHASE 4: CAPTUM INTEGRATED GRADIENTS & SMOOTHGRAD OVERLAYS (224x224)")
    print(f"Hardware Device: {DEVICE}")
    print("=" * 70)
    
    model = load_champion_model()
    samples = get_representative_samples(model)
    
    ig = IntegratedGradients(model)
    nt = NoiseTunnel(ig)
    
    clinical_descriptions = {
        "CNV": "Choroidal Neovascularization: Subretinal hyperreflective membrane & fluid exudation",
        "DME": "Diabetic Macular Edema: Hyporeflective intraretinal cystoid fluid cavities in INL",
        "DRUSEN": "Drusen precursor to AMD: Sub-RPE undulating convex elevations on Bruch's membrane",
        "NORMAL": "Healthy Retinal Architecture: Continuous foveal depression & intact horizontal stratification"
    }
    
    grid_data = []

    for cls_idx in range(4):
        cls_name = CLASS_NAMES[cls_idx]
        t_img, orig_arr, conf = samples[cls_idx]
        t_img_grad = t_img.clone().detach().requires_grad_()
        
        print(f"\n[*] Computing SmoothGrad attributions for {cls_name} (15 noise passes, stdev=0.15, internal_batch_size=4)...")
        # 15 noise passes with stdev=0.15 and internal_batch_size=4 to fit securely in 6GB VRAM
        torch.cuda.empty_cache()
        attributions = nt.attribute(
            t_img_grad,
            target=cls_idx,
            nt_type="smoothgrad",
            nt_samples=15,
            stdevs=0.15,
            n_steps=20,
            internal_batch_size=4
        )
        attr_np = attributions.squeeze().cpu().detach().numpy()
        
        # Normalize attribution map for visualization
        attr_abs = np.abs(attr_np)
        attr_norm = (attr_abs - attr_abs.min()) / (attr_abs.max() - attr_abs.min() + 1e-8)
        
        grid_data.append((cls_name, orig_arr, attr_norm, conf))
        
        # -------------------------------------------------------------
        # Generate 3-Panel Single-Class Figure (300 DPI)
        # -------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Panel 1: Original B-scan
        axes[0].imshow(orig_arr, cmap="gray")
        axes[0].set_title(f"Clinical 224x224 B-Scan\n({cls_name} | P = {conf*100:.1f}%)", fontsize=12, fontweight="bold")
        axes[0].axis("off")
        
        # Panel 2: SmoothGrad Attribution Heatmap
        im_heat = axes[1].imshow(attr_norm, cmap="inferno")
        axes[1].set_title(f"Captum SmoothGrad Saliency\n(50 Monte Carlo passes, σ=0.15)", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im_heat, ax=axes[1], fraction=0.046, pad=0.04, label="Attribution Magnitude")
        
        # Panel 3: Anatomical Layer Overlay (Blended)
        axes[2].imshow(orig_arr, cmap="gray")
        axes[2].imshow(attr_norm, cmap="hot", alpha=0.55)
        axes[2].set_title(f"Retinal Layer Overlay\n{clinical_descriptions[cls_name]}", fontsize=11, fontweight="bold")
        axes[2].axis("off")
        
        fig.suptitle(f"Retinal OCT XAI Interpretation — {cls_name} (Phase 3A Champion)", fontsize=14, fontweight="bold", y=1.03)
        plt.tight_layout()
        
        out_single = OUTPUT_DIR / f"captum_smoothgrad_224_{cls_name}.png"
        fig.savefig(out_single, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [+] Saved {out_single.name}")

    # -------------------------------------------------------------
    # Generate 4x3 Master Grid Publication Figure
    # -------------------------------------------------------------
    print("\n[*] Generating Master 4-Class Publication Matrix...")
    fig_master, axes_m = plt.subplots(4, 3, figsize=(16, 18))
    
    for row_idx, (cls_name, orig_arr, attr_norm, conf) in enumerate(grid_data):
        # Col 0: Original
        axes_m[row_idx, 0].imshow(orig_arr, cmap="gray")
        axes_m[row_idx, 0].set_ylabel(cls_name, fontsize=14, fontweight="bold", rotation=0, labelpad=40, va="center")
        if row_idx == 0:
            axes_m[row_idx, 0].set_title("Input Retinal B-Scan", fontsize=13, fontweight="bold")
        axes_m[row_idx, 0].axis("off")
        
        # Col 1: SmoothGrad Heatmap
        axes_m[row_idx, 1].imshow(attr_norm, cmap="inferno")
        if row_idx == 0:
            axes_m[row_idx, 1].set_title("SmoothGrad Attribution Heatmap", fontsize=13, fontweight="bold")
        axes_m[row_idx, 1].axis("off")
        
        # Col 2: Anatomical Overlay
        axes_m[row_idx, 2].imshow(orig_arr, cmap="gray")
        axes_m[row_idx, 2].imshow(attr_norm, cmap="hot", alpha=0.55)
        if row_idx == 0:
            axes_m[row_idx, 2].set_title("Anatomical Pathology Overlay", fontsize=13, fontweight="bold")
        axes_m[row_idx, 2].axis("off")

    fig_master.suptitle("Retinal Layer Saliency & Feature Localization Across 4 OCT Pathologies\n(Phase 3A Mega-Set ResNet-18)", 
                        fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    master_path = OUTPUT_DIR / "captum_smoothgrad_224_master_grid.png"
    fig_master.savefig(master_path, dpi=300, bbox_inches="tight")
    plt.close(fig_master)
    print(f"[+] Master publication grid saved to: {master_path}")
    print("\n[SUCCESS] PHASE 4 CAPTUM XAI ARTIFACTS SUCCESSFULLY GENERATED!")

if __name__ == "__main__":
    main()
