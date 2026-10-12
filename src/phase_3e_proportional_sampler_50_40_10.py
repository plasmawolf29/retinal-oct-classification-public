"""
PHASE 3E: Domain-Stratified Sampling Test (50% Heidelberg / 40% NEH Topcon / 10% Duke Bioptigen)
=================================================================================================
Clinical & Engineering Motivation:
  In Phase 3D, enforcing strict 33.3% domain parity proved that forced Topcon exposure lifted NEH F1 (+1.16%)
  and propelled external Kermany generalization to a record 91.56% Macro F1 with 99.97% specificity against DME.
  However, 33.3% parity caused two structural issues:
    1. Duke (N=2,610) was oversampled ~14.4x/epoch, yielding redundant gradient updates.
    2. Heidelberg (N=97,477) was starved down to 33.3%, dropping in-domain F1 from 85.76% to 84.58%.

This Phase 3E experiment implements the User's Proportional Dampening Distribution:
  - 50.0% Heidelberg Spectralis (retains 56,000+ unique patient variations per epoch)
  - 40.0% NEH Topcon (quadruples exposure vs natural 11.6%, maximizing gradient signal on noisy optics)
  - 10.0% Duke Bioptigen (limits repetition to ~4.3x/epoch, preventing overfitting on Duke's clean contrast)

Architecture & Training Configuration:
  - ResNet-18 ImageNet Transfer (56x56 feature maps)
  - Input: Grayscale single-channel conv1 (7x7, stride=2)
  - Resolution: 224x224
  - Hardware safe: batch_size=32, num_workers=0, FP32 native (AMP=False), pin_memory=True
  - Target: Push NEH Topcon >= 70.0% while restoring In-Domain F1 >= 85.5% and maintaining Duke >= 98.5%.
"""

import os
import sys
import time
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from medmnist import OCTMNIST
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_recall_fscore_support, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

# ==============================================================================
# CONFIGURATION
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
SCRATCH_DIR = SCRIPT_DIR.parent
EXPORT_DIR = SCRATCH_DIR.parent
BASE_DIR = SCRIPT_DIR.parents[3]
EXT_DIR = BASE_DIR / "external_datasets"

CONFIG = {
    "name": "Phase 3E Proportional Dampened ResNet18 (50/40/10)",
    "phase": "3E",
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 15,
    "patience": 5,
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "dropout": 0.3,
    "label_smoothing": 0.1,
    "num_tta_passes": 5,
    "seed": 42,
    # Domain sampling target quotas:
    "domain_quotas": {
        "heidelberg": 0.50, # 50%
        "neh_topcon": 0.40, # 40%
        "duke":       0.10  # 10%
    },
    "model_save_path": SCRIPT_DIR / "phase_3e_resnet18_224x224_50_40_10_best.pth",
    "results_save_path": SCRIPT_DIR / "phase_3e_224_results.json",
    "curves_save_path": SCRIPT_DIR / "phase_3e_224_learning_curves.png",
    "cm_save_path": SCRIPT_DIR / "phase_3e_224_cm.png"
}

CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]
NUM_CLASSES = 4

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class SmoothFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.eps = label_smoothing
        self.alpha = alpha

    def forward(self, logits, targets):
        c = logits.size(1)
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
        smooth_targets = one_hot * (1.0 - self.eps) + (self.eps / c)

        focal_weight = torch.pow(1.0 - p, self.gamma)
        loss = -focal_weight * smooth_targets * log_p

        if self.alpha is not None:
            alpha_tensor = self.alpha.to(logits.device)
            loss = loss * alpha_tensor.unsqueeze(0)

        return loss.sum(dim=1).mean()

def build_model(dropout_p=0.3):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)

    old_conv = model.conv1
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
    model.conv1 = new_conv

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(num_features, NUM_CLASSES)
    )
    return model

class ArrayDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        lbl = self.labels[idx]
        img_pil = T.functional.to_pil_image(img)
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, int(lbl.item() if hasattr(lbl, "item") else lbl)

class ProportionalStratifiedDataset(Dataset):
    """
    Unified training dataset enforcing exact user-specified sampling ratios:
      - Heidelberg/OCTMNIST : 50%
      - NEH Topcon          : 40%
      - Duke Bioptigen      : 10%
    """
    def __init__(self, oct_dataset, duke_npz_path, neh_npz_path, transform=None):
        self.oct_ds = oct_dataset
        self.transform = transform

        print(f"[*] Loading external training caches for 50/40/10 proportional sampling...")
        duke_data = np.load(duke_npz_path)
        self.duke_imgs = duke_data["images"]
        self.duke_lbls = duke_data["labels"]

        neh_data = np.load(neh_npz_path)
        self.neh_imgs = neh_data["images"]
        self.neh_lbls = neh_data["labels"]

        self.n_oct = len(self.oct_ds)
        self.n_duke = len(self.duke_lbls)
        self.n_neh = len(self.neh_lbls)
        self.total_len = self.n_oct + self.n_duke + self.n_neh

        print(f"[+] Proportional Sampling Pool: {self.total_len} scans")
        print(f"    - Domain 0 (Heidelberg/OCTMNIST) : {self.n_oct:,} scans -> Target Quota: 50.0%")
        print(f"    - Domain 1 (NEH Topcon)          : {self.n_neh:,} scans -> Target Quota: 40.0%")
        print(f"    - Domain 2 (Duke Bioptigen)      : {self.n_duke:,} scans -> Target Quota: 10.0%")

    def __len__(self):
        return self.total_len

    def get_proportional_weights(self):
        """
        Calculates per-sample sampling weights enforcing 50% Heidelberg / 40% Topcon / 10% Duke.
        weight_i = target_quota_d / n_d
        """
        w_oct  = 0.50 / self.n_oct
        w_neh  = 0.40 / self.n_neh
        w_duke = 0.10 / self.n_duke

        weights = np.empty(self.total_len, dtype=np.float64)
        weights[:self.n_oct] = w_oct
        weights[self.n_oct:self.n_oct + self.n_duke] = w_duke
        weights[self.n_oct + self.n_duke:] = w_neh
        return torch.from_numpy(weights).double()

    def __getitem__(self, idx):
        if idx < self.n_oct:
            img, lbl = self.oct_ds[idx]
            return img, int(lbl.item() if hasattr(lbl, "item") else lbl)
        elif idx < self.n_oct + self.n_duke:
            d_idx = idx - self.n_oct
            img = self.duke_imgs[d_idx]
            lbl = self.duke_lbls[d_idx]
        else:
            n_idx = idx - (self.n_oct + self.n_duke)
            img = self.neh_imgs[n_idx]
            lbl = self.neh_lbls[n_idx]

        img_pil = T.functional.to_pil_image(img)
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, int(lbl.item() if hasattr(lbl, "item") else lbl)

def plot_learning_curves(history, save_path):
    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, history["train_loss"], 'b-o', label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, history["val_loss"], 'r--s', label='Val Loss', linewidth=2)
    ax1.set_title('Proportional (50/40/10) Loss Curve', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.plot(epochs_range, [f * 100 for f in history["val_f1"]], 'g-^', label='Val Macro F1 (%)', linewidth=2)
    ax2.set_title('Validation Macro F1 Score', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Macro F1 (%)', fontsize=11)
    ax2.legend(loc='lower right')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle('Phase 3E Proportional Dampened Learning Curves (50% Hdbg / 40% NEH / 10% Duke)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved learning curves -> {save_path.name}")

def run_tta_evaluation(model, dataset, num_passes, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    all_pass_probs = []
    with torch.no_grad():
        for _ in range(num_passes):
            pass_probs = []
            for imgs, _ in loader:
                imgs = imgs.to(device, non_blocking=True)
                outputs = model(imgs)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                pass_probs.append(probs)
            all_pass_probs.append(np.concatenate(pass_probs, axis=0))
    avg_probs = np.mean(all_pass_probs, axis=0)
    preds = np.argmax(avg_probs, axis=1)
    return preds, avg_probs

def main():
    set_seed(CONFIG["seed"])
    print("=" * 75)
    print("PHASE 3E: PROPORTIONAL DAMPENED SAMPLING (50% HEIDELBERG / 40% TOPCON / 10% DUKE)")
    print(f"Device: {CONFIG['device']} | Batch Size: {CONFIG['batch_size']} | Workers: {CONFIG['num_workers']}")
    print(f"AMP: DISABLED (FP32 native) | Image Size: {CONFIG['img_size']}x{CONFIG['img_size']}")
    print(f"Target Epochs: {CONFIG['epochs']} | Patience: {CONFIG['patience']}")
    print("=" * 75)

    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.96, 1.04)),
        T.ColorJitter(contrast=(0.85, 1.15), brightness=(0.9, 1.1)),
        T.Resize((CONFIG["img_size"], CONFIG["img_size"]), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15), value='random'),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    eval_transform = T.Compose([
        T.Resize((CONFIG["img_size"], CONFIG["img_size"]), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    tta_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(contrast=(0.90, 1.10)),
        T.RandomAffine(degrees=4, translate=(0.02, 0.02)),
        T.Resize((CONFIG["img_size"], CONFIG["img_size"]), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])

    print("[*] Loading base OCTMNIST splits (native 64x64 cache)...")
    oct_train = OCTMNIST(split='train', transform=train_transform, download=False, size=64)
    oct_val   = OCTMNIST(split='val',   transform=eval_transform,  download=False, size=64)
    oct_test  = OCTMNIST(split='test',  transform=eval_transform,  download=False, size=64)

    duke_tr_npz = EXT_DIR / "duke_train_224.npz"
    neh_tr_npz  = EXT_DIR / "neh_train_224.npz"

    stratified_train_ds = ProportionalStratifiedDataset(oct_train, duke_tr_npz, neh_tr_npz, transform=train_transform)

    # WeightedRandomSampler with 50/40/10 proportional quotas
    proportional_weights = stratified_train_ds.get_proportional_weights()
    sampler = WeightedRandomSampler(
        weights=proportional_weights,
        num_samples=len(stratified_train_ds),
        replacement=True
    )

    train_loader = DataLoader(
        stratified_train_ds,
        batch_size=CONFIG["batch_size"],
        sampler=sampler,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        oct_val,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    model = build_model(dropout_p=CONFIG["dropout"]).to(CONFIG["device"])
    criterion = SmoothFocalLoss(gamma=2.0, label_smoothing=CONFIG["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "epoch_times": []}

    total_training_start = time.time()
    print(f"\n--- COMMENCING PROPORTIONAL TRAINING ({CONFIG['epochs']} EPOCHS MAX, PATIENCE={CONFIG['patience']}) ---")

    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        total_batches = len(train_loader)
        log_interval = 250
        slow_speed_count = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader, 1):
            imgs = imgs.to(CONFIG["device"], non_blocking=True)
            labels = labels.squeeze().long().to(CONFIG["device"], non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item() * imgs.size(0)
            running_loss += batch_loss
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            if batch_idx % log_interval == 0 or batch_idx == total_batches:
                elapsed_batch = time.time() - epoch_start
                img_per_sec = (batch_idx * CONFIG["batch_size"]) / max(elapsed_batch, 1e-4)
                curr_loss = running_loss / total_train
                curr_acc = (correct_train / total_train) * 100.0

                if img_per_sec < 45.0:
                    slow_speed_count += 1
                else:
                    slow_speed_count = 0

                if slow_speed_count >= 5:
                    print(f"\n[!] SPEED WARNING: Performance sustained at {img_per_sec:.1f} img/s below expected floor.")

                print(f"Epoch [{epoch+1:02d}/{CONFIG['epochs']:02d}] | Batch [{batch_idx:04d}/{total_batches:04d}] "
                      f"| Loss: {curr_loss:.4f} | Acc: {curr_acc:.2f}% | Speed: {img_per_sec:.1f} img/s")

        epoch_time = time.time() - epoch_start
        train_loss = running_loss / total_train
        history["train_loss"].append(float(train_loss))
        history["epoch_times"].append(float(epoch_time))

        # In-domain validation
        model.eval()
        val_loss = 0.0
        val_targets, val_preds = [], []

        with torch.no_grad():
            for val_imgs, val_lbls in val_loader:
                val_imgs = val_imgs.to(CONFIG["device"], non_blocking=True)
                val_lbls = val_lbls.squeeze().long().to(CONFIG["device"], non_blocking=True)
                val_outs = model(val_imgs)
                v_loss = criterion(val_outs, val_lbls)
                val_loss += v_loss.item() * val_imgs.size(0)

                val_preds.extend(val_outs.argmax(dim=1).cpu().numpy())
                val_targets.extend(val_lbls.cpu().numpy())

        val_loss /= len(oct_val)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        val_acc = accuracy_score(val_targets, val_preds)
        history["val_loss"].append(float(val_loss))
        history["val_f1"].append(float(val_f1))

        scheduler.step()

        print(f"\n>>> Epoch [{epoch+1:02d}/{CONFIG['epochs']:02d}] Summary ({epoch_time:.1f}s) <<<")
        print(f"    Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro F1: {val_f1*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"    [★] New Best Val Macro F1 ({best_val_f1*100:.2f}%) -> Saved Model to {CONFIG['model_save_path'].name}")
        else:
            patience_counter += 1
            print(f"    [-] Patience: {patience_counter}/{CONFIG['patience']}")

        if patience_counter >= CONFIG["patience"]:
            print(f"\n[!] Early stopping triggered at epoch {epoch+1}.")
            break

    total_training_time = time.time() - total_training_start
    print(f"\n[+] Total Training Completed in {total_training_time/60:.2f} minutes.")
    plot_learning_curves(history, CONFIG["curves_save_path"])

    # ==============================================================================
    # 5-PASS TTA BENCHMARK BATTERY
    # ==============================================================================
    print("\n" + "=" * 75)
    print("EXECUTION: 5-PASS TEST-TIME AUGMENTATION (TTA) BENCHMARK BATTERY")
    print("=" * 75)
    model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=CONFIG["device"]))
    model.eval()

    duke_te_data = np.load(EXT_DIR / "duke_test_224.npz")
    neh_te_data  = np.load(EXT_DIR / "neh_test_224.npz")
    kermany_data = np.load(EXT_DIR / "kermany_test_224.npz")

    benchmarks = {
        "OCTMNIST_Test_InDomain": {
            "dataset": OCTMNIST(split='test', transform=tta_transform, download=False, size=64),
            "active_classes": [0, 1, 2, 3]
        },
        "Kermany_Cell2018_External": {
            "dataset": ArrayDataset(kermany_data["images"], kermany_data["labels"], transform=tta_transform),
            "active_classes": [0, 1, 2, 3]
        },
        "Duke_Bioptigen_Test": {
            "dataset": ArrayDataset(duke_te_data["images"], duke_te_data["labels"], transform=tta_transform),
            "active_classes": [1, 2, 3]
        },
        "NEH_Topcon_Test": {
            "dataset": ArrayDataset(neh_te_data["images"], neh_te_data["labels"], transform=tta_transform),
            "active_classes": [0, 2, 3]
        }
    }

    benchmark_results = {}
    for bench_name, bench_info in benchmarks.items():
        print(f"\n[*] Evaluating Benchmark: {bench_name} ({len(bench_info['dataset'])} scans)...")
        preds, probs = run_tta_evaluation(
            model, bench_info["dataset"], CONFIG["num_tta_passes"],
            CONFIG["batch_size"], CONFIG["device"]
        )
        ds = bench_info["dataset"]
        if hasattr(ds, "labels"):
            targets = np.array(ds.labels).squeeze()
        else:
            targets = np.array([lbl for _, lbl in ds]).squeeze()

        active_cls = bench_info["active_classes"]
        acc = accuracy_score(targets, preds)
        bal_acc = balanced_accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average='macro', labels=active_cls, zero_division=0)
        prec, rec, f1_per_class, _ = precision_recall_fscore_support(targets, preds, average=None, labels=[0, 1, 2, 3], zero_division=0)

        false_alarms = None
        if bench_name == "Duke_Bioptigen_Test":
            cnv_hallucinations = int((preds == 0).sum())
            false_alarms = {
                "absent_class": "CNV",
                "false_alarms": cnv_hallucinations,
                "total_scans": len(targets),
                "specificity": float(1.0 - (cnv_hallucinations / len(targets)))
            }
            print(f"    Negative Control (CNV Specificity): {false_alarms['specificity']*100:.2f}% ({cnv_hallucinations}/{len(targets)} alarms)")
        elif bench_name == "NEH_Topcon_Test":
            dme_hallucinations = int((preds == 1).sum())
            false_alarms = {
                "absent_class": "DME",
                "false_alarms": dme_hallucinations,
                "total_scans": len(targets),
                "specificity": float(1.0 - (dme_hallucinations / len(targets)))
            }
            print(f"    Negative Control (DME Specificity): {false_alarms['specificity']*100:.2f}% ({dme_hallucinations}/{len(targets)} alarms)")

        print(f"    Top-1 Accuracy : {acc * 100:.2f}%")
        print(f"    Balanced Acc   : {bal_acc * 100:.2f}%")
        print(f"    Site-Valid F1  : {macro_f1 * 100:.2f}%")
        print(f"    Per-Class Rec  : CNV={rec[0]*100:.1f}%, DME={rec[1]*100:.1f}%, DRUSEN={rec[2]*100:.1f}%, NORMAL={rec[3]*100:.1f}%")

        benchmark_results[bench_name] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "site_valid_macro_f1": float(macro_f1),
            "per_class_recall": {CLASS_NAMES[i]: float(rec[i]) for i in range(4)},
            "per_class_precision": {CLASS_NAMES[i]: float(prec[i]) for i in range(4)},
            "per_class_f1": {CLASS_NAMES[i]: float(f1_per_class[i]) for i in range(4)},
            "negative_control": false_alarms
        }

        if bench_name == "OCTMNIST_Test_InDomain":
            cm = confusion_matrix(targets, preds)
            plt.figure(figsize=(8, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
            plt.title(f"Phase 3E Proportional (50/40/10) Confusion Matrix\nMacro F1: {macro_f1*100:.2f}% | Bal Acc: {bal_acc*100:.2f}%", fontweight='bold')
            plt.ylabel('Ground Truth')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig(CONFIG["cm_save_path"], dpi=300)
            plt.close()
            print(f"[+] Saved in-domain confusion matrix -> {CONFIG['cm_save_path'].name}")

    final_payload = {
        "model": CONFIG["name"],
        "phase": CONFIG["phase"],
        "resolution": f"{CONFIG['img_size']}x{CONFIG['img_size']}",
        "architecture": "ResNet18 ImageNet Transfer + Proportional Sampler (50% Heidelberg / 40% Topcon / 10% Duke)",
        "quotas": CONFIG["domain_quotas"],
        "training_set_size": len(stratified_train_ds),
        "epochs_trained": epoch + 1,
        "best_val_loss": float(best_val_loss),
        "best_val_f1": float(best_val_f1),
        "total_training_time_min": float(total_training_time / 60.0),
        "history": history,
        "benchmarks": benchmark_results
    }

    with open(CONFIG["results_save_path"], "w") as f:
        json.dump(final_payload, f, indent=4)
    print(f"\n[+] Saved full benchmark results to {CONFIG['results_save_path'].name}")
    print("=" * 75)
    print("PHASE 3E COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    main()
