"""
Phase 3C: 224x224 Multi-Scanner Mega-Set with 112-Stem Champion Progressive Transfer
====================================================================================
- Architecture: ResNet-18 initialized with weights from the 88.21% Macro F1 64x64 Champion (v7)
- Stem: 3x3 stride-2 conv1 + Identity MaxPool -> 112x112 fine-grained spatial feature maps
- Multi-Scanner Mega-Set Pool (113,265 training scans):
    * Heidelberg Spectralis (OCTMNIST Train: 97,477 scans)
    * Bioptigen SD-OCT (Duke Train: 2,610 scans, patients 1..12)
    * Topcon 3D OCT-1000 (NEH Train: 13,178 scans, 80% patient split)
- Universal CLAHE + Dynamic Contrast Augmentation
- Strict Anti-Leakage: Zero patient overlap between train and test
- Evaluated on 4 Benchmark Test Sets (5-pass TTA):
    1. OCTMNIST Test (In-Domain, 1000 scans)
    2. Kermany Cell 2018 (Full-Res Near-Domain, 1000 scans)
    3. Duke Bioptigen Test (Out-of-Domain, 621 scans, patients 13..15)
    4. NEH Topcon Test (Out-of-Domain, 3625 scans, 20% patient split)
- Speed Safeguards:
    * Proven hardware config: batch_size=32, num_workers=0, FP32 native (AMP disabled), pin_memory=True
    * Clean milestone logging every 250 batches with live ETA and throughput
    * Early speed warning/abort if throughput drops below safety threshold
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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18
from medmnist import OCTMNIST
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, log_loss,
    confusion_matrix, precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

torch.backends.cudnn.benchmark = True

SCRIPT_DIR = Path(__file__).resolve().parent
SCRATCH_DIR = SCRIPT_DIR.parent
EXPORT_DIR = SCRATCH_DIR.parent
BASE_DIR = SCRIPT_DIR.parents[3]
EXT_DIR = BASE_DIR / "external_datasets"
BASE_WEIGHTS = SCRATCH_DIR / "v7_resnet18_octmnist_64x64.pth"

CONFIG = {
    "phase": "3C",
    "name": "Phase 3C (224x224 Multi-Scanner Progressive 112-Stem)",
    "img_size": 224,
    "batch_size": 32,
    "num_workers": 0,
    "use_amp": False,
    "lr": 1e-4,
    "epochs": 15,
    "patience": 7,
    "num_classes": 4,
    "in_channels": 1,
    "num_tta_passes": 5,
    "weight_decay": 1e-3,
    "dropout": 0.3,
    "label_smoothing": 0.1,
    "min_batch_speed": 1.0,  # 112-stem executes at ~2.7-3.0 batch/s
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "model_save_path": SCRIPT_DIR / "phase_3c_resnet18_224x224_112stem_mega_best.pth",
    "results_save_path": SCRIPT_DIR / "phase_3c_224_results.json",
    "cm_save_path": SCRIPT_DIR / "phase_3c_224_cm.png",
    "roc_save_path": SCRIPT_DIR / "phase_3c_224_roc.png",
    "curves_save_path": SCRIPT_DIR / "phase_3c_224_learning_curves.png",
    "seed": 42
}

CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class SmoothFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none',
            weight=self.alpha, label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def build_model(dropout_p=0.3):
    model = resnet18(weights=None)
    # Stride-2 conv1 retains 100% of v7 3x3 filter shapes ([64, 1, 3, 3])
    model.conv1 = nn.Conv2d(CONFIG["in_channels"], 64, kernel_size=3, stride=2, padding=1, bias=False)
    # Identity MaxPool -> produces 112x112 fine-grained spatial feature maps
    model.maxpool = nn.Identity()
    in_feat = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),
        nn.Linear(in_feat, CONFIG["num_classes"])
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

class MultiScannerTrainDataset(Dataset):
    def __init__(self, oct_dataset, duke_npz_path, neh_npz_path, transform=None):
        self.oct_ds = oct_dataset
        self.transform = transform

        print(f"[*] Loading external training caches...")
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
        print(f"[+] Multi-Scanner Train Pool: {self.total_len} scans (OCTMNIST: {self.n_oct}, Duke: {self.n_duke}, NEH: {self.n_neh})")

    def __len__(self):
        return self.total_len

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
    ax1.set_title('Multi-Scanner Loss Curve', fontsize=13, fontweight='bold')
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

    plt.suptitle('Phase 3C Multi-Scanner Learning Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved learning curves -> {save_path}")

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
    print("=" * 70)
    print("PHASE 3C: 224x224 Multi-Scanner Mega-Set with 112-Stem Champion Base")
    print(f"Device: {CONFIG['device']} | Batch Size: {CONFIG['batch_size']} | Workers: {CONFIG['num_workers']}")
    print(f"AMP: DISABLED (FP32 native) | Base Weights: {BASE_WEIGHTS.name}")
    print(f"Stem: stride-2 conv1 + Identity MaxPool -> 112x112 feature maps")
    print(f"Target Epochs: {CONFIG['epochs']} | Patience: {CONFIG['patience']}")
    print("=" * 70)

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
    mega_train_ds = MultiScannerTrainDataset(oct_train, duke_tr_npz, neh_tr_npz, transform=train_transform)

    train_loader = DataLoader(mega_train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
                              num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(oct_val, batch_size=CONFIG["batch_size"], shuffle=False,
                              num_workers=CONFIG["num_workers"], pin_memory=True)

    model = build_model(dropout_p=CONFIG["dropout"])
    if BASE_WEIGHTS.exists():
        print(f"[*] Loading pre-trained 64x64 champion weights from {BASE_WEIGHTS.name}...")
        state_dict = torch.load(BASE_WEIGHTS, map_location="cpu")
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            if k.startswith("fc.") and "fc.1." in model_dict:
                new_k = k.replace("fc.", "fc.1.")
                if new_k in model_dict and model_dict[new_k].shape == v.shape:
                    filtered_dict[new_k] = v
            elif k in model_dict and model_dict[k].shape == v.shape:
                filtered_dict[k] = v
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"[+] Transferred {len(filtered_dict)}/{len(model_dict)} layers from 64x64 champion!")
    else:
        print(f"[!] Warning: Base weights not found at {BASE_WEIGHTS}. Initializing from scratch.")

    model = model.to(CONFIG["device"])
    criterion = SmoothFocalLoss(gamma=2.0, label_smoothing=CONFIG["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "epoch_times": []}

    total_training_start = time.time()
    print(f"\n--- COMMENCING MULTI-SCANNER TRAINING ({CONFIG['epochs']} EPOCHS MAX, PATIENCE={CONFIG['patience']}) ---")

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
                batches_left = total_batches - batch_idx
                sec_per_batch = elapsed_batch / batch_idx
                batch_speed = 1.0 / sec_per_batch
                eta_epoch_min = (batches_left * sec_per_batch) / 60.0
                curr_acc = (correct_train / total_train) * 100.0
                curr_loss = running_loss / total_train
                print(f"  [Epoch {epoch+1:02d}/{CONFIG['epochs']:02d} | Step {batch_idx:04d}/{total_batches} ({batch_idx/total_batches*100:4.1f}%)] "
                      f"Loss: {curr_loss:.4f} | Acc: {curr_acc:5.1f}% | "
                      f"Speed: {batch_speed:4.1f} batch/s | ETA: {eta_epoch_min:4.1f}m", flush=True)

                if batch_speed < CONFIG["min_batch_speed"] and batch_idx >= 500:
                    slow_speed_count += 1
                    if slow_speed_count >= 2:
                        print(f"\n[!] SPEED SAFETY ALERT: Throughput collapsed to {batch_speed:.2f} batch/s. Aborting run.")
                        sys.exit(1)
                else:
                    slow_speed_count = 0

        train_loss = running_loss / total_train
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(CONFIG["device"], non_blocking=True)
                labels = labels.squeeze().long().to(CONFIG["device"], non_blocking=True)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        lr_current = scheduler.get_last_lr()[0]
        epoch_elapsed = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["epoch_times"].append(epoch_elapsed)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            status_str = f"--> [BEST SAVED | Val F1: {val_f1*100:.2f}% | Val Loss: {val_loss:.5f}]"
        else:
            patience_counter += 1
            status_str = f"[Patience: {patience_counter}/{CONFIG['patience']}]"

        avg_epoch_time = np.mean(history["epoch_times"])
        remaining_epochs = CONFIG["epochs"] - (epoch + 1)
        est_remaining_min = (remaining_epochs * avg_epoch_time) / 60.0

        print(f"[*] Summary Epoch [{epoch+1:02d}/{CONFIG['epochs']:02d}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1*100:.2f}% | LR: {lr_current:.6f} | "
              f"Time: {epoch_elapsed/60:.1f}m (Est Rem: {est_remaining_min:.1f}m) {status_str}")

        if patience_counter >= CONFIG["patience"]:
            print(f"\n[!] Early stopping triggered at epoch {epoch+1}")
            break

    total_training_time = time.time() - total_training_start
    print(f"\n[+] Total training elapsed: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} mins)")
    plot_learning_curves(history, CONFIG["curves_save_path"])

    # -------------------------------------------------------------
    # 5-PASS TTA EVALUATION ACROSS ALL 4 BENCHMARKS
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("--- COMPREHENSIVE 4-DATASET BENCHMARK EVALUATION (5-PASS TTA) ---")
    print("=" * 70)
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
            "active_classes": [1, 2, 3]  # DME, DRUSEN, NORMAL
        },
        "NEH_Topcon_Test": {
            "dataset": ArrayDataset(neh_te_data["images"], neh_te_data["labels"], transform=tta_transform),
            "active_classes": [0, 2, 3]  # CNV, DRUSEN, NORMAL
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

        print(f"    Top-1 Accuracy : {acc * 100:.2f}%")
        print(f"    Balanced Acc   : {bal_acc * 100:.2f}%")
        print(f"    Macro F1       : {macro_f1 * 100:.2f}%")
        print(f"    Per-Class Rec  : CNV={rec[0]*100:.1f}%, DME={rec[1]*100:.1f}%, DRUSEN={rec[2]*100:.1f}%, NORMAL={rec[3]*100:.1f}%")

        benchmark_results[bench_name] = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(macro_f1),
            "per_class_recall": {CLASS_NAMES[i]: float(rec[i]) for i in range(4)},
            "per_class_precision": {CLASS_NAMES[i]: float(prec[i]) for i in range(4)},
            "per_class_f1": {CLASS_NAMES[i]: float(f1_per_class[i]) for i in range(4)}
        }

        if bench_name == "OCTMNIST_Test_InDomain":
            cm = confusion_matrix(targets, preds)
            plt.figure(figsize=(8, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
            plt.title(f"Phase 3C Multi-Scanner In-Domain Confusion Matrix\nMacro F1: {macro_f1*100:.2f}% | Bal Acc: {bal_acc*100:.2f}%", fontweight='bold')
            plt.ylabel('Ground Truth')
            plt.xlabel('Predicted')
            plt.tight_layout()
            plt.savefig(CONFIG["cm_save_path"], dpi=300)
            plt.close()
            print(f"[+] Saved in-domain confusion matrix -> {CONFIG['cm_save_path']}")

    final_payload = {
        "model": CONFIG["name"],
        "phase": CONFIG["phase"],
        "resolution": f"{CONFIG['img_size']}x{CONFIG['img_size']}",
        "architecture": "ResNet18 112-Stem Champion Base (112x112 feature maps)",
        "training_set_size": len(mega_train_ds),
        "epochs_trained": epoch + 1,
        "best_val_loss": float(best_val_loss),
        "best_val_f1": float(best_val_f1),
        "total_training_minutes": float(total_training_time / 60.0),
        "benchmarks": benchmark_results,
        "history": history
    }

    with open(CONFIG["results_save_path"], "w") as f:
        json.dump(final_payload, f, indent=4)
    print(f"\n[+] Master Phase 3C results saved to {CONFIG['results_save_path']}")

if __name__ == "__main__":
    main()
