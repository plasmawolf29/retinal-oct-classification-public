"""
Prepare Clean Phase 3 Multi-Scanner Datasets & Manifests
=========================================================
1. Re-indexes external_datasets/processed_224_clahe_clean:
   - Duke: 45 patients (AMD1..15 -> DRUSEN, DME1..15 -> DME, NORMAL1..15 -> NORMAL)
     Split: patients 1..12 (80% train), patients 13..15 (20% test)
   - NEH: 441 patients (161 CNV, 160 DRUSEN, 120 NORMAL)
     Split: 80% train patients, 20% test patients per class
2. Strict Patient-Level Anti-Leakage:
   - Zero overlap of patients between train and test
3. Generates JSON manifests and high-speed contiguous .npz caches for instant DataLoader access
   (avoids Windows NTFS file-handle latency and CPU worker overhead).
4. Provides get_benchmark_dataloaders() for evaluation scripts.
"""

import os
import io
import json
import random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = SCRIPT_DIR.parents[1]
BASE_DIR   = SCRIPT_DIR.parents[3]
EXT_DIR    = BASE_DIR / "external_datasets"
PROC_DIR   = EXT_DIR / "processed_224_clahe_clean"
KERMANY_PQ = EXT_DIR / "kermany_test.parquet"

CLS_MAP = {"CNV": 0, "DME": 1, "DRUSEN": 2, "NORMAL": 3}
REVERSE_MAP = {0: "CNV", 1: "DME", 2: "DRUSEN", 3: "NORMAL"}


class NPZDataset(Dataset):
    """PyTorch Dataset wrapper for cached .npz benchmark files."""
    def __init__(self, npz_path, transform=None, in_channels=1):
        data = np.load(npz_path)
        self.images = data["images"]
        self.labels = data["labels"]
        self.transform = transform
        self.in_channels = in_channels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        # Convert to 1-channel Grayscale ("L") or 3-channel RGB ("RGB")
        if self.in_channels == 1:
            img_pil = Image.fromarray(img).convert("L")
        else:
            img_pil = Image.fromarray(img).convert("RGB")
            
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)
            
        return img_tensor, torch.tensor(label, dtype=torch.long)


def get_benchmark_dataloaders(image_size=224, batch_size=32, num_workers=0, in_channels=1):
    """
    Returns PyTorch DataLoaders for all benchmark test sets (Kermany, Duke, NEH).
    Defaults to in_channels=1 for grayscale OCT stem processing.
    """
    if in_channels == 1:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    benchmark_files = {
        "kermany": EXT_DIR / "kermany_test_224.npz",
        "duke": EXT_DIR / "duke_test_224.npz",
        "neh": EXT_DIR / "neh_test_224.npz"
    }
    
    loaders = {}
    for name, npz_path in benchmark_files.items():
        if npz_path.exists():
            ds = NPZDataset(npz_path, transform=transform, in_channels=in_channels)
            loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            print(f"[+] Loaded benchmark dataloader '{name}': {len(ds)} samples.")
        else:
            print(f"[!] Warning: Benchmark file not found at {npz_path}")
            
    return loaders


def process_datasets():
    print("=" * 70)
    print("STEP 1 & 2: MULTI-SCANNER DATASET AUDIT, SPLITTING & NPZ CACHING")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. DUKE BIOPTIGEN AUDIT & PATIENT-LEVEL SPLIT
    # -------------------------------------------------------------
    duke_dir = PROC_DIR / "duke_raw"
    duke_patients = {}
    for p in duke_dir.rglob("*"):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            parts = p.parts
            lbl = None
            pid = None
            for part in parts:
                p_up = part.upper()
                if p_up.startswith("AMD"):
                    lbl = "DRUSEN"
                    pid = part
                    break
                elif p_up.startswith("DME"):
                    lbl = "DME"
                    pid = part
                    break
                elif p_up.startswith("NORMAL"):
                    lbl = "NORMAL"
                    pid = part
                    break
            if pid and lbl:
                duke_patients.setdefault((lbl, pid), []).append(p)

    print(f"[*] Duke total patient cohorts: {len(duke_patients)} (15 AMD/Drusen, 15 DME, 15 Normal)")
    
    duke_train_records = []
    duke_test_records  = []
    
    for (lbl, pid), files in sorted(duke_patients.items()):
        p_num = int(''.join(filter(str.isdigit, pid))) if any(c.isdigit() for c in pid) else 0
        target = duke_test_records if p_num in [13, 14, 15] else duke_train_records
        for f in files:
            target.append({
                "file_path": str(f.resolve()),
                "label_name": lbl,
                "label": CLS_MAP[lbl],
                "patient": pid,
                "source": "Duke_Bioptigen"
            })

    print(f"[+] Duke Train: {len(duke_train_records)} scans (Patients 1..12)")
    print(f"    Distribution: {Counter(r['label_name'] for r in duke_train_records)}")
    print(f"[+] Duke Test:  {len(duke_test_records)} scans (Patients 13..15)")
    print(f"    Distribution: {Counter(r['label_name'] for r in duke_test_records)}")

    # -------------------------------------------------------------
    # 2. NEH TOPCON AUDIT & PATIENT-LEVEL SPLIT
    # -------------------------------------------------------------
    neh_dir = PROC_DIR / "neh_topcon"
    neh_patients = {}
    for p in neh_dir.rglob("*"):
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
            parts = p.parts
            lbl = None
            pid = None
            for i, part in enumerate(parts):
                p_up = part.upper()
                if p_up in ["CNV", "DRUSEN", "NORMAL"]:
                    lbl = p_up
                    if i + 1 < len(parts):
                        pid = f"{lbl}_{parts[i+1]}"
                    break
            if lbl and pid:
                neh_patients.setdefault((lbl, pid), []).append(p)

    print(f"\n[*] NEH total patient cohorts: {len(neh_patients)}")
    print(f"    Patients per class: {Counter(lbl for (lbl, pid) in neh_patients.keys())}")

    random.seed(42)
    neh_train_records = []
    neh_test_records  = []

    for cls in ["CNV", "DRUSEN", "NORMAL"]:
        pids = sorted([pid for (l, pid) in neh_patients.keys() if l == cls])
        random.shuffle(pids)
        n_tr = int(0.8 * len(pids))
        tr_pids = set(pids[:n_tr])
        te_pids = set(pids[n_tr:])

        for pid in tr_pids:
            for f in neh_patients[(cls, pid)]:
                neh_train_records.append({
                    "file_path": str(f.resolve()),
                    "label_name": cls,
                    "label": CLS_MAP[cls],
                    "patient": pid,
                    "source": "NEH_Topcon"
                })
        for pid in te_pids:
            for f in neh_patients[(cls, pid)]:
                neh_test_records.append({
                    "file_path": str(f.resolve()),
                    "label_name": cls,
                    "label": CLS_MAP[cls],
                    "patient": pid,
                    "source": "NEH_Topcon"
                })

    print(f"[+] NEH Train: {len(neh_train_records)} scans (80% patient split)")
    print(f"    Distribution: {Counter(r['label_name'] for r in neh_train_records)}")
    print(f"[+] NEH Test:  {len(neh_test_records)} scans (20% patient split)")
    print(f"    Distribution: {Counter(r['label_name'] for r in neh_test_records)}")

    # -------------------------------------------------------------
    # 3. SAVE JSON MANIFESTS
    # -------------------------------------------------------------
    manifests = {
        "manifest_phase3_duke_train.json": duke_train_records,
        "manifest_phase3_duke_test.json": duke_test_records,
        "manifest_phase3_neh_train.json": neh_train_records,
        "manifest_phase3_neh_test.json": neh_test_records,
        "manifest_phase3_external_train.json": duke_train_records + neh_train_records
    }

    for fname, data in manifests.items():
        out_p = EXT_DIR / fname
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[+] Saved manifest: {out_p} ({len(data)} items)")

    # -------------------------------------------------------------
    # 4. BUILD FAST CONTIGUOUS .NPZ ARRAYS (224x224)
    # -------------------------------------------------------------
    print("\n[*] Packing Duke and NEH scans into contiguous 224x224 .npz files for high-speed I/O...")
    
    def pack_npz(records, out_npz_path):
        imgs = []
        lbls = []
        for r in records:
            im = Image.open(r["file_path"]).convert("L")
            if im.size != (224, 224):
                im = im.resize((224, 224), Image.Resampling.BILINEAR)
            imgs.append(np.array(im, dtype=np.uint8))
            lbls.append(r["label"])
        
        imgs_np = np.stack(imgs, axis=0)
        lbls_np = np.array(lbls, dtype=np.int64)
        np.savez_compressed(out_npz_path, images=imgs_np, labels=lbls_np)
        print(f"[+] Created {out_npz_path.name}: {imgs_np.shape} | Size: {os.path.getsize(out_npz_path)/(1024*1024):.1f} MB")

    pack_npz(duke_train_records, EXT_DIR / "duke_train_224.npz")
    pack_npz(duke_test_records,  EXT_DIR / "duke_test_224.npz")
    pack_npz(neh_train_records,  EXT_DIR / "neh_train_224.npz")
    pack_npz(neh_test_records,   EXT_DIR / "neh_test_224.npz")

    # -------------------------------------------------------------
    # 5. PACK KERMANY FULL-RES TEST (1000 SCANS)
    # -------------------------------------------------------------
    if KERMANY_PQ.exists():
        print("\n[*] Packing Kermany Cell 2018 external test set (1000 scans)...")
        df_k = pd.read_parquet(KERMANY_PQ)
        k_imgs = []
        k_lbls = []
        for _, row in df_k.iterrows():
            im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("L")
            im = im.resize((224, 224), Image.Resampling.BILINEAR)
            k_imgs.append(np.array(im, dtype=np.uint8))
            k_lbls.append(int(row["label"]))
        k_imgs_np = np.stack(k_imgs, axis=0)
        k_lbls_np = np.array(k_lbls, dtype=np.int64)
        np.savez_compressed(EXT_DIR / "kermany_test_224.npz", images=k_imgs_np, labels=k_lbls_np)
        print(f"[+] Created kermany_test_224.npz: {k_imgs_np.shape} | Size: {os.path.getsize(EXT_DIR / 'kermany_test_224.npz')/(1024*1024):.1f} MB")

    print("\n" + "=" * 70)
    print("ALL STEP 1 & 2 DELIVERABLES PRODUCED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    process_datasets()