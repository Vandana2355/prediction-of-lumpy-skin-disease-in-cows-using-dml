# import os
# from glob import glob
# from pathlib import Path
# import numpy as np
# from PIL import Image
# from sklearn.model_selection import train_test_split

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import timm


# #########################################
# # SETTINGS
# #########################################

# DATA_ROOT = "./data/"   # <-- CHANGE THIS
# BATCH_SIZE = 16
# IMG_SIZE = 224
# EPOCHS = 10
# LR = 2e-5
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# #########################################
# # TRANSFORMS
# #########################################

# vit_mean = [0.485, 0.456, 0.406]
# vit_std  = [0.229, 0.224, 0.225]

# train_transforms = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
#     transforms.ColorJitter(0.2,0.2,0.1,0.02),
#     transforms.ToTensor(),
#     transforms.Normalize(vit_mean, vit_std),
# ])

# val_transforms = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize(vit_mean, vit_std),
# ])


# #########################################
# # DATASET
# #########################################

# class LSDDataset(Dataset):
#     def __init__(self, df, root_dir, transform=None):
#         self.df = df
#         self.root = Path(root_dir)
#         self.transform = transform
    
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         fname, label = self.df[idx]
#         img_path = self.root / fname

#         img = Image.open(img_path).convert("RGB")
#         if self.transform:
#             img = self.transform(img)

#         return img, torch.tensor([label], dtype=torch.float32)


# #########################################
# # LOAD DATA + AUTO-LABEL
# #########################################

# def load_dataset():
#     all_files = []
#     for ext in ["jpg", "jpeg", "png"]:
#         all_files.extend(glob(os.path.join(DATA_ROOT, f"*.{ext}")))

#     data = []
#     for f in all_files:
#         name = os.path.basename(f)
#         if name.lower().startswith("normal"):
#             label = 0
#         else:
#             label = 1
#         data.append([name, label])

#     return data


# #########################################
# # MODEL
# #########################################

# def create_model():
#     model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True)

#     in_features = model.head.in_features
#     model.head = nn.Linear(in_features, 1)

#     # Freeze all layers
#     for p in model.parameters():
#         p.requires_grad = False

#     # Unfreeze last 4 transformer blocks
#     for block in model.blocks[-4:]:
#         for p in block.parameters():
#             p.requires_grad = True

#     # Head always trainable
#     for p in model.head.parameters():
#         p.requires_grad = True

#     return model


# #########################################
# # TRAINING & EVALUATION
# #########################################

# def train_one_epoch(model, loader, optimizer, criterion):
#     model.train()
#     total_loss = 0

#     for imgs, labels in loader:
#         imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

#         optimizer.zero_grad()
#         logits = model(imgs)
#         loss = criterion(logits, labels)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * imgs.size(0)
    
#     return total_loss / len(loader.dataset)


# def evaluate(model, loader):
#     model.eval()
#     preds, trues = [], []

#     with torch.no_grad():
#         for imgs, labels in loader:
#             imgs = imgs.to(DEVICE)
#             logits = model(imgs)
#             prob = torch.sigmoid(logits).cpu().numpy()

#             preds.extend(prob.flatten())
#             trues.extend(labels.numpy().flatten())
    
#     preds = np.array(preds)
#     trues = np.array(trues)
#     acc = ((preds > 0.5).astype(int) == trues).mean()
#     return acc, preds, trues


# #########################################
# # INFERENCE (WITH SEVERITY)
# #########################################

# def classify_and_severity(model, image_path):
#     img = Image.open(image_path).convert("RGB")
#     img_t = val_transforms(img).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         logit = model(img_t)
#         prob = torch.sigmoid(logit).item()

#     severity = prob * 10

#     if severity < 4:
#         category = "Mild"
#     elif severity <= 7:
#         category = "Moderate"
#     else:
#         category = "Severe"

#     return prob, severity, category


# #########################################
# # MAIN TRAINING LOOP
# #########################################

# def main():
#     print("Loading dataset...")
#     data = load_dataset()

#     train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

#     train_ds = LSDDataset(train_data, DATA_ROOT, train_transforms)
#     val_ds   = LSDDataset(val_data, DATA_ROOT, val_transforms)

#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

#     print("Creating model...")
#     model = create_model().to(DEVICE)
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

#     best_acc = 0

#     for epoch in range(EPOCHS):
#         loss = train_one_epoch(model, train_loader, optimizer, criterion)
#         acc, _, _ = evaluate(model, val_loader)

#         print(f"Epoch {epoch+1}/{EPOCHS}  Loss={loss:.4f}  Val Acc={acc:.4f}")

#         if acc > best_acc:
#             best_acc = acc
#             torch.save(model.state_dict(), "best_vit_lsd_classifier.pth")
#             print("Model saved!")

#     print("Training complete!")


# #########################################
# # RUN
# #########################################

# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
"""
train_vit_lsd_gtx1650.py

Optimized for: GTX1650 (4GB VRAM), 8GB RAM laptop.
Dataset format (Option A): all images in one folder.
Filenames starting with "Normal" (case-insensitive) -> label 0, others -> label 1.

Outputs:
 - best_vit_lsd_classifier.pth  (best model by validation accuracy)
 - Training logs printed to console
"""

import os
from glob import glob
from pathlib import Path
import random
import time
import gc
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# ---------------------------
# USER SETTINGS (edit here)
# ---------------------------
DATA_ROOT = "data/"   # <-- change this to your folder containing all images
IMG_SIZE = 224
EPOCHS = 12
BASE_LR = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPU/VRAM safe settings
BATCH_SIZE = 2                     # micro-batch (keeps VRAM low)
ACCUM_STEPS = 4                    # accumulates grads to simulate effective batch of BATCH_SIZE * ACCUM_STEPS
NUM_WORKERS = 2                    # small number to avoid extra RAM usage (increase if you have headroom)
PIN_MEMORY = True if DEVICE == "cuda" else False

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# Transforms
# ---------------------------
vit_mean = [0.485, 0.456, 0.406]
vit_std  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(vit_mean, vit_std),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(vit_mean, vit_std),
])

# ---------------------------
# Dataset
# ---------------------------
class OneFolderLSDDataset(Dataset):
    def __init__(self, items, root_dir, transform=None):
        """
        items: list of (filename, label)
        root_dir: folder path
        """
        self.items = items
        self.root = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, label = self.items[idx]
        img_path = self.root / fname
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float32)

# ---------------------------
# Helpers: dataset loading
# ---------------------------
def build_items_from_folder(root):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob(os.path.join(root, e)))
    files = sorted(files)
    print(f"Found {len(files)} images in {root}")
    items = []
    for p in files:
        name = os.path.basename(p)
        label = 0 if name.lower().startswith("normal") else 1
        items.append([name, label])
    return items

# ---------------------------
# Model creation (ViT-B/16)
# ---------------------------
def create_vit_model(unfreeze_last_n=4):
    model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 1)   # single logit for BCEWithLogits
    # freeze everything first
    for p in model.parameters():
        p.requires_grad = False
    # unfreeze last transformer blocks if available
    if hasattr(model, "blocks"):
        n_blocks = len(model.blocks)
        start = max(0, n_blocks - unfreeze_last_n)
        for i in range(start, n_blocks):
            for p in model.blocks[i].parameters():
                p.requires_grad = True
    # unfreeze head
    for p in model.head.parameters():
        p.requires_grad = True
    return model

# ---------------------------
# Training utilities
# ---------------------------
def train_epoch(model, loader, optimizer, scaler, criterion, epoch):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Train E{epoch+1}")
    for step, (imgs, labels) in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss / ACCUM_STEPS).backward()   # scale down for accumulation

        if (step + 1) % ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=running_loss / ((step + 1) * imgs.size(0)))

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss

def validate(model, loader):
    model.eval()
    all_probs = []
    all_trues = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val ")
        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            trues = labels.numpy().flatten()
            all_probs.extend(probs.tolist())
            all_trues.extend(trues.tolist())
    all_probs = np.array(all_probs)
    all_trues = np.array(all_trues)
    preds = (all_probs > 0.5).astype(int)
    acc = (preds == all_trues).mean()
    return acc, all_probs, all_trues

# ---------------------------
# Severity mapping & inference
# ---------------------------
def severity_from_prob(prob):
    # Map [0,1] probability -> [0,10] severity
    return float(prob) * 10.0

def categorize_severity(sev):
    if sev < 4.0:
        return "Mild"
    elif sev <= 7.0:
        return "Moderate"
    else:
        return "Severe"

def infer_image(model, image_path):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    x = val_transforms(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()
    severity = severity_from_prob(prob)
    category = categorize_severity(severity)
    return prob, severity, category

# ---------------------------
# Main
# ---------------------------
def main():
    # checks
    if not os.path.isdir(DATA_ROOT):
        raise SystemExit(f"DATA_ROOT not found: {DATA_ROOT}. Edit the script and set correct path.")

    items = build_items_from_folder(DATA_ROOT)
    if len(items) == 0:
        raise SystemExit("No images found. Check DATA_ROOT and image file extensions.")

    # split
    train_items, val_items = train_test_split(items, test_size=0.20, random_state=SEED, shuffle=True)
    print(f"Train: {len(train_items)} images, Val: {len(val_items)} images")

    # datasets & loaders
    train_ds = OneFolderLSDDataset(train_items, DATA_ROOT, transform=train_transforms)
    val_ds   = OneFolderLSDDataset(val_items, DATA_ROOT, transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # model
    model = create_vit_model(unfreeze_last_n=4).to(DEVICE)
    print("Model created. Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # optimizer / loss / scaler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=BASE_LR, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(EPOCHS):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, criterion, epoch)
        val_acc, val_probs, val_trues = validate(model, val_loader)

        # clear mem
        torch.cuda.empty_cache()
        gc.collect()

        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}/{EPOCHS}  TrainLoss={train_loss:.4f}  ValAcc={val_acc:.4f}  Time={epoch_time:.1f}s")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_vit_lsd_classifier.pth")
            print(f"Saved best model (ValAcc={best_acc:.4f})")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time/60:.2f} minutes. Best ValAcc={best_acc:.4f}")

    # quick sanity inference on a few validation images
    print("Sanity checking a few validation images (prob -> severity -> category):")
    sample_paths = [os.path.join(DATA_ROOT, name) for name, _ in val_items[:6]]
    for p in sample_paths:
        prob, sev, cat = infer_image(model, p)
        print(f"{os.path.basename(p):30s}  prob={prob:.3f}  sev={sev:.2f}  {cat}")

if __name__ == "__main__":
    main()
