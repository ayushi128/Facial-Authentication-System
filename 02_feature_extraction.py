"""
STEP 2: Feature Extraction using Pure PyTorch ResNet50
CO01 + CO02: Fine-tune pretrained model → extract 512-d embeddings → encrypt

Uses only: torch + torchvision + sklearn + cryptography (no TensorFlow, no dlib)
"""

import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from cryptography.fernet import Fernet
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = "./processed"
FEAT_DIR   = "./features"
os.makedirs(FEAT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ── BUILD FINE-TUNABLE MODEL (CO02) ───────────────────────────────────────────
def build_model(num_classes):
    """
    CO02: ResNet50 pretrained → freeze base → unfreeze last 2 blocks → fine-tune.
    512-d embedding layer is the second-to-last layer.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Freeze ALL layers first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 2 blocks (layer3 + layer4)
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True

    # New head: 2048 → 512-d embedding → num_classes
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model

# ── FINE-TUNE (CO02) ──────────────────────────────────────────────────────────
def fine_tune(model, images, labels, epochs=10):
    print(f"\n=== Fine-tuning ResNet50 for {epochs} epochs (CO02) ===")
    model = model.to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    X = torch.stack([transform((img * 255).astype(np.uint8)) for img in images])
    y = torch.tensor(labels, dtype=torch.long)

    loader    = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([
        {"params": model.layer3.parameters(), "lr": 1e-4},
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(),     "lr": 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0.0, 0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out  = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(1) == yb).sum().item()
            total      += len(yb)
        scheduler.step()
        print(f"  Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(loader):.4f} | Acc: {correct/total*100:.1f}%")

    torch.save(model.state_dict(), f"{FEAT_DIR}/finetuned_model.pth")
    print(f"✓ Model saved to {FEAT_DIR}/finetuned_model.pth")
    return model

# ── EXTRACT 512-d EMBEDDINGS ──────────────────────────────────────────────────
def extract_embeddings(model, images):
    """Extract from the ReLU layer (512-d, before final classifier)."""
    print("\n=== Extracting 512-d embeddings ===")
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    embeddings = []
    captured   = []

    def hook_fn(module, input, output):
        captured.append(output.detach().cpu().numpy())

    handle = model.fc[2].register_forward_hook(hook_fn)

    with torch.no_grad():
        for i in range(0, len(images), 64):
            batch = images[i:i+64]
            tensors = torch.stack([
                transform((img * 255).astype(np.uint8)) for img in batch
            ]).to(device)
            captured.clear()
            _ = model(tensors)
            embeddings.append(captured[0])

    handle.remove()
    result = np.vstack(embeddings)
    print(f"✓ Embeddings shape: {result.shape}")
    return result

# ── ENCRYPT EMBEDDINGS (CO01) ─────────────────────────────────────────────────
def encrypt_and_save(embeddings, labels, label_map, key):
    print("\n=== Encrypting embeddings (CO01) ===")
    fernet = Fernet(key)
    db = {}
    for emb, label in zip(embeddings, labels):
        identity = label_map[int(label)]
        if identity not in db:
            db[identity] = []
        db[identity].append(fernet.encrypt(pickle.dumps(emb)))
    with open(f"{FEAT_DIR}/encrypted_embeddings.pkl", "wb") as f:
        pickle.dump(db, f)
    print(f"✓ Encrypted {len(embeddings)} embeddings for {len(db)} identities")

# ── TRAIN / TEST SPLIT (CO02) ─────────────────────────────────────────────────
def create_splits(embeddings, labels):
    print("\n=== Creating Train/Test Split (CO02) ===")
    scaler    = StandardScaler()
    X_scaled  = scaler.fit_transform(embeddings)
    le        = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.30, stratify=y_encoded, random_state=42)

    split_data = {"X_train": X_train, "X_test": X_test,
                  "y_train": y_train, "y_test": y_test,
                  "scaler": scaler, "label_encoder": le}

    with open(f"{FEAT_DIR}/splits.pkl", "wb") as f:
        pickle.dump(split_data, f)

    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Classes: {len(le.classes_)}")
    print(f"✓ Saved to {FEAT_DIR}/splits.pkl")
    return split_data

# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading preprocessed images...")
    images    = np.load(f"{OUTPUT_DIR}/images.npy")
    labels    = np.load(f"{OUTPUT_DIR}/labels.npy")
    with open(f"{OUTPUT_DIR}/label_map.pkl", "rb") as f:
        label_map = pickle.load(f)

    num_classes = len(label_map)
    print(f"Loaded {len(images)} images, {num_classes} identities")

    model      = build_model(num_classes)
    model      = fine_tune(model, images, labels, epochs=3)
    embeddings = extract_embeddings(model, images)

    key = open("./encryption.key", "rb").read()
    encrypt_and_save(embeddings, labels, label_map, key)

    splits = create_splits(embeddings, labels)

    with open(f"{FEAT_DIR}/model_config.pkl", "wb") as f:
        pickle.dump({"num_classes": num_classes, "label_map": label_map}, f)

    print("\n=== Done! Next: run 03_model_comparison.py ===")
