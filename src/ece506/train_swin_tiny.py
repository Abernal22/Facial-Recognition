"""
train_swin_tiny.py
Fine-tune Swin-Tiny on RAF-DB using Juan's custom data loaders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import timm
from time import time

import data as dp


# -------------------------------------------------------
# Convert generator output (numpy) → torch batch
# -------------------------------------------------------
def to_torch_batch(X, y):
    """
    X: (N,H,W,3) float32
    y: (N,)
    returns X_torch: (N,3,H,W), y_torch
    """
    X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


# -------------------------------------------------------
# Build Swin-Tiny Model
# -------------------------------------------------------
def create_swin_tiny(num_classes):
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=num_classes
    )
    return model


# -------------------------------------------------------
# Evaluation loop
# -------------------------------------------------------
def evaluate(model, gen, criterion, device="cuda"):
    model.eval()
    total, correct, loss_sum = 0, 0, 0

    with torch.no_grad():
        for X_np, y_np, _ in gen:
            X, y = to_torch_batch(X_np, y_np)
            X, y = X.to(device), y.to(device)

            out = model(X)
            loss = criterion(out, y)

            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            loss_sum += loss.item() * len(X_np)
            total += len(X_np)

    return loss_sum / total, correct / total


# -------------------------------------------------------
# Training loop
# -------------------------------------------------------
def train(model, train_gen, val_gen, epochs, lr, device="cuda"):

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):

        model.train()
        t0 = time()
        running_loss = 0
        total_correct = 0
        total_samples = 0

        print(f"\n--- Epoch {epoch}/{epochs} ---")

        for X_np, y_np, _ in train_gen:

            X, y = to_torch_batch(X_np, y_np)
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            preds = out.argmax(1)
            total_correct += (preds == y).sum().item()
            running_loss += loss.item() * len(X_np)
            total_samples += len(X_np)

        # Epoch metrics
        train_acc = total_correct / total_samples
        train_loss = running_loss / total_samples

        # Validation
        val_loss, val_acc = evaluate(model, val_gen, criterion, device)

        print(f"Train Loss: {train_loss:.4f}   Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}   Val   Acc: {val_acc:.4f}")
        print(f"Time per epoch: {time() - t0:.1f} sec")

    return model


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():

    # ---------------------------
    # 1. Load RAF-DB using your pipeline
    # ---------------------------
    train_dir, test_dir = dp.download_rafdb()

    num_classes = 7   # RAF-DB emotions (fixed)
    img_size = 224
    batch_size = 32

    train_gen = dp.folder_image_generator(
        train_dir,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    val_gen = dp.folder_image_generator(
        test_dir,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # ---------------------------
    # 2. Create model
    # ---------------------------
    print("Creating Swin-Tiny model...")
    model = create_swin_tiny(num_classes)

    # ---------------------------
    # 3. Train
    # ---------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on:", device)

    model = train(
        model,
        train_gen=train_gen,
        val_gen=val_gen,
        epochs=10,
        lr=3e-4,
        device=device
    )

    # ---------------------------
    # 4. Save model
    # ---------------------------
    torch.save(model.state_dict(), "swin_tiny_rafdb.pth")
    print("\nModel saved to swin_tiny_rafdb.pth")


if __name__ == "__main__":
    main()
