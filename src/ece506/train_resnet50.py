# ----------------------------
# train_resnet50.py
# ----------------------------

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data_processing import get_dataloader

# Path to save models and logs
MODELS_PATH = r"C:\Users\jlope\Downloads\UNM_Courses\Optimization_Theory\Project\Facial-Recognition\src\models\saved_models"
LOG_PATH = r"C:\Users\jlope\Downloads\UNM_Courses\Optimization_Theory\Project\Facial-Recognition\src\logs"

# ----------------------------
# Training function
# ----------------------------
def train_resnet50(dataset_name="raf-db", img_size=224, batch_size=32, epochs=10, lr=3e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(MODELS_PATH, exist_ok=True)
    log_dir = os.path.join(LOG_PATH, dataset_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    best_val_acc = 0.0

    print("Training on:", device)

    # ----------------------------
    # Load dataset
    # ----------------------------
    print(f"Loading {dataset_name} dataset...")
    train_loader = get_dataloader(
        dataset_name, split="train", img_size=img_size,
        batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = get_dataloader(
        dataset_name, split="test", img_size=img_size,
        batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Determine number of classes dynamically
    sample_batch = next(iter(train_loader))
    num_classes = 7
    print(f"Number of classes: {num_classes}")

    # ----------------------------
    # Create ResNet50 model
    # ----------------------------
    print("Creating ResNet50 model...")
    model = models.resnet50(pretrained=True)
    # Replace the final classifier layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # ----------------------------
    # Class weights for imbalanced datasets
    # ----------------------------
    class_counts = torch.zeros(num_classes, dtype=torch.int64)
    for _, labels in train_loader:
        for l in labels:
            class_counts[l] += 1
    inv_counts = 1.0 / class_counts.float()
    weights = inv_counts / inv_counts.sum()
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ----------------------------
    # Training loop
    # ----------------------------
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Train
        model.train()
        train_loss, train_correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_loss /= total
        train_acc = train_correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # ----------------------------
        # Log to TensorBoard
        # ----------------------------
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        # ----------------------------
        # Save best model only
        # ----------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(MODELS_PATH, f"resnet50_{dataset_name}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, best_model_path)
            print(f"New best model saved: {best_model_path}")

        print(f"\n--- Epoch {epoch}/{epochs} ---")
        print(f"Train Loss: {train_loss:.4f}   Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}   Val   Acc: {val_acc:.4f}")
        print(f"Time per epoch: {time.time() - t0:.1f} sec\n")
        
    writer.close()


# ----------------------------
# Run the training
# ----------------------------
if __name__ == "__main__":
    train_resnet50(
        dataset_name="raf-db",
        img_size=224,
        batch_size=32,
        epochs=10,
        lr=3e-4
    )
