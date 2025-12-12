# # import os
# # import shutil
# # import kagglehub

# # # ----------------------------
# # # Set base data directory
# # # ----------------------------
# # BASE_DATA_PATH = r"C:\Users\jlope\Downloads\UNM_Courses\Optimization_Theory\Project\Facial-Recognition\data"

# # DATASETS = {
# #     "RAF-DB": "shuvoalok/raf-db-dataset",
# #     "FER2013": "msambare/fer2013",
# #     "CKPLUS": "shawon10/ckplus",
# #     "AffectNet": "mstjebashazida/affectnet"
# # }


# # def download_and_prepare_dataset(name, kaggle_id, target_base_path):
# #     """
# #     Downloads dataset via KaggleHub and moves it to target folder.
# #     """
# #     target_path = os.path.join(target_base_path, name)
# #     if os.path.exists(target_path):
# #         print(f"{name} already exists at {target_path}, skipping download.")
# #         return target_path

# #     print(f"Downloading {name} to temporary cache...")
# #     cached_path = kagglehub.dataset_download(kaggle_id)
    
# #     print(f"Moving {name} to {target_path} ...")
# #     os.makedirs(target_base_path, exist_ok=True)
# #     shutil.move(cached_path, target_path)
    
# #     print(f"{name} is ready at {target_path}\n")
# #     return target_path


# # def main():
# #     for name, kaggle_id in DATASETS.items():
# #         download_and_prepare_dataset(name, kaggle_id, BASE_DATA_PATH)


# # if __name__ == "__main__":
# #     main()

# import os
# import cv2
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# import torch

# DATA_ROOT = r"C:\Users\jlope\Downloads\UNM_Courses\Optimization_Theory\Project\Facial-Recognition\data"

# class RAFDBDataset(Dataset):
#     """RAF-DB dataset loader, recursively scanning train/test folders."""
#     def __init__(self, root_dir, split="train", img_size=224):
#         self.img_size = img_size
#         self.samples = []

#         dataset_path = os.path.join(root_dir, "DATASET", split)
#         if not os.path.exists(dataset_path):
#             raise FileNotFoundError(f"Path {dataset_path} does not exist")

#         # Collect all image files recursively
#         for class_folder in sorted(os.listdir(dataset_path)):
#             class_path = os.path.join(dataset_path, class_folder)
#             if os.path.isdir(class_path):
#                 for root, _, files in os.walk(class_path):
#                     for fname in files:
#                         if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                             # Label = class folder name as integer (1-7)
#                             label_idx = int(class_folder) - 1  # 0-based index
#                             self.samples.append((os.path.join(root, fname), label_idx))

#         if len(self.samples) == 0:
#             raise RuntimeError(f"No images found in {dataset_path}")

#         self.classes = [str(i) for i in range(1, 8)]  # ['1','2',...,'7']

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         img = cv2.imread(path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (self.img_size, self.img_size))
#         img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
#         return img, label


# # ----------------------------
# # CK+ / Folder-based Dataset
# # ----------------------------
# class ImageFolderDataset(Dataset):
#     """Generic folder-based dataset loader (CK+)."""
#     def __init__(self, root_dir, split="train", img_size=224):
#         self.root_dir = os.path.join(root_dir, split)
#         if not os.path.exists(self.root_dir):
#             self.root_dir = root_dir  # fallback if no train/test split
#         self.img_size = img_size
#         self.samples = []
#         self.classes = sorted(
#             d for d in os.listdir(self.root_dir)
#             if os.path.isdir(os.path.join(self.root_dir, d))
#         )
#         for label_idx, cls in enumerate(self.classes):
#             cls_folder = os.path.join(self.root_dir, cls)
#             for fname in os.listdir(cls_folder):
#                 if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                     self.samples.append((os.path.join(cls_folder, fname), label_idx))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         img = cv2.imread(path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (self.img_size, self.img_size))
#         img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)/255.0
#         return img, label

# # ----------------------------
# # FER2013 Dataset
# # ----------------------------
# class FER2013Dataset(Dataset):
#     """FER2013 CSV-based dataset."""
#     def __init__(self, csv_path, split="train", img_size=224):
#         df = pd.read_csv(csv_path)
#         if split == "train":
#             df = df[df["Usage"] == "Training"]
#         else:
#             df = df[df["Usage"] != "Training"]
#         df = df.reset_index(drop=True)
#         self.pixels = df["pixels"].tolist()
#         self.labels = df["emotion"].values.astype(int)
#         self.img_size = img_size

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         arr = np.array(self.pixels[idx].split(), dtype="uint8").reshape(48,48)
#         img = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
#         img = cv2.resize(img, (self.img_size, self.img_size))
#         img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)/255.0
#         label = self.labels[idx]
#         return img, label

# # ----------------------------
# # AffectNet Dataset
# # ----------------------------
# class AffectNetDataset(Dataset):
#     """AffectNet dataset, CSV + images."""
#     def __init__(self, csv_path, root_dir, split="train", img_size=224):
#         df = pd.read_csv(csv_path)
#         if "split" in df.columns:
#             df = df[df["split"] == split]
#         df = df.dropna(subset=["face_path", "expression"]).reset_index(drop=True)
#         self.paths = [os.path.join(root_dir, p) for p in df["face_path"]]
#         self.labels = df["expression"].values.astype(int)
#         self.img_size = img_size

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         img = cv2.imread(self.paths[idx])
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (self.img_size, self.img_size))
#         img = torch.tensor(img, dtype=torch.float32).permute(2,0,1)/255.0
#         label = self.labels[idx]
#         return img, label

# # ----------------------------
# # Generic DataLoader
# # ----------------------------
# def get_dataloader(dataset_name, split="train", img_size=224, batch_size=32, shuffle=True, num_workers=2):
#     """Return PyTorch DataLoader for a given dataset."""
#     dataset_name = dataset_name.lower()

#     if dataset_name.lower() == "raf-db":
#         dataset = RAFDBDataset(os.path.join(DATA_ROOT, "RAF-DB"), split=split, img_size=img_size)

#     elif dataset_name == "ckplus":
#         dataset_dir = os.path.join(DATA_ROOT, "CKPLUS", "CK+48")
#         full_dataset = ImageFolderDataset(dataset_dir)
#         N = len(full_dataset)
#         split = int(0.85 * N)
#         train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [split, N - split])
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

#     elif dataset_name == "fer2013":
#         csv_path = os.path.join(DATA_ROOT, "FER2013", "fer2013.csv")
#         dataset = FER2013Dataset(csv_path, split=split, img_size=img_size)

#     elif dataset_name == "affectnet":
#         csv_path = os.path.join(DATA_ROOT, "AffectNet", "labels.csv")
#         root_dir = os.path.join(DATA_ROOT, "AffectNet")
#         dataset = AffectNetDataset(csv_path, root_dir, split=split, img_size=img_size)

#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")

#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
#                       num_workers=num_workers, pin_memory=True)


import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split

DATA_ROOT = r"C:\Users\jlope\Downloads\UNM_Courses\Optimization_Theory\Project\Facial-Recognition\data"

# ----------------------------
# Generic folder-based dataset
# ----------------------------
class ImageFolderDataset(Dataset):
    """Generic folder-based dataset loader."""
    def __init__(self, root_dir, img_size=224):
        self.root_dir = root_dir
        self.img_size = img_size
        self.samples = []

        # classes are folder names
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        for label_idx, cls in enumerate(self.classes):
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_folder, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, label

# ----------------------------
# RAF-DB Dataset
# ----------------------------
class RAFDBDataset(Dataset):
    """RAF-DB dataset loader (folders 1..7)."""
    def __init__(self, root_dir, split="train", img_size=224):
        self.img_size = img_size
        dataset_dir = os.path.join(root_dir, "DATASET", split)
        self.samples = []
        # folders named 1..7
        self.classes = sorted(os.listdir(dataset_dir))
        for label_idx, cls in enumerate(self.classes):
            cls_folder = os.path.join(dataset_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_folder, fname), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return img, label

# ----------------------------
# Generic DataLoader function
# ----------------------------
def get_dataloader(dataset_name, split="train", img_size=224, batch_size=32, shuffle=True, num_workers=2):
    dataset_name = dataset_name.lower()

    if dataset_name == "raf-db":
        dataset = RAFDBDataset(os.path.join(DATA_ROOT, "RAF-DB"), split=split, img_size=img_size)

    elif dataset_name == "ckplus":
        dataset_dir = os.path.join(DATA_ROOT, "CKPLUS", "CK+48")
        full_dataset = ImageFolderDataset(dataset_dir, img_size=img_size)
        # manual split: 85% train, 15% val
        N = len(full_dataset)
        train_size = int(0.85 * N)
        val_size = N - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        if split == "train":
            dataset = train_dataset
        else:
            dataset = val_dataset

    elif dataset_name == "fer2013":
        folder = "train" if split == "train" else "test"
        dataset_dir = os.path.join(DATA_ROOT, "FER2013", folder)
        dataset = ImageFolderDataset(dataset_dir, img_size=img_size)

    elif dataset_name == "affectnet":
        folder = "Train" if split == "train" else "Test"
        dataset_dir = os.path.join(DATA_ROOT, "AffectNet", "archive (3)", folder)
        dataset = ImageFolderDataset(dataset_dir, img_size=img_size)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
