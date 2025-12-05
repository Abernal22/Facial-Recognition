import os
import random
import numpy as np
import cv2
import pandas as pd
import kagglehub

def download_rafdb():
    """
    Download / locate RAF-DB dataset via kagglehub.

    Returns:
        train_dir, test_dir
    """
    print("Downloading / locating RAF-DB dataset...")
    rafdb_path = kagglehub.dataset_download("shuvoalok/raf-db-dataset")
    print("Path to RAF-DB dataset files:", rafdb_path)
    train_dir = os.path.join(rafdb_path, "DATASET", "train")
    test_dir = os.path.join(rafdb_path, "DATASET", "test")
    return train_dir, test_dir


def download_fer2013():
    print("Downloading / locating FER2013 dataset...")
    fer2013_path = kagglehub.dataset_download("msambare/fer2013")
    print("Path to FER2013 dataset files:", fer2013_path)
    return fer2013_path


def download_ckplus():
    print("Downloading / locating CK+ dataset...")
    ckplus_path = kagglehub.dataset_download("davilsena/ckdataset")
    print("Path to CK+ dataset files:", ckplus_path)
    return ckplus_path


def download_affectnet():
    print("Downloading / locating AffectNet dataset...")
    affectnet_path = kagglehub.dataset_download("fatihkgg/affectnet-yolo-format")
    print("Path to AffectNet dataset files:", affectnet_path)
    return affectnet_path


def folder_image_generator(dataset_path, img_size=128, batch_size=64,
                           shuffle=True, with_labels=True):
    """
    Generic folder-based image generator.

    If with_labels=True:
        dataset_path/
            class_0/
                *.jpg|png|jpeg
            class_1/
                ...

    Yields:
        X_batch (N,H,W,3), y_batch (N,), global_indices (N,)
    """
    if with_labels:
        classes = sorted(
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        )
        file_list, labels = [], []
        for idx, cls in enumerate(classes):
            folder = os.path.join(dataset_path, cls)
            for fn in os.listdir(folder):
                if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                    file_list.append(os.path.join(folder, fn))
                    labels.append(idx)
        file_list = np.array(file_list)
        labels = np.array(labels, dtype="int32")
    else:
        file_list = np.array([
            os.path.join(dataset_path, f)
            for f in os.listdir(dataset_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])
        labels = None

    total = len(file_list)
    indices = np.arange(total)

    if shuffle and with_labels:
        perm = np.random.permutation(total)
        file_list = file_list[perm]
        labels = labels[perm]
        indices = indices[perm]

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_files = file_list[start:end]
        batch_idx = indices[start:end]

        X_list = []
        y_list = []

        for i, f in enumerate(batch_files):
            img = cv2.imread(f)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            X_list.append(img)
            if with_labels:
                y_list.append(labels[start + i])

        if not X_list:
            continue

        X = np.array(X_list, dtype="float32") / 255.0
        if with_labels:
            y = np.array(y_list, dtype="int32")
        else:
            y = None

        yield X, y, batch_idx[: len(X)]


def ckplus_generator(csv_path, img_size=128, batch_size=64, shuffle=False):
    """
    Reads CSV/XLSX with 'pixels' and 'emotion' columns.
    Yields X_batch, y_batch, global_indices (row numbers).
    """
    df = pd.read_excel(csv_path) if csv_path.endswith(".xlsx") else pd.read_csv(csv_path)
    df = df.dropna(subset=["pixels", "emotion"]).reset_index(drop=True)

    pixels = df["pixels"].tolist()
    labels = df["emotion"].values.astype(int)
    total = len(pixels)
    indices = np.arange(total)

    if shuffle:
        perm = np.random.permutation(total)
        pixels = [pixels[i] for i in perm]
        labels = labels[perm]
        indices = indices[perm]

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        X, y, gl = [], [], []
        for j in range(start, end):
            arr = np.array(pixels[j].split(), dtype="uint8").reshape(48, 48)
            img = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(labels[j])
            gl.append(indices[j])
        if not X:
            continue
        yield (
            np.array(X, dtype="float32") / 255.0,
            np.array(y, dtype="int32"),
            np.array(gl, dtype="int32"),
        )


def load_train_data(train_dir, img_size=224):
    """
    Load RAF-DB train split fully into memory.

    Returns:
        X: (N, H, W, 3)
        y: (N,)
        cats: list of class folder names
    """
    classes = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )
    file_paths = []
    labels = []

    for idx, cls in enumerate(classes):
        folder = os.path.join(train_dir, cls)
        for fn in os.listdir(folder):
            if fn.lower().endswith((".jpg", ".png", ".jpeg")):
                file_paths.append(os.path.join(folder, fn))
                labels.append(idx)

    labels = np.array(labels, dtype="int32")
    N = len(file_paths)
    X = np.zeros((N, img_size, img_size, 3), dtype="float32")

    print(f"Loading {N} images from RAF-DB train into memory...")
    for i, path in enumerate(file_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
        X[i] = img / 255.0

    return X, labels, classes


def pick_random_test_image(test_dir, img_size=100):
    """
    Pick a random image from RAF-DB test split.
    Returns:
        img_rgb_display (H,W,3 uint8),
        img_batch (1,H,W,3 float32),
        class_idx,
        class_label (folder name),
        path
    """
    classes = sorted(
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    )
    if not classes:
        return None

    cls = random.choice(classes)
    folder = os.path.join(test_dir, cls)
    imgs = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    if not imgs:
        return None

    fn = random.choice(imgs)
    path = os.path.join(folder, fn)

    img = cv2.imread(path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size, img_size))

    arr = img_resized.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    class_idx = classes.index(cls)
    return img_resized, arr, class_idx, cls, path


emotion_map = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happiness",
    "5": "Sadness",
    "6": "Anger",
    "7": "Neutral",
}
