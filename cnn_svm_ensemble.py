import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

import kagglehub

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing import image

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# =========================
# 1. Dataset download & paths
# =========================
print("Downloading / locating RAF-DB dataset...")
base_path = kagglehub.dataset_download("shuvoalok/raf-db-dataset")
print("Path to dataset files:", base_path)

DATASET_FOLDER = os.path.join(base_path, "DATASET")
TRAIN_FOLDER = os.path.join(DATASET_FOLDER, "train")
TEST_FOLDER = os.path.join(DATASET_FOLDER, "test")

if not os.path.exists(TRAIN_FOLDER):
    raise RuntimeError(f"Train folder not found at {TRAIN_FOLDER}")
if not os.path.exists(TEST_FOLDER):
    raise RuntimeError(f"Test folder not found at {TEST_FOLDER}")

print("Train subfolders:", os.listdir(TRAIN_FOLDER))


# =========================
# 2. Helpers
# =========================
def find_class_folders(directory_path):
    if not os.path.isdir(directory_path):
        raise ValueError(f"'{directory_path}' is not a valid directory.")
    entries = os.listdir(directory_path)
    folders = [
        e for e in entries
        if os.path.isdir(os.path.join(directory_path, e))
    ]
    return folders


# Emotion map from numeric folder names
emotion_map = {
    '1': 'Surprise',
    '2': 'Fearful',
    '3': 'Disgusted',
    '4': 'Happy',
    '5': 'Sad',
    '6': 'Angry',
    '7': 'Neutral'
}


# =========================
# 3. Load data
# =========================
print("Loading training data...")

categories = sorted(find_class_folders(TRAIN_FOLDER), key=lambda x: int(x))
print("Categories (label -> folder -> emotion):")
for i, f in enumerate(categories):
    print(i, "->", f, "->", emotion_map.get(f, "Unknown"))

data = []
img_size = 100

for folder in categories:
    folder_path = os.path.join(TRAIN_FOLDER, folder)
    label = categories.index(folder)  # 0..6

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        fpath = os.path.join(folder_path, fname)
        img_arr = cv2.imread(fpath)
        if img_arr is None:
            continue
        img_arr = cv2.resize(img_arr, (img_size, img_size))
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        data.append([img_arr, label])

print("Total images loaded:", len(data))
random.shuffle(data)

X = np.array([feat for feat, _ in data], dtype="float32") / 255.0
Y = np.array([lbl for _, lbl in data], dtype="int64")

print("X shape:", X.shape)
print("Y shape:", Y.shape)


# =========================
# 4. Train/Val split
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("Train:", X_train.shape, "Val:", X_val.shape)


# =========================
# 5. CNN model
# =========================
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=X.shape[1:]),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation='relu'),     # feature layer
    Dense(7, activation='softmax')     # output
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    "fer_cnn_best.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    min_delta=0.001,
    patience=10,
    verbose=1,
    restore_best_weights=True
)

print("Training CNN...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, earlystop],
    verbose=1
)

print("Evaluating CNN on val...")
cnn_loss, cnn_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"CNN val accuracy: {cnn_acc:.4f}")


# =========================
# 6. Feature extractor + SVM
# =========================
print("Extracting features for SVM...")

# Dense(128) is the second-to-last layer (-2)
feature_extractor = Model(
    inputs=model.input,
    outputs=model.layers[-2].output
)

X_train_feats = feature_extractor.predict(X_train)
X_val_feats = feature_extractor.predict(X_val)

print("Feature shape:", X_train_feats.shape)

svm_clf = SVC(
    kernel='rbf',
    probability=True,
    class_weight='balanced'
)

print("Training SVM on CNN features...")
svm_clf.fit(X_train_feats, y_train)

y_val_pred_svm = svm_clf.predict(X_val_feats)
svm_acc = accuracy_score(y_val, y_val_pred_svm)
print(f"SVM val accuracy: {svm_acc:.4f}")


# =========================
# 7. Ensemble
# =========================
def ensemble_predict_proba(img_batch, alpha=0.5):
    """
    alpha: weight for CNN; (1 - alpha) for SVM
    """
    cnn_proba = model.predict(img_batch, verbose=0)
    feats = feature_extractor.predict(img_batch, verbose=0)
    svm_proba = svm_clf.predict_proba(feats)
    return alpha * cnn_proba + (1 - alpha) * svm_proba


def ensemble_predict_label(img_batch, alpha=0.5):
    proba = ensemble_predict_proba(img_batch, alpha)
    return np.argmax(proba, axis=1)


print("Evaluating ensemble on val...")
y_val_pred_ens = ensemble_predict_label(X_val, alpha=0.5)
ens_acc = accuracy_score(y_val, y_val_pred_ens)
print(f"Ensemble val accuracy (alpha=0.5): {ens_acc:.4f}")
print(classification_report(y_val, y_val_pred_ens))


# =========================
# 8. Demo on random test image
# =========================
print("Running demo prediction on random TEST image...")

test_subfolders = [
    f for f in os.listdir(TEST_FOLDER)
    if os.path.isdir(os.path.join(TEST_FOLDER, f))
]

chosen_folder = random.choice(test_subfolders)
test_img_files = [
    f for f in os.listdir(os.path.join(TEST_FOLDER, chosen_folder))
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not test_img_files:
    print("No test images found in chosen folder.")
else:
    chosen_img = random.choice(test_img_files)
    img_path = os.path.join(TEST_FOLDER, chosen_folder, chosen_img)
    print("Random image:", img_path)

    true_idx = int(chosen_folder) - 1  # folders are '1'..'7'
    true_label = emotion_map.get(chosen_folder, "Unknown")

    img_obj = image.load_img(img_path, target_size=(img_size, img_size))
    img_arr = image.img_to_array(img_obj) / 255.0
    img_batch = np.expand_dims(img_arr, axis=0)

    proba = ensemble_predict_proba(img_batch, alpha=0.5)
    pred_idx = int(np.argmax(proba, axis=1)[0])  # 0..6
    pred_folder = str(pred_idx + 1)
    pred_label = emotion_map.get(pred_folder, "Unknown")

    plt.imshow(img_obj)
    plt.title(f"Ensemble Pred: {pred_label}\nTrue: {true_label}")
    plt.axis("off")
    plt.show()

    print("Ensemble probabilities:", proba)
    print(f"Pred index: {pred_idx} ({pred_label})")
    print(f"True index: {true_idx} ({true_label})")
