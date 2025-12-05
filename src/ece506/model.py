import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras import layers, models, applications
from sklearn.model_selection import StratifiedKFold


def build_cnn(input_shape, n_classes=7):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def get_feature_extractor(model):
    """Return a Keras Model whose output is the penultimate dense layer."""
    return Model(inputs=model.inputs, outputs=model.layers[-2].output)


class FastFeatureCVTrainer:
    """
    Precompute deep features with a pretrained backbone (ResNet, etc),
    then train a small top model with StratifiedKFold CV.
    """

    def __init__(self, model_name='ResNet50', dataset_path=None,
                 input_shape=(128, 128, 3), n_classes=7, batch_size=32,
                 epochs_per_fold=5, n_splits=5, save_dir="fast_models",
                 freeze_base=True, unfreeze_layers=50):

        self.model_name = model_name
        self.dataset_path = dataset_path
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.epochs_per_fold = epochs_per_fold
        self.n_splits = n_splits
        self.save_dir = save_dir
        self.freeze_base = freeze_base
        self.unfreeze_layers = unfreeze_layers

        os.makedirs(save_dir, exist_ok=True)
        self.global_best_model_path = None
        self.global_best_acc = 0.0

    def index_labels(self, generator_fn):
        """Collect labels for stratified k-fold CV (if needed)."""
        all_labels = []
        gen = generator_fn(self.dataset_path, batch_size=self.batch_size, shuffle=False)
        for _, yb, _ in gen:
            if yb is not None:
                all_labels.append(yb)
        return np.concatenate(all_labels)

    def compute_features(self, generator_fn):
        """Compute features for all images once using frozen base model."""
        base_model_cls = getattr(applications, self.model_name)
        base_model = base_model_cls(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )

        if self.freeze_base:
            # Freeze all except last unfreeze_layers
            if self.unfreeze_layers > 0:
                for layer in base_model.layers[:-self.unfreeze_layers]:
                    layer.trainable = False
                for layer in base_model.layers[-self.unfreeze_layers:]:
                    layer.trainable = True
            else:
                for layer in base_model.layers:
                    layer.trainable = False
        else:
            for layer in base_model.layers:
                layer.trainable = True

        features_list, labels_list = [], []
        gen = generator_fn(self.dataset_path, batch_size=self.batch_size, shuffle=False)
        for Xb, yb, _ in gen:
            feats = base_model.predict(Xb, verbose=0)
            features_list.append(feats)
            labels_list.append(yb)

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        print(f"Features computed: {features.shape}")
        return features, labels

    def build_top_model(self, input_shape):
        """Small top model to train on precomputed features."""
        inp = layers.Input(shape=input_shape)
        x = layers.Dense(256, activation='relu')(inp)
        x = layers.Dropout(0.5)(x)
        out = layers.Dense(self.n_classes, activation='softmax')(x)
        model = models.Model(inputs=inp, outputs=out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_model(self, generator_fn):
        """
        Train using precomputed features + Stratified K-Fold CV.
        Returns:
            path to best model (overall).
        """
        features, labels = self.compute_features(generator_fn)
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=42
        )

        for fold_num, (train_idx, val_idx) in enumerate(skf.split(features, labels), start=1):
            print(f"\n=== Fold {fold_num} ===")
            top_model = self.build_top_model(features.shape[1:])
            fold_dir = os.path.join(self.save_dir, f"fold_{fold_num}")
            os.makedirs(fold_dir, exist_ok=True)
            fold_best_path = os.path.join(fold_dir, "best_model.h5")
            best_val_acc = 0.0

            for epoch in range(1, self.epochs_per_fold + 1):
                print(f"Fold {fold_num} | Epoch {epoch}/{self.epochs_per_fold}")
                # Train in batches
                for start in range(0, len(train_idx), self.batch_size):
                    end = start + self.batch_size
                    Xb = features[train_idx[start:end]]
                    yb = labels[train_idx[start:end]]
                    if len(Xb) == 0:
                        continue
                    top_model.train_on_batch(Xb, yb)

                # Validation
                val_accs = []
                for start in range(0, len(val_idx), self.batch_size):
                    end = start + self.batch_size
                    Xb = features[val_idx[start:end]]
                    yb = labels[val_idx[start:end]]
                    if len(Xb) == 0:
                        continue
                    _, acc = top_model.evaluate(Xb, yb, verbose=0)
                    val_accs.append(acc)

                mean_val_acc = float(np.mean(val_accs)) if val_accs else 0.0
                print(f"Fold {fold_num} | Epoch {epoch} | Val Acc: {mean_val_acc:.4f}")

                if mean_val_acc > best_val_acc:
                    best_val_acc = mean_val_acc
                    top_model.save(fold_best_path)
                    print(f"Saved best model for fold {fold_num}")

            if best_val_acc > self.global_best_acc:
                self.global_best_acc = best_val_acc
                self.global_best_model_path = fold_best_path

            print(f"Best accuracy for fold {fold_num}: {best_val_acc:.4f}")

        print("\n=== Training Finished ===")
        print(f"Best overall model: {self.global_best_model_path}")
        print(f"Accuracy: {self.global_best_acc:.4f}")
        return self.global_best_model_path
