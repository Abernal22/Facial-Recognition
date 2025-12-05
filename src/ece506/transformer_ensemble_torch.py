import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import torch
from transformers import AutoImageProcessor, ViTModel


class ViTFeatureExtractorTorch:
    """
    PyTorch ViT feature extractor using HuggingFace.
    Only used to get features; SVM + rest of the pipeline stays in NumPy / sklearn.
    """

    def __init__(self, model_name="google/vit-base-patch16-224-in21k"):
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded ViT model: {model_name} on {self.device}")

    @torch.no_grad()
    def compute_features(self, X, batch_size=32):
        """
        X: np.ndarray of shape (N, H, W, 3), values in [0,1] or [0,255]
        Returns:
            features: np.ndarray of shape (N, D)
        """
        N = X.shape[0]
        feats = []

        for start in tqdm(range(0, N, batch_size), desc="Computing ViT features"):
            end = min(start + batch_size, N)
            batch = X[start:end]

            # Convert to uint8 in [0,255] if needed
            if batch.dtype != np.uint8:
                batch_uint8 = (batch * 255).clip(0, 255).astype("uint8")
            else:
                batch_uint8 = batch

            inputs = self.processor(
                images=list(batch_uint8),
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            # CLS token (first token) as global feature
            cls_tokens = outputs.last_hidden_state[:, 0, :]  # (B, D)
            feats.append(cls_tokens.cpu().numpy())

        features = np.concatenate(feats, axis=0)
        print("ViT features shape:", features.shape)
        return features


def train_vit_svm_ensemble(
    Xtr,
    ytr,
    Xval,
    yval,
    model_name="google/vit-base-patch16-224-in21k",
    batch_size=16,
):
    """
    Train SVM on ViT features (transformer-only model).

    Returns:
        extractor: ViTFeatureExtractorTorch
        svm_clf: trained SVC
        acc: validation accuracy
    """
    extractor = ViTFeatureExtractorTorch(model_name=model_name)

    print("Computing ViT features for train set...")
    Ftr = extractor.compute_features(Xtr, batch_size=batch_size)

    print("Computing ViT features for validation set...")
    Fval = extractor.compute_features(Xval, batch_size=batch_size)

    svm_clf = SVC(kernel="rbf", probability=True, class_weight="balanced")
    print("Training SVM on ViT features...")
    svm_clf.fit(Ftr, ytr)

    pred = svm_clf.predict(Fval)
    acc = accuracy_score(yval, pred)
    print(f"ViT+SVM val accuracy: {acc:.4f}")
    print(classification_report(yval, pred))

    return extractor, svm_clf, acc


def vit_predict_proba(extractor, svm_clf, X, batch_size=32):
    """
    Transformer-only prediction: ViT features -> SVM probabilities.
    """
    F = extractor.compute_features(X, batch_size=batch_size)
    proba = svm_clf.predict_proba(F)
    return proba
