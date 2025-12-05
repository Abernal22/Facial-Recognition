import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from .model import get_feature_extractor


def train_svm_on_cnn_features(model, Xtr, ytr, Xval, yval):
    print("Extracting features for SVM...")
    extractor = get_feature_extractor(model)
    Ftr = extractor.predict(Xtr, verbose=0)
    Fval = extractor.predict(Xval, verbose=0)
    print("Feature shape:", Ftr.shape)

    clf = SVC(kernel='rbf', probability=True, class_weight='balanced')
    print("Training SVM on CNN features...")
    clf.fit(Ftr, ytr)
    pred = clf.predict(Fval)
    acc = accuracy_score(yval, pred)
    print(f"SVM val accuracy: {acc:.4f}")
    return clf, extractor, acc


def ensemble_predict_proba(model, extractor, svm_clf, img_batch, alpha=0.5):
    """
    Combine CNN and SVM probabilities with weight alpha for CNN.
    """
    cnn_proba = model.predict(img_batch, verbose=0)
    feats = extractor.predict(img_batch, verbose=0)
    svm_proba = svm_clf.predict_proba(feats)
    return alpha * cnn_proba + (1 - alpha) * svm_proba


def evaluate_ensemble(model, extractor, svm_clf, Xval, yval, alpha=0.5):
    proba = ensemble_predict_proba(model, extractor, svm_clf, Xval, alpha)
    pred = np.argmax(proba, axis=1)
    acc = accuracy_score(yval, pred)
    print(f"Ensemble val accuracy (alpha={alpha}): {acc:.4f}")
    print(classification_report(yval, pred))
    return acc
