from ece506.train_cnn import train_cnn
from ece506.transformer_ensemble_torch import (
    train_vit_svm_ensemble,
    vit_predict_proba,
)


def main():
    # 1) Use your existing RAF-DB loader + split
    #    (we don't really care about the CNN here, just the data)
    model, (Xtr, ytr, Xval, yval) = train_cnn(
        img_size=224,   # ViT expects 224x224
        batch_size=32,
        epochs=0        # you can keep epochs=1 if your train_cnn doesn't handle 0
    )

    # 2) Train transformer-only (ViT + SVM) ensemble
    extractor, svm_clf, acc = train_vit_svm_ensemble(
        Xtr, ytr, Xval, yval,
        model_name="google/vit-base-patch16-224-in21k",
        batch_size=8,  # smaller batch to save RAM
    )

    print(f"Final ViT+SVM val accuracy: {acc:.4f}")

    # 3) Example prediction on a few validation samples
    proba = vit_predict_proba(extractor, svm_clf, Xval[:8], batch_size=8)
    print("Example transformer-only probabilities for first 8 val samples:")
    print(proba)


if __name__ == "__main__":
    main()
