from ece506.train_cnn import train_cnn
from ece506.svm_ensemble import (
    train_svm_on_cnn_features,
    evaluate_ensemble,
)
from ece506.demo import demo_random_image


def main():
    # You can adjust these hyperparameters
    model, (Xtr, ytr, Xval, yval) = train_cnn(
        img_size=100,
        batch_size=64,
        epochs=1,
    )

    svm_clf, extractor, svm_acc = train_svm_on_cnn_features(
        model, Xtr, ytr, Xval, yval
    )
    _ = evaluate_ensemble(
        model, extractor, svm_clf, Xval, yval, alpha=0.5
    )

    # Optional: visualize a random test image
    demo_random_image(
        model, extractor, svm_clf, img_size=100, alpha=0.5
    )


if __name__ == "__main__":
    main()
