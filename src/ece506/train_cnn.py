from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from .data import download_rafdb, load_train_data
from .model import build_cnn


def train_cnn(model_path= "models/fer_cnn_best.keras", img_size=224, batch_size=64, epochs=30):
    train_dir, _ = download_rafdb()
    X, y, cats = load_train_data(train_dir, img_size=img_size)

    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("Train:", Xtr.shape, "Val:", Xval.shape)

    model = build_cnn(input_shape=X.shape[1:], n_classes=len(cats))

    ckpt = ModelCheckpoint(
        model_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        min_delta=0.001,
        patience=10,
        verbose=1,
        restore_best_weights=True,
    )

    print("Training CNN...")
    model.fit(
        Xtr,
        ytr,
        validation_data=(Xval, yval),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ckpt, early],
        verbose=1,
    )

    print("Evaluating CNN on val...")
    loss, acc = model.evaluate(Xval, yval, verbose=0)
    print(f"CNN val accuracy: {acc:.4f}")

    return model, (Xtr, ytr, Xval, yval)


if __name__ == "__main__":
    train_cnn()
