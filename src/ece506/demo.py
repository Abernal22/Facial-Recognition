import matplotlib.pyplot as plt

from .data import download_rafdb, pick_random_test_image, emotion_map
from .svm_ensemble import ensemble_predict_proba

emotion_map = {
    "1": "Surprise",
    "2": "Fear",
    "3": "Disgust",
    "4": "Happiness",
    "5": "Sadness",
    "6": "Anger",
    "7": "Neutral",
}

def demo_random_image(model, extractor, svm_clf, img_size=100, alpha=0.5):
    _, test_dir = download_rafdb()
    picked = pick_random_test_image(test_dir, img_size=img_size)
    if picked is None:
        print("No test image found.")
        return

    img, arr, true_idx, true_label, path = picked
    proba = ensemble_predict_proba(model, extractor, svm_clf, arr, alpha=alpha)
    pred_idx = int(proba.argmax(axis=1)[0])
    pred_label = emotion_map.get(str(pred_idx + 1), "Unknown")

    print("Random image path:", path)
    plt.imshow(img)
    plt.title(f"Ensemble Pred: {pred_label} | True label: {emotion_map[str(true_label)]}")
    plt.axis("off")
    plt.show()

    print("Ensemble probabilities:", proba)
    print(f"Pred idx: {pred_idx} ({pred_label}) | True idx: {true_idx} ({emotion_map[str(true_label)]})")
