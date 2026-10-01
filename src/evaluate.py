from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "model.h5"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()

    x_test = np.load(DATA_DIR / "x_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    model = load_model(args.model)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.tight_layout()
    plt.savefig(ROOT / "confusion_matrix.png")
    plt.close()

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
    }

    with open(ROOT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()