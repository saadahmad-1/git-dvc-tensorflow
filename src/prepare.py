from pathlib import Path
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    np.save(RAW_DIR / "x_train.npy", x_train)
    np.save(RAW_DIR / "y_train.npy", y_train)
    np.save(RAW_DIR / "x_test.npy", x_test)
    np.save(RAW_DIR / "y_test.npy", y_test)

    print(f"Saved raw Fashion-MNIST arrays to {RAW_DIR}")


if __name__ == "__main__":
    main()