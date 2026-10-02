from pathlib import Path
import argparse

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, "r") as f:
        params = yaml.safe_load(f)

    cfg = params["preprocess"]

    test_size = cfg["test_size"]
    seed = cfg["seed"]

    rng = np.random.default_rng(seed)

    x_train = np.load(RAW_DIR / "x_train.npy").astype("float32")
    y_train = np.load(RAW_DIR / "y_train.npy")

    x_test = np.load(RAW_DIR / "x_test.npy").astype("float32")
    y_test = np.load(RAW_DIR / "y_test.npy")

    # Teammate normalization: standardization
    x_train = (x_train - x_train.mean()) / x_train.std()
    x_test = (x_test - x_test.mean()) / x_test.std()

    indices = rng.permutation(len(x_train))

    x_train = x_train[indices]
    y_train = y_train[indices]

    split = int(len(x_train) * (1 - test_size))

    x_train, x_val = x_train[:split], x_train[split:]
    y_train, y_val = y_train[:split], y_train[split:]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(PROCESSED_DIR / "x_train.npy", x_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "x_val.npy", x_val)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DIR / "x_test.npy", x_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    print(f"Saved processed arrays to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()