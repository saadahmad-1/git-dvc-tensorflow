from pathlib import Path
import argparse
import csv

import numpy as np
import yaml

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    with open(ROOT / args.config, "r") as f:
        params = yaml.safe_load(f)

    cfg = params["train"]

    x_train = np.load(DATA_DIR / "x_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    x_val = np.load(DATA_DIR / "x_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")

    model = Sequential([
        Flatten(input_shape=x_train.shape[1:]),
        Dense(cfg["dense_units"], activation="relu"),
        Dropout(cfg["dropout_rate"]),
        Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_DIR / "model.h5")

    with open(MODEL_DIR / "history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(history.history.keys())

        for row in zip(*history.history.values()):
            writer.writerow(row)

    print(f"Saved model and history to {MODEL_DIR}")


if __name__ == "__main__":
    main()