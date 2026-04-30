from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_OUTPUT = Path("models/dog_breed_model.keras")
LABEL_OUTPUT = Path("models/breed_labels.json")


def build_model(class_count: int) -> keras.Model:
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(class_count, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_datasets(data_dir: Path, validation_split: float, seed: int) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    return train_ds, val_ds, class_names


def train(data_dir: Path, epochs: int, validation_split: float, seed: int) -> None:
    if (data_dir / "Images").exists():
        data_dir = data_dir / "Images"

    train_ds, val_ds, class_names = load_datasets(data_dir, validation_split, seed)
    class_names = [name.split("-", 1)[-1].replace("_", " ") for name in class_names]
    model = build_model(len(class_names))

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_OUTPUT, save_best_only=True),
    ]

    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

    if not MODEL_OUTPUT.exists():
        model.save(MODEL_OUTPUT)

    LABEL_OUTPUT.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_OUTPUT}")
    print(f"Saved labels to {LABEL_OUTPUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 dog breed classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed-dogs"),
        help="Image-folder dataset path. Defaults to the prepared Stanford Dogs folder.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_dir, args.epochs, args.validation_split, args.seed)
