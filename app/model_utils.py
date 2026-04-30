from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageOps

MODEL_PATH = Path("models/dog_breed_model.keras")
LABELS_PATH = Path("models/breed_labels.json")
IMAGE_SIZE = (224, 224)


class DogBreedClassifier:
    def __init__(self, model_path: Path = MODEL_PATH, labels_path: Path = LABELS_PATH) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.model: Optional[Any] = None
        self.labels: list[str] = []
        self.load()

    @property
    def is_ready(self) -> bool:
        return self.model is not None and bool(self.labels)

    def load(self) -> None:
        if not self.model_path.exists() or not self.labels_path.exists():
            return

        from tensorflow import keras

        self.model = keras.models.load_model(self.model_path)
        self.labels = json.loads(self.labels_path.read_text(encoding="utf-8"))

    def predict(self, image: Image.Image) -> tuple[str, float]:
        if not self.is_ready:
            return "Model not trained yet", 0.0

        image = ImageOps.exif_transpose(image)
        prediction = self.model.predict(self._preprocess(image), verbose=0)[0]
        index = int(np.argmax(prediction))
        confidence = float(prediction[index])

        if confidence < 0.55:
            rotated = image.rotate(180, expand=True)
            rotated_prediction = self.model.predict(self._preprocess(rotated), verbose=0)[0]
            prediction = np.maximum(prediction, rotated_prediction)
            index = int(np.argmax(prediction))
            confidence = float(prediction[index])

        if index >= len(self.labels):
            return "Unknown breed", confidence

        return self.labels[index], confidence

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        rgb_image = image.convert("RGB").resize(IMAGE_SIZE)
        array = np.asarray(rgb_image, dtype=np.float32)
        return np.expand_dims(array, axis=0)
