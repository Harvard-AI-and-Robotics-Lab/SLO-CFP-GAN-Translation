"""TensorFlow model construction and stable image loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: Path, size: int = 224) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            image = Image.fromarray(np.asarray(data["colorfundus"], dtype=np.uint8))
    else:
        image = Image.open(path)
    with image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def build_model(backbone: str = "vgg19", size: int = 224, learning_rate: float = 1e-5):
    """Reproduce the notebook's 224-pixel binary classifier architecture."""
    import tensorflow as tf

    backbones = {
        "vgg19": tf.keras.applications.VGG19,
        "resnet101": tf.keras.applications.ResNet101,
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
    }
    if backbone not in backbones:
        raise ValueError(f"backbone must be one of {sorted(backbones)}")
    base = backbones[backbone](input_shape=(size, size, 3), include_top=False, weights="imagenet")
    model = tf.keras.Sequential([
        base,
        tf.keras.layers.AveragePooling2D(pool_size=(3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc")],
    )
    return model


def make_sequence(records, batch_size: int, size: int, *, shuffle: bool, augment: bool, seed: int):
    """Create one epoch with each image exactly once, including the last partial batch."""
    import math
    import tensorflow as tf

    if not records or batch_size < 1:
        raise ValueError("Need records and a positive batch size")

    class ImageSequence(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
            self.indices = np.arange(len(records))
            self.rng = np.random.default_rng(seed)
            self.on_epoch_end()

        def __len__(self):
            return math.ceil(len(records) / batch_size)

        def __getitem__(self, index):
            selected = self.indices[index * batch_size:(index + 1) * batch_size]
            images = np.stack([load_image(records[i].image_path, size) for i in selected])
            labels = np.asarray([records[i].label for i in selected], dtype=np.float32)
            if augment:
                # Image-level augmentation preserves patient and label alignment.
                flips = self.rng.random(len(images)) < 0.5
                images[flips] = images[flips, :, ::-1, :]
            return images, labels

        def on_epoch_end(self):
            if shuffle:
                self.rng.shuffle(self.indices)

    return ImageSequence()
