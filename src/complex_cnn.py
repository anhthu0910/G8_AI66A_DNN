import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models


# Change to "raw" after the model has been checked on cut.
DATASET = "cut"
ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 15


def build_model(number_of_classes):
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.02),
        layers.RandomZoom(0.05),
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(256, 3, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(number_of_classes, activation="softmax"),
    ], name="complex_cnn")

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    data_dir = ROOT / "data" / DATASET
    if any(not (data_dir / split).is_dir() for split in ("train", "val", "test")):
        raise FileNotFoundError(f"Expected train/val/test folders under {data_dir}")

    train = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )
    test = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train.class_names
    if val.class_names != class_names or test.class_names != class_names:
        raise ValueError("Train, validation, and test class folders must match.")

    train = train.prefetch(tf.data.AUTOTUNE)
    val = val.prefetch(tf.data.AUTOTUNE)
    test = test.prefetch(tf.data.AUTOTUNE)

    model_path = ROOT / "saved_models" / f"complex_cnn_{DATASET}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(len(class_names))
    model.summary()
    model.fit(
        train,
        validation_data=val,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                str(model_path),
                monitor="val_loss",
                save_best_only=True,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            ),
        ],
    )

    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(test, verbose=0)
    actual = np.concatenate([np.argmax(y, axis=1) for _, y in test])
    predicted = np.argmax(model.predict(test, verbose=0), axis=1)

    print(f"\nTest loss: {loss:.4f} | Test accuracy: {accuracy:.4f}")
    print("\nPrecision / Recall / F1-score:")
    print(classification_report(actual, predicted, target_names=class_names, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(actual, predicted))
    print(f"\nBest model saved to: {model_path}")


if __name__ == "__main__":
    main()