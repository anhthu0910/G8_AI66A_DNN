import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models


PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SIZE = (224, 224)
EXPECTED_CLASSES = {"CNV", "DME", "DRUSEN", "NORMAL"}


def build_complex_cnn(input_shape=(224, 224, 3), num_classes=4):
    """Build the scratch-trained CNN used as Model 2."""
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.02),
            layers.RandomZoom(0.05),
        ],
        name="data_augmentation",
    )

    model = models.Sequential(name="complex_cnn")
    model.add(layers.Input(shape=input_shape))
    model.add(augmentation)
    model.add(layers.Rescaling(1.0 / 255))

    for filters, dropout_rate in ((32, 0.0), (64, 0.0), (128, 0.25), (256, 0.3)):
        model.add(layers.Conv2D(filters, 3, padding="same", use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=2))
        if dropout_rate:
            model.add(layers.Dropout(dropout_rate))

    # Global pooling avoids the very large dense layer created by Flatten.
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(num_classes, activation="softmax", name="output"))
    return model


def _load_dataset(directory, batch_size, shuffle, seed):
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        color_mode="rgb",
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )


def _measure_inference_ms_per_image(model, dataset):
    images, _ = next(iter(dataset))
    model(images, training=False).numpy()  # Warm up the selected device.
    repetitions = 10
    started = time.perf_counter()
    for _ in range(repetitions):
        model(images, training=False).numpy()
    elapsed = time.perf_counter() - started
    return elapsed * 1000 / (repetitions * int(images.shape[0]))


def _save_training_plot(history, output_path):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_confusion_matrix(matrix, class_names, output_path):
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Complex CNN — Test Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def run_training(dataset_name="cut", epochs=30, batch_size=32, seed=42):
    if dataset_name not in {"cut", "raw"}:
        raise ValueError("dataset_name must be either 'cut' or 'raw'.")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive integers.")

    tf.keras.utils.set_random_seed(seed)
    data_root = PROJECT_ROOT / "data" / dataset_name
    split_dirs = {name: data_root / name for name in ("train", "val", "test")}
    missing = [str(path) for path in split_dirs.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Required dataset split directory/directories missing: {missing}")

    train_ds = _load_dataset(split_dirs["train"], batch_size, shuffle=True, seed=seed)
    val_ds = _load_dataset(split_dirs["val"], batch_size, shuffle=False, seed=seed)
    test_ds = _load_dataset(split_dirs["test"], batch_size, shuffle=False, seed=seed)
    class_names = train_ds.class_names

    if set(class_names) != EXPECTED_CLASSES:
        raise ValueError(
            f"Expected class folders {sorted(EXPECTED_CLASSES)}, found {class_names}."
        )
    for split_name, dataset in (("val", val_ds), ("test", test_ds)):
        if dataset.class_names != class_names:
            raise ValueError(
                f"Class order in {split_name} does not match train: "
                f"{dataset.class_names} != {class_names}"
            )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    model = build_complex_cnn(num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model_dir = PROJECT_ROOT / "saved_models"
    result_dir = PROJECT_ROOT / "results" / "complex_cnn" / dataset_name
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"complex_cnn_{dataset_name}_gap.keras"
    history_path = result_dir / "training_history.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(history_path),
    ]

    print(f"\n--- Training on data/{dataset_name} ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    _save_training_plot(history, result_dir / "training_curves.png")

    # Evaluate the selected best-validation checkpoint once on the held-out test split.
    best_model = tf.keras.models.load_model(model_path)
    class_map_path = model_dir / f"{model_path.stem}_classes.json"
    with class_map_path.open("w", encoding="utf-8") as class_map_file:
        json.dump(
            {index: name for index, name in enumerate(class_names)},
            class_map_file,
            indent=2,
            ensure_ascii=False,
        )

    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=0)
    y_true = np.concatenate(
        [np.argmax(labels.numpy(), axis=1) for _, labels in test_ds], axis=0
    )
    probabilities = best_model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    with (result_dir / "classification_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as report_file:
        writer = csv.DictWriter(report_file, fieldnames=["class", *report[class_names[0]].keys()])
        writer.writeheader()
        for class_name in class_names:
            writer.writerow({"class": class_name, **report[class_name]})
        for aggregate in ("macro avg", "weighted avg"):
            writer.writerow({"class": aggregate, **report[aggregate]})

    with (result_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as matrix_file:
        writer = csv.writer(matrix_file)
        writer.writerow(["true/predicted", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row.tolist()])

    _save_confusion_matrix(matrix, class_names, result_dir / "confusion_matrix.png")
    inference_ms = _measure_inference_ms_per_image(best_model, test_ds)
    metrics = {
        "dataset": dataset_name,
        "class_names": class_names,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1_score": float(report["macro avg"]["f1-score"]),
        "weighted_f1_score": float(report["weighted avg"]["f1-score"]),
        "parameter_count": int(best_model.count_params()),
        "model_size_mb": round(model_path.stat().st_size / (1024 * 1024), 3),
        "inference_ms_per_image": round(inference_ms, 3),
        "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)

    print("\n--- Held-out test results ---")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"\nModel: {model_path}")
    print(f"Class mapping: {class_map_path}")
    print(f"Reports and plots: {result_dir}")
    return history, best_model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Complex CNN (Model 2).")
    parser.add_argument("--dataset", choices=("cut", "raw"), default="cut")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_training(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()