#!/usr/bin/env python3
"""Train an eye or mouth state classifier for Raspberry Pi inference.

The exported model expects RGB float input normalized to [0, 1]. This matches
infer_model/run_inference.py's default --input-normalization zero_one setting.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
AUTOTUNE = tf.data.AUTOTUNE


def default_image_size(task: str) -> int:
    if task == "eye":
        return 96
    if task == "mouth":
        return 128
    return 128


def default_class_names(task: str) -> list[str]:
    if task == "eye":
        return ["open", "closed"]
    if task == "mouth":
        return ["normal", "yawn"]
    raise ValueError(f"Unsupported task: {task}")


def image_count_by_class(split_dir: Path, class_names: Sequence[str]) -> dict[str, int]:
    counts = {}
    for class_name in class_names:
        class_dir = split_dir / class_name
        count = 0
        if class_dir.exists():
            for path in class_dir.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    count += 1
        counts[class_name] = count
    return counts


def build_class_weight(
    split_dir: Path,
    class_names: Sequence[str],
    mode: str,
) -> Optional[dict[int, float]]:
    if mode == "none":
        return None

    counts = image_count_by_class(split_dir, class_names)
    total = sum(counts.values())
    if total == 0:
        return None

    weights = {}
    for index, class_name in enumerate(class_names):
        count = counts[class_name]
        weights[index] = 0.0 if count == 0 else total / (len(class_names) * count)
    return weights


def normalize_dataset(dataset: tf.data.Dataset, cache: bool) -> tf.data.Dataset:
    def normalize(images, labels):
        return tf.cast(images, tf.float32) / 255.0, labels

    dataset = dataset.map(normalize, num_parallel_calls=AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    return dataset.prefetch(AUTOTUNE)


def dataset_from_directory(
    directory: Path,
    image_size: Tuple[int, int],
    batch_size: int,
    class_names: Sequence[str],
    seed: int,
    shuffle: bool,
    validation_split: Optional[float] = None,
    subset: Optional[str] = None,
) -> tf.data.Dataset:
    kwargs = {
        "directory": directory,
        "labels": "inferred",
        "label_mode": "categorical",
        "class_names": list(class_names),
        "color_mode": "rgb",
        "batch_size": batch_size,
        "image_size": image_size,
        "shuffle": shuffle,
        "seed": seed,
    }
    if validation_split is not None and subset is not None:
        kwargs["validation_split"] = validation_split
        kwargs["subset"] = subset
    return keras.utils.image_dataset_from_directory(**kwargs)


def load_datasets(args: argparse.Namespace):
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    class_names = args.class_names or default_class_names(args.task)
    image_size = (args.image_size, args.image_size)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if train_dir.exists():
        if val_dir.exists():
            train_raw = dataset_from_directory(
                train_dir,
                image_size,
                args.batch_size,
                class_names,
                args.seed,
                shuffle=True,
            )
            val_raw = dataset_from_directory(
                val_dir,
                image_size,
                args.batch_size,
                class_names,
                args.seed,
                shuffle=False,
            )
        else:
            train_raw = dataset_from_directory(
                train_dir,
                image_size,
                args.batch_size,
                class_names,
                args.seed,
                shuffle=True,
                validation_split=args.validation_split,
                subset="training",
            )
            val_raw = dataset_from_directory(
                train_dir,
                image_size,
                args.batch_size,
                class_names,
                args.seed,
                shuffle=False,
                validation_split=args.validation_split,
                subset="validation",
            )
        class_weight_dir = train_dir
    else:
        train_raw = dataset_from_directory(
            data_dir,
            image_size,
            args.batch_size,
            class_names,
            args.seed,
            shuffle=True,
            validation_split=args.validation_split,
            subset="training",
        )
        val_raw = dataset_from_directory(
            data_dir,
            image_size,
            args.batch_size,
            class_names,
            args.seed,
            shuffle=False,
            validation_split=args.validation_split,
            subset="validation",
        )
        class_weight_dir = data_dir

    test_raw = None
    if test_dir.exists():
        test_raw = dataset_from_directory(
            test_dir,
            image_size,
            args.batch_size,
            class_names,
            args.seed,
            shuffle=False,
        )

    train_ds = normalize_dataset(train_raw, args.cache)
    val_ds = normalize_dataset(val_raw, args.cache)
    test_ds = normalize_dataset(test_raw, args.cache) if test_raw is not None else None
    class_weight = build_class_weight(class_weight_dir, class_names, args.class_weight)

    return train_ds, val_ds, test_ds, list(class_names), class_weight


def build_model(
    image_size: int,
    num_classes: int,
    dropout: float,
    augmentation: bool,
) -> tuple[keras.Model, keras.Model]:
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image_0_1")
    x = inputs

    if augmentation:
        x = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.04),
                layers.RandomZoom(0.08),
                layers.RandomContrast(0.12),
            ],
            name="augmentation",
        )(x)

    # Inference code sends [0, 1] images. MobileNetV3 with include_preprocessing
    # expects [0, 255], so this layer keeps the exported TFLite input contract
    # simple while still using the pretrained ImageNet preprocessing path.
    x = layers.Rescaling(255.0, name="to_0_255")(x)

    base_model = keras.applications.MobileNetV3Small(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        include_preprocessing=True,
        weights="imagenet",
        pooling="avg",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probs")(x)
    model = keras.Model(inputs, outputs, name="drowsiness_state_classifier")
    return model, base_model


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks(output_dir: Path) -> list[keras.callbacks.Callback]:
    return [
        keras.callbacks.ModelCheckpoint(
            output_dir / "best.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(output_dir / "history.csv", append=True),
    ]


def unfreeze_for_finetuning(
    base_model: keras.Model,
    fine_tune_layers: int,
) -> None:
    base_model.trainable = True
    cutoff = max(0, len(base_model.layers) - fine_tune_layers)
    for index, layer in enumerate(base_model.layers):
        if index < cutoff or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def export_tflite(
    model: keras.Model,
    output_path: Path,
    quantization: str,
) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization != "float32":
        raise ValueError(f"Unsupported train-time export quantization: {quantization}")

    output_path.write_bytes(converter.convert())


def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    class_names: Sequence[str],
    test_metrics: Optional[dict[str, float]],
) -> None:
    metadata = {
        "task": args.task,
        "class_names": list(class_names),
        "image_size": args.image_size,
        "input_normalization": "zero_one",
        "model": "MobileNetV3Small",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "test_metrics": test_metrics,
        "notes": [
            "TFLite input is RGB float normalized to [0, 1].",
            "Default positive class index is 1 for infer_model.",
        ],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small classifier")
    parser.add_argument("--task", choices=("eye", "mouth"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--class-names", nargs="+", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")

    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-layers", type=int, default=40)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--no-augmentation", action="store_true")

    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--export-tflite", action="store_true")
    parser.add_argument(
        "--tflite-quantization",
        choices=("float32", "float16", "dynamic"),
        default="float32",
    )
    args = parser.parse_args(argv)

    if args.image_size is None:
        args.image_size = default_image_size(args.task)
    if args.class_names is None:
        args.class_names = default_class_names(args.task)
    if len(args.class_names) < 2:
        raise ValueError("--class-names must contain at least two classes")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    keras.utils.set_random_seed(args.seed)

    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir / args.task / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names, class_weight = load_datasets(args)
    model, base_model = build_model(
        image_size=args.image_size,
        num_classes=len(class_names),
        dropout=args.dropout,
        augmentation=not args.no_augmentation,
    )

    print(f"Output directory: {output_dir}")
    print(f"Class names: {class_names}")
    print(f"Class weight: {class_weight}")

    compile_model(model, args.head_lr)
    if args.head_epochs > 0:
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.head_epochs,
            class_weight=class_weight,
            callbacks=make_callbacks(output_dir),
        )

    if args.fine_tune_epochs > 0 and args.fine_tune_layers > 0:
        unfreeze_for_finetuning(base_model, args.fine_tune_layers)
        compile_model(model, args.fine_tune_lr)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weight,
            callbacks=make_callbacks(output_dir),
        )

    best_path = output_dir / "best.keras"
    if best_path.exists():
        model = keras.models.load_model(best_path)
    else:
        model.save(best_path)

    test_metrics = None
    if test_ds is not None:
        values = model.evaluate(test_ds, return_dict=True)
        test_metrics = {key: float(value) for key, value in values.items()}
        print(f"Test metrics: {test_metrics}")

    if args.export_tflite:
        model_name = "eye_state_model.tflite" if args.task == "eye" else "mouth_state_model.tflite"
        tflite_path = output_dir / model_name
        export_tflite(model, tflite_path, args.tflite_quantization)
        print(f"Saved TFLite model: {tflite_path}")

    save_metadata(output_dir, args, class_names, test_metrics)
    print("Training complete.")


if __name__ == "__main__":
    main(sys.argv[1:])
