#!/usr/bin/env python3
"""Convert a trained Keras classifier to TensorFlow Lite."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import tensorflow as tf
from tensorflow import keras


def normalize_images(images: tf.Tensor, mode: str) -> tf.Tensor:
    images = tf.cast(images, tf.float32)
    if mode == "zero_one":
        return images / 255.0
    if mode == "minus_one_one":
        return (images / 127.5) - 1.0
    if mode == "raw":
        return images
    raise ValueError(f"Unknown normalization mode: {mode}")


def representative_dataset(
    directory: Path,
    image_size: int,
    samples: int,
    normalization: str,
):
    dataset = keras.utils.image_dataset_from_directory(
        directory,
        labels=None,
        color_mode="rgb",
        batch_size=1,
        image_size=(image_size, image_size),
        shuffle=True,
        seed=42,
    )
    dataset = dataset.take(samples)

    def generator():
        for images in dataset:
            yield [normalize_images(images, normalization)]

    return generator


def convert(args: argparse.Namespace) -> None:
    model = keras.models.load_model(args.keras_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantization == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif args.quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif args.quantization == "int8":
        if args.representative_dir is None:
            raise ValueError("--representative-dir is required for int8 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset(
            args.representative_dir,
            args.image_size,
            args.representative_samples,
            args.input_normalization,
        )
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif args.quantization != "float32":
        raise ValueError(f"Unsupported quantization: {args.quantization}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(converter.convert())
    print(f"Saved: {args.output}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Keras model to TFLite")
    parser.add_argument("--keras-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quantization",
        choices=("float32", "float16", "dynamic", "int8"),
        default="float32",
    )
    parser.add_argument("--representative-dir", type=Path, default=None)
    parser.add_argument("--representative-samples", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--input-normalization",
        choices=("zero_one", "minus_one_one", "raw"),
        default="zero_one",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    convert(parse_args(argv))


if __name__ == "__main__":
    main()
