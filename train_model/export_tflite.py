#!/usr/bin/env python3
"""훈련된 Keras 분류 모델을 TensorFlow Lite로 변환합니다."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import tensorflow as tf
from tensorflow import keras


def normalize_images(images: tf.Tensor, mode: str) -> tf.Tensor:
    """모델 훈련 때 사용한 것과 같은 입력 정규화를 적용합니다."""
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
    """full int8 quantization에 필요한 calibration sample을 만듭니다."""
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
        # TFLite는 input tensor 리스트를 yield하는 generator를 기대합니다.
        for images in dataset:
            yield [normalize_images(images, normalization)]

    return generator


def convert(args: argparse.Namespace) -> None:
    """Keras 모델을 불러와 요청한 TFLite 형식으로 변환합니다."""
    model = keras.models.load_model(args.keras_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if args.quantization == "dynamic":
        # dynamic range quantization은 float input을 유지하면서 weight를 압축합니다.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif args.quantization == "float16":
        # float16은 보통 정확도 손실을 작게 유지하면서 모델 크기를 줄입니다.
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
        # full-int8 export는 edge accelerator나 작은 모델 크기가 필요할 때 유용합니다.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif args.quantization != "float32":
        raise ValueError(f"Unsupported quantization: {args.quantization}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(converter.convert())
    print(f"Saved: {args.output}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """독립 실행형 TFLite export에 사용할 command-line option을 정의합니다."""
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
    # 나중에 테스트나 도구에서 import하기 쉽도록 main은 작게 유지합니다.
    convert(parse_args(argv))


if __name__ == "__main__":
    main()
