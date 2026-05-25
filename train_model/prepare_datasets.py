#!/usr/bin/env python3
"""훈련용 데이터셋 폴더를 표준 구조로 정리합니다.

원본 데이터는 삭제하거나 수정하지 않습니다. 기본 동작은 같은 파일시스템 안에서
hard link를 만들어 디스크 사용량을 크게 늘리지 않고 `datasets/processed` 구조를
생성하는 것입니다. hard link가 실패하면 자동으로 파일 복사로 대체합니다.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")


def iter_images(directory: Path) -> Iterable[Path]:
    """지정한 폴더 아래의 이미지 파일을 정렬된 순서로 반환합니다."""
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def ensure_clean_target(target_dir: Path, clean: bool) -> None:
    """출력 폴더가 이미 있을 때의 처리 방식을 결정합니다."""
    if not target_dir.exists():
        return
    if clean:
        shutil.rmtree(target_dir)
        return
    raise FileExistsError(
        f"{target_dir} already exists. Use --clean to rebuild it."
    )


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    """학습 폴더에 파일을 연결하거나 복사합니다."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return

    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def copy_class_folder(
    src_dir: Path,
    dst_dir: Path,
    prefix: str,
    mode: str,
) -> int:
    """class 폴더 안의 이미지를 목적지 class 폴더로 정리합니다."""
    count = 0
    for src in iter_images(src_dir):
        dst = dst_dir / f"{prefix}_{src.name}"
        link_or_copy(src, dst, mode)
        count += 1
    return count


def prepare_eye_dataset(datasets_dir: Path, output_dir: Path, mode: str) -> dict[str, int]:
    """eye_state_128을 processed/eye 구조로 옮깁니다."""
    source_dir = datasets_dir / "eye_state_128"
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing eye dataset: {source_dir}")

    counts: dict[str, int] = {}
    for split in SPLITS:
        for class_name in ("awake", "sleepy"):
            src_dir = source_dir / split / class_name
            dst_dir = output_dir / "eye" / split / class_name
            key = f"eye/{split}/{class_name}"
            counts[key] = copy_class_folder(src_dir, dst_dir, "eye128", mode)
    return counts


def prepare_mouth_dataset(datasets_dir: Path, output_dir: Path, mode: str) -> dict[str, int]:
    """mouth_state_128을 processed/mouth 구조로 옮깁니다."""
    source_dir = datasets_dir / "mouth_state_128"
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing mouth dataset: {source_dir}")

    counts: dict[str, int] = {}
    for split in SPLITS:
        for class_name in ("normal", "yawn"):
            src_dir = source_dir / split / class_name
            dst_dir = output_dir / "mouth" / split / class_name
            key = f"mouth/{split}/{class_name}"
            counts[key] = copy_class_folder(src_dir, dst_dir, "mouth128", mode)
    return counts


def write_label_maps(output_dir: Path) -> None:
    """추론 코드와 맞는 class 순서를 명시적으로 저장합니다."""
    maps = {
        output_dir / "eye" / "label_map.json": {"awake": 0, "sleepy": 1},
        output_dir / "mouth" / "label_map.json": {"normal": 0, "yawn": 1},
    }
    for path, payload in maps.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def write_summary(output_dir: Path, counts: dict[str, int], mode: str) -> None:
    """생성 결과를 사람이 확인하기 쉬운 JSON으로 남깁니다."""
    summary = {
        "output_dir": str(output_dir),
        "mode": mode,
        "counts": counts,
        "notes": [
            "원본 데이터는 수정하지 않았습니다.",
            "기본 데이터는 eye_state_128, mouth_state_128에서 가져왔습니다.",
            "Processed_Dataset과 preprocessed_data는 보조 데이터로 남겨두었습니다.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare processed training datasets")
    parser.add_argument("--datasets-dir", type=Path, default=base_dir / "datasets")
    parser.add_argument("--output-dir", type=Path, default=base_dir / "datasets" / "processed")
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="hardlink saves disk space when source and target are on the same filesystem",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove output-dir before rebuilding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_dir = args.datasets_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    ensure_clean_target(output_dir, args.clean)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {}
    counts.update(prepare_eye_dataset(datasets_dir, output_dir, args.mode))
    counts.update(prepare_mouth_dataset(datasets_dir, output_dir, args.mode))
    write_label_maps(output_dir)
    write_summary(output_dir, counts, args.mode)

    print(f"Prepared dataset: {output_dir}")
    for key in sorted(counts):
        print(f"{counts[key]:7d}  {key}")


if __name__ == "__main__":
    main()
