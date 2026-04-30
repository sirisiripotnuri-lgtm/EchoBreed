from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, UnidentifiedImageError

DEFAULT_RAW_DIR = Path("data/stanford-dogs")
DEFAULT_OUTPUT_DIR = Path("data/processed-dogs")


def clean_breed_name(folder_name: str) -> str:
    return folder_name.split("-", 1)[-1].replace("_", " ")


def parse_box(annotation_path: Path) -> tuple[int, int, int, int] | None:
    root = ET.parse(annotation_path).getroot()
    boxes = []

    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = int(float(box.findtext("xmin", "0")))
        ymin = int(float(box.findtext("ymin", "0")))
        xmax = int(float(box.findtext("xmax", "0")))
        ymax = int(float(box.findtext("ymax", "0")))
        area = max(0, xmax - xmin) * max(0, ymax - ymin)
        boxes.append((area, xmin, ymin, xmax, ymax))

    if not boxes:
        return None

    _, xmin, ymin, xmax, ymax = max(boxes, key=lambda item: item[0])
    return xmin, ymin, xmax, ymax


def clamp_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = box
    return (
        max(0, min(xmin, width - 1)),
        max(0, min(ymin, height - 1)),
        max(1, min(xmax, width)),
        max(1, min(ymax, height)),
    )


def prepare_dataset(raw_dir: Path, output_dir: Path, crop: bool) -> None:
    images_dir = raw_dir / "Images"
    annotations_dir = raw_dir / "Annotation"

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(images_dir.glob("*/*.jpg"))
    processed = 0
    skipped = 0

    for image_path in image_paths:
        source_folder = image_path.parent.name
        breed_name = clean_breed_name(source_folder)
        target_dir = output_dir / breed_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / image_path.name

        try:
            if crop and annotations_dir.exists():
                annotation_path = annotations_dir / source_folder / image_path.stem
                box = parse_box(annotation_path) if annotation_path.exists() else None

                with Image.open(image_path) as image:
                    rgb_image = image.convert("RGB")
                    if box is not None:
                        rgb_image = rgb_image.crop(clamp_box(box, *rgb_image.size))
                    rgb_image.save(target_path, quality=95)
            else:
                shutil.copy2(image_path, target_path)
            processed += 1
        except (OSError, UnidentifiedImageError, ET.ParseError):
            skipped += 1

    print(f"Prepared {processed} images in {output_dir}")
    print(f"Skipped {skipped} unreadable files")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Stanford Dogs images for training.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-crop", action="store_true", help="Copy full images instead of cropping with annotations.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args.raw_dir, args.output_dir, crop=not args.no_crop)
