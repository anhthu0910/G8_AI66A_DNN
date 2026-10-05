import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".pgm", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def main():
    parser = argparse.ArgumentParser(description="Create a smaller, stratified image dataset.")
    parser.add_argument("--total", type=int, default=10_000, help="Total number of images to copy.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible sampling.")
    args = parser.parse_args()

    if args.total <= 0:
        parser.error("--total must be greater than zero")

    project_root = Path(__file__).resolve().parent.parent
    source_root = project_root / "data" / "raw"
    output_root = project_root / "data" / "cut"

    if not source_root.is_dir():
        parser.error(f"Source directory does not exist: {source_root}")
    if output_root.exists() and any(output_root.iterdir()):
        parser.error(f"Output directory is not empty: {output_root}. Move or remove its contents before running.")

    source_splits = {split: source_root / split for split in SPLITS}
    missing_splits = [split for split, path in source_splits.items() if not path.is_dir()]
    if missing_splits:
        parser.error(f"Missing source split directories: {', '.join(missing_splits)}")

    class_names = sorted(path.name for path in source_splits["train"].iterdir() if path.is_dir())
    if not class_names:
        parser.error(f"No class directories found in {source_splits['train']}")
    for split, split_path in source_splits.items():
        split_classes = sorted(path.name for path in split_path.iterdir() if path.is_dir())
        if split_classes != class_names:
            parser.error(f"Class directories in {split_path} do not match the train split")

    buckets = {}
    for split, split_path in source_splits.items():
        for class_name in class_names:
            class_path = split_path / class_name
            buckets[(split, class_name)] = sorted(
                path for path in class_path.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

    available = sum(len(files) for files in buckets.values())
    if available < args.total:
        parser.error(f"Requested {args.total:,} images, but found only {available:,} images.")

    quotas = {
        key: len(files) * args.total // available
        for key, files in buckets.items()
    }
    remainder = args.total - sum(quotas.values())
    remainder_order = sorted(
        buckets,
        key=lambda key: (-(len(buckets[key]) * args.total % available), key),
    )
    for key in remainder_order[:remainder]:
        quotas[key] += 1

    rng = random.Random(args.seed)
    copied_counts = {split: 0 for split in SPLITS}
    for key, files in buckets.items():
        split, class_name = key
        selected = rng.sample(files, quotas[key])
        for source_file in selected:
            destination = output_root / source_file.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
        copied_counts[split] += len(selected)
        print(f"{split}/{class_name}: {len(selected):,} images")

    print(f"Total: {sum(copied_counts.values()):,} images")
    print(f"Output: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())