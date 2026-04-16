import argparse
from pathlib import Path
from PIL import Image, ImageOps


def collect_images(root: Path):
    return sorted(p for p in root.glob("*.jpg") if p.is_file())


def prepare_image(img: Image.Image, size: int) -> Image.Image:
    gray = img.convert("L")
    fitted = ImageOps.fit(gray, (size, size), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return fitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/coco/val2017")
    parser.add_argument("--output-dir", default="data/coco/prepared")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = collect_images(source_dir)
    if not image_files:
        print(f"No JPEG source images found under {source_dir}")
        return

    prepared = 0
    skipped_existing = 0

    for img_path in image_files:
        if prepared >= args.limit:
            break

        out_path = output_dir / f"{img_path.stem}.pgm"
        if out_path.exists():
            skipped_existing += 1
            prepared += 1
            continue

        try:
            with Image.open(img_path) as img:
                result = prepare_image(img, args.size)
                result.save(out_path)
                prepared += 1
        except Exception as exc:
            print(f"Skipping {img_path.name}: {exc}")

    print(f"Prepared {prepared} images in {output_dir}")
    if skipped_existing:
        print(f"Reused {skipped_existing} existing prepared images")


if __name__ == "__main__":
    main()
