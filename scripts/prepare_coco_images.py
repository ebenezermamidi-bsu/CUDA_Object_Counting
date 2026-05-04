import argparse
from pathlib import Path
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def collect_images(root: Path):
    return sorted(p for p in root.glob("*.jpg") if p.is_file())


def prepare_image(img_path, output_dir, size):
    out_path = output_dir / f"{img_path.stem}.pgm"

    if out_path.exists():
        return "skipped"

    try:
        with Image.open(img_path) as img:
            gray = img.convert("L")
            fitted = ImageOps.fit(
                gray,
                (size, size),
                method=Image.Resampling.LANCZOS,
                centering=(0.5, 0.5),
            )
            fitted.save(out_path)
        return "done"
    except Exception as exc:
        return f"error: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/coco/val2017")
    parser.add_argument("--output-dir", default="data/coco/prepared")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    workers = os.cpu_count()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = collect_images(source_dir)[: args.limit]

    done = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(prepare_image, img_path, output_dir, args.size): img_path
            for img_path in image_files
        }

        for future in as_completed(futures):
            result = future.result()
            if result == "done":
                done += 1
            elif result == "skipped":
                skipped += 1

    print(f"Prepared {done} images in {output_dir}")
    if skipped:
        print(f"Reused {skipped} existing images")


if __name__ == "__main__":
    main()