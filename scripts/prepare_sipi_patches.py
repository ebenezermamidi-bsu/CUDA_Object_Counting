import argparse
from pathlib import Path
from PIL import Image

def collect_images(root: Path):
    exts = {".tif", ".tiff", ".TIF", ".TIFF", ".pgm", ".ppm", ".png", ".jpg", ".jpeg", ".bmp"}
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in exts:
            files.append(p)
    return sorted(files)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/sipi/extracted")
    parser.add_argument("--output-dir", default="data/sipi/patches")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--max-per-image", type=int, default=300)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = collect_images(source_dir)
    if not image_files:
        print(f"No source images found under {source_dir}")
        return

    patch_count = 0
    for img_path in image_files:
        if patch_count >= args.limit:
            break

        patches_from_this_image = 0

        with Image.open(img_path) as img:
            gray = img.convert("L")
            width, height = gray.size

            if width < args.patch_size or height < args.patch_size:
                continue

            stem = img_path.stem.replace(" ", "_")

            y_positions = list(range(0, height - args.patch_size + 1, args.stride))
            x_positions = list(range(0, width - args.patch_size + 1, args.stride))

            for top in y_positions:
                for left in x_positions:
                    patch = gray.crop((left, top, left + args.patch_size, top + args.patch_size))
                    patch_name = f"{stem}_{left:04d}_{top:04d}.pgm"
                    patch.save(output_dir / patch_name)
                    patch_count += 1
                    patches_from_this_image += 1

                    if patch_count >= args.limit:
                        break
                    if patches_from_this_image >= args.max_per_image:
                        break

                if patch_count >= args.limit:
                    break
                if patches_from_this_image >= args.max_per_image:
                    break

    print(f"Prepared {patch_count} patch images in {output_dir}")

if __name__ == "__main__":
    main()
