import argparse
from pathlib import Path
from PIL import Image

def convert_dir(path: Path):
    if not path.exists():
        return
    for pgm in path.glob("*.pgm"):
        png = pgm.with_suffix(".png")
        img = Image.open(pgm)
        img.save(png)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    base = Path(args.output_dir)
    convert_dir(base / "masks")
    convert_dir(base / "cleaned")
    convert_dir(base / "labeled")
    print(f"PNG conversion complete under {base}")

if __name__ == "__main__":
    main()
