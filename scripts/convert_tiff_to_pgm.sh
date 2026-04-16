#!/usr/bin/env bash
set -e

INPUT_DIR="data/sipi/extracted"
OUTPUT_DIR="data/sipi/converted"

mkdir -p "$OUTPUT_DIR"

found=0
for file in $(find "$INPUT_DIR" -type f \( -iname "*.tif" -o -iname "*.tiff" \)); do
  found=1
  base=$(basename "$file")
  name="${base%.*}"
  convert "$file" -colorspace Gray "$OUTPUT_DIR/${name}.pgm"
done

if [ "$found" -eq 0 ]; then
  echo "No TIFF files found in $INPUT_DIR"
  exit 0
fi

echo "Conversion complete. Files written to $OUTPUT_DIR"
