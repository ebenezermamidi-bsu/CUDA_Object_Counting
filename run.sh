#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="data/coco"
ARCHIVE_DIR="$DATASET_DIR/archive"
TRAIN_DIR="$DATASET_DIR/val2017"
PREPARED_DIR="$DATASET_DIR/prepared"
MAX_IMAGES="${MAX_IMAGES:-5000}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
THRESHOLD="${THRESHOLD:-140}"
MIN_AREA="${MIN_AREA:-50}"
SAVE_SAMPLES="${SAVE_SAMPLES:-7}"

mkdir -p "$ARCHIVE_DIR" "$PREPARED_DIR"

echo "=== Step 1: Download COCO 2017 dataset files (if not already present) ==="
if [ ! -d "$TRAIN_DIR" ] || [ -z "$(find "$TRAIN_DIR" -maxdepth 1 -type f -name '*.jpg' 2>/dev/null | head -n 1)" ]; then
    bash scripts/download_coco_2017.sh
else
    echo "COCO val2017 images already exist. Skipping download."
fi

echo "=== Step 2: Prepare normalized grayscale images (if needed) ==="
PREPARED_COUNT=$(find "$PREPARED_DIR" -maxdepth 1 -type f -name '*.pgm' 2>/dev/null | wc -l | tr -d ' ')
if [ "$PREPARED_COUNT" -lt "$MAX_IMAGES" ]; then
    echo "Preparing COCO images..."
    python3 scripts/prepare_coco_images.py \
        --source-dir "$TRAIN_DIR" \
        --output-dir "$PREPARED_DIR" \
        --limit "$MAX_IMAGES" \
        --size "$IMAGE_SIZE"
else
    echo "Sufficient prepared images already exist ($PREPARED_COUNT). Skipping preparation."
fi

echo "=== Step 3: Build project ==="
make

echo "=== Step 4: Run pipeline ==="
mkdir -p output/masks output/cleaned output/labeled output/stats

./bin/coco_object_counter \
  --input-dir "$PREPARED_DIR" \
  --max-images "$MAX_IMAGES" \
  --output-dir output \
  --threshold "$THRESHOLD" \
  --min-area "$MIN_AREA" \
  --save-samples "$SAVE_SAMPLES"

echo "=== Step 5: Convert outputs to PNG ==="
python3 scripts/convert_output_pgm_to_png.py --output-dir output

echo "=== Done ==="
