#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="data/coco"
ARCHIVE_DIR="$BASE_DIR/archive"
TRAIN_ZIP="$ARCHIVE_DIR/val2017.zip"
ANNOTATIONS_ZIP="$ARCHIVE_DIR/annotations_trainval2017.zip"
TRAIN_DIR="$BASE_DIR/val2017"
ANNOTATIONS_DIR="$BASE_DIR/annotations"

mkdir -p "$ARCHIVE_DIR" "$BASE_DIR"

echo "Downloading COCO 2017 training images..."
if [ ! -f "$TRAIN_ZIP" ]; then
    curl -L --insecure "https://images.cocodataset.org/zips/val2017.zip" -o "$TRAIN_ZIP"
else
    echo "Found $TRAIN_ZIP, skipping download."
fi

echo "Downloading COCO 2017 annotations..."
if [ ! -f "$ANNOTATIONS_ZIP" ]; then
    curl -L --insecure "https://images.cocodataset.org/annotations/annotations_trainval2017.zip" -o "$ANNOTATIONS_ZIP"
else
    echo "Found $ANNOTATIONS_ZIP, skipping download."
fi

echo "Extracting training images..."
if [ ! -d "$TRAIN_DIR" ] || [ -z "$(find "$TRAIN_DIR" -maxdepth 1 -type f -name '*.jpg' 2>/dev/null | head -n 1)" ]; then
    unzip -q "$TRAIN_ZIP" -d "$BASE_DIR"
else
    echo "Training images already extracted."
fi

echo "Extracting annotations..."
if [ ! -d "$ANNOTATIONS_DIR" ] || [ ! -f "$ANNOTATIONS_DIR/instances_train2017.json" ]; then
    unzip -q "$ANNOTATIONS_ZIP" -d "$BASE_DIR"
else
    echo "Annotations already extracted."
fi

echo "COCO download and extraction complete."
