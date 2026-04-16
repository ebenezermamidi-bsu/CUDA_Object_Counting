#!/usr/bin/env bash
set -e

ARCHIVE_DIR="data/sipi/archive"
EXTRACT_DIR="data/sipi/extracted"
URL="https://sipi.usc.edu/database/aerials.tar.gz"

mkdir -p "$ARCHIVE_DIR"
mkdir -p "$EXTRACT_DIR"

ARCHIVE_PATH="$ARCHIVE_DIR/aerials.tar.gz"

echo "Downloading USC SIPI aerials archive..."
curl -L "$URL" -o "$ARCHIVE_PATH"

echo "Extracting archive..."
tar -xzf "$ARCHIVE_PATH" -C "$EXTRACT_DIR"

echo "Done."
echo "Archive: $ARCHIVE_PATH"
echo "Extracted to: $EXTRACT_DIR"