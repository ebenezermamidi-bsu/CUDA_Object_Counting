# Installation Guide

## System Prerequisites

This project was developed and tested on the following setup:

### Hardware

* NVIDIA GPU (Tested on RTX 4070 Laptop GPU)

### Operating System

* Windows 11 with WSL2 (Ubuntu 22.04)

---

## Software Requirements

### NVIDIA Drivers

Verify:

```bash
nvidia-smi
```

### CUDA Toolkit

Verify:

```bash
nvcc --version
```

---

### Build Tools

```bash
sudo apt update
sudo apt install -y build-essential make python3 python3-pil
```

---

### Dataset Preparation Tools

```bash
sudo apt install -y imagemagick curl wget unzip
```

If you see:
```
unzip: command not found
```

Install it with:

```bash
sudo apt install -y unzip
```

---

## Quick Setup Script

```bash
sudo apt update

# Build + Python
sudo apt install -y build-essential make python3 python3-pil

# Dataset tools
sudo apt install -y imagemagick curl wget unzip

# CUDA paths (if not already set)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

---

## Dataset Preparation (COCO)

### 1. Download COCO dataset

```bash
bash scripts/download_coco_2017.sh
```

---

### 2. Prepare images

```bash
python3 scripts/prepare_coco_images.py --limit 5000
```

This will:
- convert images to grayscale  
- resize to 256×256  
- write outputs to:

```
data/coco/prepared/
```

---

## Build

```bash
make clean
make
```

---

## Run

### Full pipeline

```bash
bash run.sh
```

---

### Manual run example

```bash
./bin/coco_object_counter   --input-dir data/coco/prepared   --max-images 5000   --output-dir output   --threshold 140   --min-area 50   --save-samples 7
```

---

## Output Review

After the run, review:

```
output/stats/processing_summary.csv
output/stats/object_stats.csv
output/run_log.txt
output/masks/
output/cleaned/
output/labeled/
```

---
