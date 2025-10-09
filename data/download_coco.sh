#!/bin/bash

# Script to download MS-COCO sample dataset
# Authors: Yassine Zanned, Seifeddine Ghozzi, Ameni Hidouri

set -e  # Exit on error

echo "=========================================="
echo "MS-COCO Dataset Download Script"
echo "=========================================="

# Configuration
COCO_DIR="coco"
EXTRACTED_DIR="extracted_images"
COCO_URL="https://httpmail.onera.fr/21/050a4e5c4611d260c1b8035b5dc8617eO1A12h/coco_sample.pth"
COCO_FILE="coco_sample.pth"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p "$COCO_DIR"
mkdir -p "$EXTRACTED_DIR"
echo "✓ Directories created"

# Download dataset
echo ""
echo "Downloading MS-COCO sample dataset..."
if [ -f "$COCO_DIR/$COCO_FILE" ]; then
    echo "⚠ File already exists: $COCO_DIR/$COCO_FILE"
    read -p "Do you want to re-download? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        wget -O "$COCO_DIR/$COCO_FILE" "$COCO_URL"
        echo "✓ Dataset downloaded"
    else
        echo "⊘ Skipping download"
    fi
else
    wget -O "$COCO_DIR/$COCO_FILE" "$COCO_URL"
    echo "✓ Dataset downloaded"
fi

# Extract images using Python
echo ""
echo "Extracting images from .pth file..."
python3 << EOF
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Load the .pth file
print("Loading .pth file...")
data = torch.load("$COCO_DIR/$COCO_FILE")
print(f"Dataset shape: {data.shape}")

# Extract images
to_pil = transforms.ToPILImage()
num_images = data.shape[0]

print(f"Extracting {num_images} images...")
for i in range(num_images):
    img_tensor = data[i]
    img = to_pil(img_tensor)
    img.save(os.path.join("$EXTRACTED_DIR", f"image_{i}.png"))
    if (i + 1) % 10 == 0 or i == num_images - 1:
        print(f"  Extracted {i + 1}/{num_images} images")

print("✓ All images extracted successfully!")
EOF

echo ""
echo "=========================================="
echo "Download and extraction completed!"
echo "Images saved to: $EXTRACTED_DIR/"
echo "=========================================="
