"""
Data loading utilities for MS-COCO dataset.
"""

import os
import torch
import torchvision.transforms as transforms
from PIL import Image


def load_coco_images(data_path="extracted_images", num_images=9, image_size=(520, 520)):
    """
    Load MS-COCO images from directory.
    
    Args:
        data_path (str): Path to image directory
        num_images (int): Number of images to load
        image_size (tuple): Size to resize images to
    
    Returns:
        list: List of preprocessed image tensors
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])
    
    image_tensors = []
    image_files = sorted([f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    for i in range(min(num_images, len(image_files))):
        img_path = os.path.join(data_path, image_files[i])
        image = Image.open(img_path).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        image_tensors.append(tensor)
    
    return image_tensors


def download_coco_sample(output_path="coco"):
    """
    Download MS-COCO sample dataset.
    
    Args:
        output_path (str): Directory to save downloaded data
    """
    import urllib.request
    
    os.makedirs(output_path, exist_ok=True)
    url = "https://httpmail.onera.fr/21/050a4e5c4611d260c1b8035b5dc8617eO1A12h/coco_sample.pth"
    file_path = os.path.join(output_path, "coco_sample.pth")
    
    print(f"Downloading MS-COCO sample to {file_path}...")
    urllib.request.urlretrieve(url, file_path)
    print("Download complete!")
    
    return file_path


def extract_images_from_pth(pth_file, output_dir="extracted_images"):
    """
    Extract images from .pth file.
    
    Args:
        pth_file (str): Path to .pth file
        output_dir (str): Directory to save extracted images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    data = torch.load(pth_file)
    to_pil = transforms.ToPILImage()
    
    num_images = data.shape[0]
    for i in range(num_images):
        img_tensor = data[i]
        img = to_pil(img_tensor)
        img.save(os.path.join(output_dir, f"image_{i}.png"))
    
    print(f"Extracted {num_images} images to {output_dir}/")
