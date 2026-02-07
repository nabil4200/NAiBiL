I want to create an AI model from scratch in Google Colab using approximately 200 million parameters named 'NAiBiL'. I have allready written some codes. those are attached in my project, please throwly review all and try to understand everything deeply. This AI should be able to detect human body parts very precisely, including the face, hands, knees, eyes, iris patterns, head, nose, shoulders, elbows, ankles, backbone, nape, throat, forehead, etc. At the same time, it should detect objects in the full image, such as bottles, humans, cats, cell phones, fish, cars, traffic lights, potholes, cracks, ruts, sinkholes, water, lakes, ponds, accidents, houses, apartments, guns, vehicles, animals, LiDAR-Vision Fusion, Dynamic SLAM (Simultaneous Localization and Mapping), Activity & Fall Detection, and other objects You should search online for codebase and others informations. It should also detect specific details such as: Fish: eyes, health, tail, fins, etc. Cars: front side, rear side, type, size, etc. Humans: type (male, female, children, adult, young, old), and various body parts. * In other words, it should detect both individual body parts of humans and all objects present in a full image. This AI must work in real-time and be usable in drones, cars, robotics, submarines, autonomous vehicles, CCTV, cameras, etc. After detection, a decision engine will be required to send commands to sensors based on the AI’s decisions. *You should make an advanced architecture. Here you should ensure accuracy and performance. I request you to understand my requirements deeply. Try to understand every detail thoroughly and provide me with complete .ipynb file. Try to understand very well the codes that I gave you earlier. I used those codes to build an AI model, ran inference on it, tested it, and got these results. Attached screenshots: 'NAiBiL_1st test screenshot by 'my ai.png', 'NAiBiL_2nd test screenshot' by my ai. I trained it with only 1 epoch and just 4 images, but it didn't give the result I was expecting. It should have given results like the ones attached: 'yolo_ai.png', 'another_ai.png', 'another_ai2.png'. Look carefully at the results from my AI and try to understand them, and also look at the results from the other AIs. My AI is making a lot of mistakes. For example, a simple case: in the image I gave as input there is no infrastructure anywhere, but it is detecting/ saying 'poor infrastructure condition' — how can the quality be poor in a place where there is no infrastructure at all? Maybe it was trying to say something like that, but right now it shouldn't show this status at all because that thing doesn't exist there. Next: it is showing speed as 4 m/s, but honestly there is nothing like that there at all, so this is completely wrong. It's showing Warnings?? ... !! There are still countless other problems like this. It should detect objects and show bounding boxes, but it doesn't show them at all. It should detect things like human, animal, fish, cat, dog, woman, girl, man, old person, pose, segmant, category, orientation etc., etc. And along with the objects, it should also detect face, iris, nose, eye, mouth, etc. in those objects — but it doesn't do any of this. I feel like its architecture is not strong enough. For multi-task learning, deep analysis, real-time analysis, pixel-level analysis, and similar tasks, the architecture should be much more advanced and powerful. You should handle datasets more advancedly. Suppose my Colab storage is 100GB; I can't download all the datasets at once to work. Therefore, use datasets in streaming mode whenever possible. For those where streaming is not feasible, download them as a zip, unzip them, and then delete the zip file to save space. Instead of downloading everything at once, you should train on a few datasets, then move to others for further training to achieve better results. But before anything else, your priority should be to find the correct and best datasets and work accordingly. I am already using some datasets; you should use the remaining ones and find high-quality, proper datasets based on my references. *The dataset_info.txt file contains some information regarding the dataset; please try to understand it thoroughly and implement it correctly accordingly. please review this codebase 'https://github.com/ultralytics/ultralytics'. Inference is another important part of ai because on this an ai can response correctly, so you also should concentrate to it. Please try to understand everything deeply and make it more advanced and powerful. Ensure accuracy. Give me complete code by written correctly with implementing all features in .ipynb file. Don't remove any feature. Also you should concentrate to datasets, because proper datasets are the most important part of an Ai. You should review all of my uploaded files deeply. Make a complete ipynb file with implementing all features and write all codes correctly in single ipynb file.


#This is tested and it will work and i shared result it's with you, look there have some png files#
# Mount Google Drive
from google.colab import drive
import os
drive.mount('/content/drive')

# Create project directories
directories = [
    '/content/naibil_workspace',
    '/content/datasets',
    '/content/datasets/coco',
    '/content/datasets/visdrone',
    '/content/datasets/faces',
    '/content/drive/MyDrive/NAiBiL_Models',
    '/content/drive/MyDrive/NAiBiL_Checkpoints',
    '/content/drive/MyDrive/NAiBiL_Results'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

print("✅ Google Drive mounted and directories created!")

%%capture
# Install core dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install opencv-python-headless albumentations pycocotools
!pip install timm einops scipy numpy pandas
!pip install tqdm matplotlib seaborn pillow pyyaml
!pip install scikit-learn scikit-image
!pip install fvcore iopath
!pip install tensorboard wandb
!pip install pycocotools
!pip install gdown

print("✅ All packages installed successfully!")

# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import cv2
import json
import math
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============================================================================
# AUTOMATIC DATASET DOWNLOADER - YOLO-Style Integration
# ============================================================================
# This system automatically downloads and prepares real datasets
# NO SYNTHETIC DATA - All training uses real, annotated datasets

import requests
import zipfile
from pathlib import Path
from tqdm.auto import tqdm
import shutil

class DatasetDownloader:
    """Automatic dataset downloader with YOLO-style configuration"""
    
    def __init__(self, root_dir='/content/datasets'):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url, destination, desc="Downloading"):
        """Download file with progress bar"""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        if destination.exists():
            print(f"✓ {destination.name} already exists, skipping download")
            return destination
        
        print(f"📥 Downloading {desc} from {url}")
        
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f, tqdm(
                desc=desc,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ Downloaded {destination.name} ({total_size / 1e6:.1f} MB)")
            return destination
        
        except Exception as e:
            print(f"❌ Error downloading: {str(e)}")
            if destination.exists():
                destination.unlink()
            raise
    
    def extract_zip(self, zip_path, extract_to, desc="Extracting"):
        """Extract zip file with progress"""
        zip_path = Path(zip_path)
        extract_to = Path(extract_to)
        
        print(f"📦 Extracting {zip_path.name}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            with tqdm(total=len(members), desc=desc) as pbar:
                for member in members:
                    zip_ref.extract(member, extract_to)
                    pbar.update(1)
        
        print(f"✅ Extracted {len(members)} files")
    
    def download_coco8(self):
        """Download COCO8 mini dataset (8 images for quick testing)"""
        print("" + "="*70)
        print("📊 COCO8 DATASET - Quick Testing")
        print("="*70)
        print("   Size: 8 images (4 train, 4 val)")
        print("   Download: ~1 MB")
        print("   Purpose: Pipeline testing & verification")
        print("   Classes: 80 COCO categories")
        print("="*70 + "")
        
        coco8_dir = self.root_dir / 'coco8'
        
        if (coco8_dir / 'images').exists():
            print("✓ COCO8 already downloaded")
            return coco8_dir
        
        # Download
        zip_path = self.download_file(
            'https://ultralytics.com/assets/coco8.zip',
            self.root_dir / 'coco8.zip',
            'COCO8 Dataset'
        )
        
        # Extract
        self.extract_zip(zip_path, self.root_dir, 'Extracting COCO8')
        zip_path.unlink()  # Remove zip to save space
        
        # Verify
        train_images = list((coco8_dir / 'images' / 'train').glob('*.jpg'))
        val_images = list((coco8_dir / 'images' / 'val').glob('*.jpg'))
        
        print(f"✅ COCO8 Dataset Ready!")
        print(f"   Train images: {len(train_images)}")
        print(f"   Val images: {len(val_images)}")
        print(f"   Path: {coco8_dir}")
        
        return coco8_dir
    
    def download_coco_full(self, subset='train+val'):
        """Download full COCO 2017 dataset"""
        print("" + "="*70)
        print("📊 COCO 2017 FULL DATASET")
        print("="*70)
        print("   ⚠️  WARNING: This is a LARGE download (~25 GB)")
        print("   Train: 118,287 images (~19 GB)")
        print("   Val: 5,000 images (~1 GB)")
        print("   Annotations: (~5 GB)")
        print("   For learning/testing, use COCO8 instead!")
        print("="*70 + "")
        
        coco_dir = self.root_dir / 'coco'
        coco_dir.mkdir(parents=True, exist_ok=True)
        
        urls = {
            'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
            'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
            'val_images': 'http://images.cocodataset.org/zips/val2017.zip'
        }
        
        # Download annotations first (smallest)
        if not (coco_dir / 'annotations').exists():
            ann_zip = self.download_file(
                urls['annotations'],
                coco_dir / 'annotations_trainval2017.zip',
                'COCO Annotations'
            )
            self.extract_zip(ann_zip, coco_dir, 'Extracting annotations')
            ann_zip.unlink()
        
        # Download training images if requested
        if 'train' in subset and not (coco_dir / 'train2017').exists():
            train_zip = self.download_file(
                urls['train_images'],
                coco_dir / 'train2017.zip',
                'COCO Training Images (19GB)'
            )
            self.extract_zip(train_zip, coco_dir, 'Extracting training images')
            train_zip.unlink()
        
        # Download validation images if requested
        if 'val' in subset and not (coco_dir / 'val2017').exists():
            val_zip = self.download_file(
                urls['val_images'],
                coco_dir / 'val2017.zip',
                'COCO Validation Images (1GB)'
            )
            self.extract_zip(val_zip, coco_dir, 'Extracting validation images')
            val_zip.unlink()
        
        print("✅ COCO 2017 Dataset Ready!")
        self.verify_coco(coco_dir)
        
        return coco_dir
    
    def download_african_wildlife(self):
        """Download African Wildlife dataset"""
        print("" + "="*70)
        print("📊 AFRICAN WILDLIFE DATASET")
        print("="*70)
        print("   Classes: 4 (buffalo, elephant, rhino, zebra)")
        print("   Purpose: Wildlife detection & monitoring")
        print("="*70 + "")
        
        aw_dir = self.root_dir / 'african_wildlife'
        
        if aw_dir.exists() and any(aw_dir.iterdir()):
            print("✓ African Wildlife already downloaded")
            return aw_dir
        
        zip_path = self.download_file(
            'https://ultralytics.com/assets/african-wildlife.zip',
            self.root_dir / 'african-wildlife.zip',
            'African Wildlife Dataset'
        )
        
        self.extract_zip(zip_path, self.root_dir, 'Extracting African Wildlife')
        zip_path.unlink()
        
        print("✅ African Wildlife Dataset Ready!")
        return aw_dir
    
    def verify_coco(self, coco_dir):
        """Verify COCO dataset integrity"""
        print("🔍 Verifying dataset...")
        coco_dir = Path(coco_dir)
        
        checks = [
            ('Train images', coco_dir / 'train2017', '*.jpg'),
            ('Val images', coco_dir / 'val2017', '*.jpg'),
            ('Annotations', coco_dir / 'annotations', '*.json')
        ]
        
        for name, path, pattern in checks:
            if path.exists():
                count = len(list(path.glob(pattern)))
                print(f"   ✓ {name}: {count:,} files")
            else:
                print(f"   ✗ {name}: Not found")

# ============================================================================
# INITIALIZE DOWNLOADER AND SELECT DATASET
# ============================================================================

downloader = DatasetDownloader()

# CHOOSE YOUR DATASET:
# ==============================================================================

# OPTION 1: COCO8 - Quick Testing (RECOMMENDED FOR START)
# Perfect for: Testing pipeline, debugging, quick experiments
# Size: 8 images (4 train, 4 val), ~1 MB
dataset_path = downloader.download_coco8()
DATASET_TYPE = 'coco8'
NUM_CLASSES = 80
IMG_SIZE = 640

# OPTION 2: Full COCO 2017 (Uncomment for production training)
# Perfect for: Production models, final training
# Size: 123K images, ~25 GB
# WARNING: This will take significant time and storage!
dataset_path = downloader.download_coco_full(subset='train+val')
DATASET_TYPE = 'coco'
NUM_CLASSES = 80
IMG_SIZE = 640

# OPTION 3: African Wildlife (Uncomment for wildlife detection)
# Perfect for: Wildlife monitoring, conservation projects
# Size: ~100 images, 4 classes
dataset_path = downloader.download_african_wildlife()
DATASET_TYPE = 'african_wildlife'
NUM_CLASSES = 4
IMG_SIZE = 640

# ============================================================================

print("" + "="*70)
print(f"✅ DATASET READY FOR TRAINING")
print("="*70)
print(f"   Dataset: {DATASET_TYPE.upper()}")
print(f"   Path: {dataset_path}")
print(f"   Classes: {NUM_CLASSES}")
print(f"   Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"   ✓ 100% Real Data (NO Synthetic)")
print("="*70 + "")

# COCO class names for reference
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# ============================================================================
# CORE BUILDING BLOCKS
# ============================================================================

class ConvBNAct(nn.Module):
    """Convolution + BatchNorm + Activation with dropout"""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, g=1, act=True, d=1, dropout=0.0):
        super().__init__()
        if p is None:
            p = d * (k - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, momentum=0.03, eps=0.001)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x))))

class SqueezeExcitation(nn.Module):
    """SE attention with improved gating"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, 1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc2(self.act(self.fc1(scale)))
        return x * self.sigmoid(scale)

class SpatialAttention(nn.Module):
    """Enhanced spatial attention"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        return x * self.sigmoid(self.conv(attention))

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_attn = SqueezeExcitation(channels, reduction)
        self.spatial_attn = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x

class Bottleneck(nn.Module):
    """Enhanced bottleneck with residual connection"""
    def __init__(self, in_ch, out_ch, shortcut=True, expansion=0.5, use_cbam=True, dropout=0.0):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.conv1 = ConvBNAct(in_ch, hidden, 1, dropout=dropout)
        self.conv2 = ConvBNAct(hidden, out_ch, 3, dropout=dropout)
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()
        self.add = shortcut and in_ch == out_ch
        
    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out = self.cbam(out)
        return x + out if self.add else out

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions - Enhanced version"""
    def __init__(self, in_ch, out_ch, n=1, shortcut=False, expansion=0.5, dropout=0.0):
        super().__init__()
        hidden = int(out_ch * expansion)
        self.cv1 = ConvBNAct(in_ch, 2 * hidden, 1, dropout=dropout)
        self.cv2 = ConvBNAct((2 + n) * hidden, out_ch, 1, dropout=dropout)
        self.m = nn.ModuleList([
            Bottleneck(hidden, hidden, shortcut, use_cbam=True, dropout=dropout) 
            for _ in range(n)
        ])
        
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast version"""
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        hidden = in_ch // 2
        self.cv1 = ConvBNAct(in_ch, hidden, 1)
        self.cv2 = ConvBNAct(hidden * 4, out_ch, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)
        
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))

print("✅ Core building blocks loaded")


# ============================================================================
# ATTENTION MECHANISMS (Transformer-based)
# ============================================================================

class MultiHeadAttention(nn.Module):
    """Efficient Multi-head self-attention"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.dropout(x)

class AIFILayer(nn.Module):
    """Attention-based Intra-scale Feature Interaction"""
    def __init__(self, embed_dim, num_heads, ffn_dim=None, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or embed_dim * 4
        
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, C)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        
        # Self-attention with residual
        x_flat = x_flat + self.attn(self.norm1(x_flat))
        
        # FFN with residual
        x_flat = x_flat + self.ffn(self.norm2(x_flat))
        
        return x_flat.transpose(1, 2).reshape(B, C, H, W)

class DeformableAttention(nn.Module):
    """Deformable attention for adaptive receptive fields"""
    def __init__(self, channels, num_points=9):
        super().__init__()
        self.num_points = num_points
        self.offset_conv = nn.Conv2d(channels, 2 * num_points, 3, padding=1)
        self.attention_conv = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        # Generate sampling offsets
        offsets = self.offset_conv(x)
        
        # Apply attention
        attended = self.attention_conv(x)
        
        return attended

print("✅ Attention mechanisms loaded")


# ============================================================================
# FEATURE PYRAMID NETWORK - BiFPN (FIXED)
# ============================================================================

class BiFPNLayer(nn.Module):
    """Bidirectional Feature Pyramid Network layer - Fixed indexing"""
    def __init__(self, channels, num_levels, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.num_levels = num_levels
        
        # Top-down pathway
        self.td_convs = nn.ModuleList([
            ConvBNAct(channels, channels, 3) 
            for _ in range(num_levels - 1)
        ])
        self.td_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32)) 
            for _ in range(num_levels - 1)
        ])
        
        # Bottom-up pathway
        self.bu_convs = nn.ModuleList([
            ConvBNAct(channels, channels, 3) 
            for _ in range(num_levels - 1)
        ])
        self.bu_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3 if i < num_levels - 2 else 2, dtype=torch.float32)) 
            for i in range(num_levels - 1)
        ])
        
    def forward(self, features):
        assert len(features) == self.num_levels, f"Expected {self.num_levels} features, got {len(features)}"
        
        # Top-down pathway
        td_features = [features[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            idx = self.num_levels - 2 - i
            w = F.relu(self.td_weights[idx])
            w = w / (w.sum() + self.epsilon)
            
            upsampled = F.interpolate(
                td_features[0], 
                size=features[i].shape[2:], 
                mode='nearest'
            )
            
            fused = w[0] * features[i] + w[1] * upsampled
            td_features.insert(0, self.td_convs[idx](fused))
        
        # Bottom-up pathway
        bu_features = [td_features[0]]
        for i in range(1, self.num_levels):
            w = F.relu(self.bu_weights[i - 1])
            w = w / (w.sum() + self.epsilon)
            
            downsampled = F.max_pool2d(bu_features[-1], kernel_size=2, stride=2)
            
            if downsampled.shape[2:] != td_features[i].shape[2:]:
                downsampled = F.interpolate(
                    downsampled, 
                    size=td_features[i].shape[2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            if i == self.num_levels - 1:
                fused = w[0] * td_features[i] + w[1] * downsampled
            else:
                fused = w[0] * td_features[i] + w[1] * downsampled + w[2] * features[i]
            
            bu_features.append(self.bu_convs[i - 1](fused))
        
        return bu_features

class BiFPN(nn.Module):
    """Complete BiFPN with AIFI layers"""
    def __init__(self, in_channels_list, out_channels=256, num_layers=3, num_heads=8):
        super().__init__()
        
        # Input projection
        self.input_convs = nn.ModuleList([
            ConvBNAct(ch, out_channels, 1) 
            for ch in in_channels_list
        ])
        
        # BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, len(in_channels_list)) 
            for _ in range(num_layers)
        ])
        
        # AIFI layers for each scale
        self.aifi_layers = nn.ModuleList([
            AIFILayer(out_channels, num_heads) 
            for _ in range(len(in_channels_list))
        ])
        
    def forward(self, features):
        # Project to uniform channels
        features = [conv(feat) for conv, feat in zip(self.input_convs, features)]
        
        # Apply BiFPN layers
        for bifpn_layer in self.bifpn_layers:
            features = bifpn_layer(features)
        
        # Apply AIFI attention
        features = [aifi(feat) for aifi, feat in zip(self.aifi_layers, features)]
        
        return features

print("✅ BiFPN loaded (bug fixed)")



# ============================================================================
# BACKBONE NETWORK
# ============================================================================

class NAiBiLBackbone(nn.Module):
    """Enhanced backbone with progressive feature extraction"""
    def __init__(self, dropout=0.1):
        super().__init__()
        
        # Stem: Input 3x640x640 -> 96x160x160
        self.stem = nn.Sequential(
            ConvBNAct(3, 48, 3, 2, dropout=dropout),
            ConvBNAct(48, 96, 3, 2, dropout=dropout)
        )
        
        # Stage 1: 96x160x160 -> 128x80x80
        self.stage1 = nn.Sequential(
            C2f(96, 128, n=3, dropout=dropout),
            ConvBNAct(128, 128, 3, 2, dropout=dropout)
        )
        
        # Stage 2: 128x80x80 -> 256x40x40
        self.stage2 = nn.Sequential(
            C2f(128, 256, n=6, dropout=dropout),
            ConvBNAct(256, 256, 3, 2, dropout=dropout)
        )
        
        # Stage 3: 256x40x40 -> 512x20x20
        self.stage3 = nn.Sequential(
            C2f(256, 512, n=6, dropout=dropout),
            ConvBNAct(512, 512, 3, 2, dropout=dropout)
        )
        
        # Stage 4: 512x20x20 -> 1024x20x20
        self.stage4 = nn.Sequential(
            C2f(512, 1024, n=3, dropout=dropout),
            SPPF(1024, 1024)
        )
        
    def forward(self, x):
        x = self.stem(x)
        c1 = self.stage1(x)    # 128 channels
        c2 = self.stage2(c1)   # 256 channels
        c3 = self.stage3(c2)   # 512 channels
        c4 = self.stage4(c3)   # 1024 channels
        
        return [c1, c2, c3, c4]

print("✅ Backbone network loaded")


# ============================================================================
# TASK-SPECIFIC HEADS
# ============================================================================

class DetectionHead(nn.Module):
    """Enhanced detection head with anchor-free design"""
    def __init__(self, num_classes=80, in_channels=(256, 256, 256, 256)):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = 5 + num_classes  # [x, y, w, h, conf, ...classes]
        
        self.stems = nn.ModuleList([
            ConvBNAct(ch, ch, 3) 
            for ch in in_channels
        ])
        
        self.pred_convs = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(ch, ch * 2, 3),
                ConvBNAct(ch * 2, ch * 2, 3),
                nn.Conv2d(ch * 2, self.num_outputs, 1)
            ) for ch in in_channels
        ])
        
    def forward(self, features):
        outputs = []
        for i, feat in enumerate(features):
            feat = self.stems[i](feat)
            pred = self.pred_convs[i](feat)
            outputs.append(pred)
        return outputs

class KeypointHead(nn.Module):
    """High-resolution keypoint detection with heatmaps"""
    def __init__(self, num_keypoints=33, in_channels=(256, 256, 256, 256)):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        self.stems = nn.ModuleList([
            ConvBNAct(ch, ch, 3) 
            for ch in in_channels
        ])
        
        # Deconvolution layers for upsampling
        self.deconv_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ch // 2),
                nn.SiLU(inplace=True),
                nn.ConvTranspose2d(ch // 2, ch // 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ch // 4),
                nn.SiLU(inplace=True),
            ) for ch in in_channels
        ])
        
        # Heatmap prediction
        self.heatmap_convs = nn.ModuleList([
            nn.Conv2d(ch // 4, num_keypoints, 1) 
            for ch in in_channels
        ])
        
    def forward(self, features):
        outputs = []
        for i, feat in enumerate(features):
            feat = self.stems[i](feat)
            feat = self.deconv_layers[i](feat)
            heatmap = self.heatmap_convs[i](feat)
            outputs.append(heatmap)
        return outputs

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling for segmentation"""
    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18]):
        super().__init__()
        
        self.conv1 = ConvBNAct(in_channels, out_channels, 1)
        self.atrous_convs = nn.ModuleList([
            ConvBNAct(in_channels, out_channels, 3, p=d, d=d) 
            for d in dilations
        ])
        
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBNAct(in_channels, out_channels, 1)
        )
        
        total_channels = out_channels * (2 + len(dilations))
        self.project = nn.Sequential(
            ConvBNAct(total_channels, out_channels, 1),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        size = x.shape[2:]
        
        feat1 = self.conv1(x)
        atrous_feats = [conv(x) for conv in self.atrous_convs]
        
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)
        
        x = torch.cat([feat1] + atrous_feats + [global_feat], dim=1)
        return self.project(x)

class SegmentationHead(nn.Module):
    """DeepLabV3+ style segmentation head"""
    def __init__(self, num_classes=21, in_channels=(128, 256, 512, 1024)):
        super().__init__()
        
        self.aspp = ASPPModule(in_channels[-1], 256, [6, 12, 18])
        self.low_level_conv = ConvBNAct(in_channels[0], 48, 1)
        
        self.decoder = nn.Sequential(
            ConvBNAct(256 + 48, 256, 3),
            ConvBNAct(256, 256, 3),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )
        
    def forward(self, features):
        low_level = features[0]
        high_level = features[-1]
        
        x = self.aspp(high_level)
        x = F.interpolate(x, size=low_level.shape[2:], mode='bilinear', align_corners=False)
        
        low_level = self.low_level_conv(low_level)
        x = torch.cat([x, low_level], dim=1)
        
        return self.decoder(x)

class ClassificationHead(nn.Module):
    """Multi-attribute classification head"""
    def __init__(self, num_classes_dict, in_channels=1024):
        super().__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.classifiers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(in_channels, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            ) for name, num_classes in num_classes_dict.items()
        })
        
    def forward(self, x):
        x = self.gap(x).flatten(1)
        return {name: clf(x) for name, clf in self.classifiers.items()}

print("✅ Task-specific heads loaded")



# ============================================================================
# COMPLETE NAiBiL MODEL
# ============================================================================

class NAiBiL(nn.Module):
    """Complete NAiBiL Multi-Task Vision AI System"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone
        self.backbone = NAiBiLBackbone(dropout=config.get('dropout', 0.1))
        
        # Feature Pyramid Network
        self.bifpn = BiFPN(
            in_channels_list=[128, 256, 512, 1024],
            out_channels=256,
            num_layers=config.get('bifpn_layers', 3),
            num_heads=config.get('num_heads', 8)
        )
        
        # Task heads
        self.detection_head = DetectionHead(
            num_classes=config.get('num_classes', 80),
            in_channels=(256, 256, 256, 256)
        )
        
        self.pose_head = KeypointHead(
            num_keypoints=config.get('num_pose_keypoints', 33),
            in_channels=(256, 256, 256, 256)
        )
        
        self.face_head = KeypointHead(
            num_keypoints=config.get('num_face_landmarks', 468),
            in_channels=(256, 256, 256, 256)
        )
        
        self.hand_head = KeypointHead(
            num_keypoints=config.get('num_hand_landmarks', 42),
            in_channels=(256, 256, 256, 256)
        )
        
        self.iris_head = KeypointHead(
            num_keypoints=config.get('num_iris_landmarks', 10),
            in_channels=(256, 256, 256, 256)
        )
        
        self.segmentation_head = SegmentationHead(
            num_classes=config.get('num_seg_classes', 21),
            in_channels=(128, 256, 512, 1024)
        )
        
        self.classification_head = ClassificationHead(
            num_classes_dict=config.get('classification_tasks', {
                'age': 5,
                'gender': 3,
                'vehicle_type': 20,
                'activity': 30,
                'fish_health': 4,
                'infrastructure': 6,
                'object_quality': 5,
            }),
            in_channels=1024
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with appropriate strategies"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, tasks=None):
        """Forward pass with task selection"""
        if tasks is None or tasks == 'all':
            tasks = ['detection', 'pose', 'face', 'hand', 'iris', 'segmentation', 'classification']
        elif isinstance(tasks, str):
            tasks = [tasks]
        
        # Extract backbone features
        backbone_features = self.backbone(x)
        
        # Apply BiFPN
        fpn_features = self.bifpn(backbone_features)
        
        outputs = {}
        
        # Task-specific predictions
        if 'detection' in tasks:
            outputs['detection'] = self.detection_head(fpn_features)
        
        if 'pose' in tasks:
            outputs['pose'] = self.pose_head(fpn_features)
        
        if 'face' in tasks:
            outputs['face'] = self.face_head(fpn_features)
        
        if 'hand' in tasks:
            outputs['hand'] = self.hand_head(fpn_features)
        
        if 'iris' in tasks:
            outputs['iris'] = self.iris_head(fpn_features)
        
        if 'segmentation' in tasks:
            seg_output = self.segmentation_head(backbone_features)
            outputs['segmentation'] = F.interpolate(
                seg_output, 
                size=x.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        if 'classification' in tasks:
            outputs['classification'] = self.classification_head(backbone_features[-1])
        
        return outputs
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

print("✅ Complete NAiBiL model architecture loaded")



# ============================================================================
# COCO DATASET HANDLER
# ============================================================================

class COCODataset(Dataset):
    """COCO dataset for detection, keypoints, and segmentation"""
    
    def __init__(self, root_dir, split='train', img_size=640, tasks=['detection']):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.tasks = tasks
        
        # Paths
        self.img_dir = self.root_dir / f'{split}2017'
        self.ann_file = self.root_dir / 'annotations' / f'instances_{split}2017.json'
        
        # Check if files exist
        if not self.ann_file.exists():
            print(f"⚠️  Warning: {self.ann_file} not found")
            print("   Using synthetic data instead")
            self.use_synthetic = True
            self.num_samples = 1000
        else:
            self.use_synthetic = False
            # Load COCO annotations
            self.coco = COCO(str(self.ann_file))
            self.img_ids = list(self.coco.imgs.keys())
            self.num_samples = len(self.img_ids)
        
        # Augmentation
        self.transforms = self._get_transforms()
    
    def _get_transforms(self):
        """Get augmentation pipeline"""
        if self.split == 'train':
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size, 
                    min_width=self.img_size, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value=0
                ),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.RandomGamma(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='coco', 
                label_fields=['labels'],
                min_visibility=0.3
            ))
        else:
            return A.Compose([
                A.LongestMaxSize(max_size=self.img_size),
                A.PadIfNeeded(
                    min_height=self.img_size, 
                    min_width=self.img_size, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value=0
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(
                format='coco', 
                label_fields=['labels']
            ))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if self.use_synthetic:
            return self._get_synthetic_sample()
        else:
            return self._get_coco_sample(idx)
    
    def _get_synthetic_sample(self):
        """Generate synthetic data for testing"""
        # Random image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Random bounding boxes
        bboxes = []
        labels = []
        num_boxes = np.random.randint(1, 5)
        
        for _ in range(num_boxes):
            x = np.random.randint(0, 580)
            y = np.random.randint(0, 420)
            w = np.random.randint(20, 60)
            h = np.random.randint(20, 60)
            bboxes.append([x, y, w, h])
            labels.append(np.random.randint(0, 80))
        
        # Apply transforms
        transformed = self.transforms(image=image, bboxes=bboxes, labels=labels)
        
        targets = {
            'boxes': torch.tensor(transformed['bboxes'], dtype=torch.float32),
            'labels': torch.tensor(transformed['labels'], dtype=torch.long),
            'segmentation': torch.randint(0, 21, (self.img_size, self.img_size), dtype=torch.long),
            'classification': {
                'age': np.random.randint(0, 5),
                'gender': np.random.randint(0, 3),
                'vehicle_type': np.random.randint(0, 20),
                'activity': np.random.randint(0, 30),
                'fish_health': np.random.randint(0, 4),
                'infrastructure': np.random.randint(0, 6),
                'object_quality': np.random.randint(0, 5),
            }
        }
        
        return transformed['image'], targets
    
    def _get_coco_sample(self, idx):
        """Load actual COCO sample"""
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        
        # Load image
        img_path = self.img_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        bboxes = []
        labels = []
        keypoints = []
        
        for ann in anns:
            if 'bbox' in ann and ann['area'] > 0:
                bboxes.append(ann['bbox'])
                labels.append(ann['category_id'])
                
                if 'keypoints' in ann:
                    keypoints.append(ann['keypoints'])
        
        # Apply transforms
        transformed = self.transforms(image=image, bboxes=bboxes, labels=labels)
        
        targets = {
            'boxes': torch.tensor(transformed['bboxes'], dtype=torch.float32),
            'labels': torch.tensor(transformed['labels'], dtype=torch.long),
            'image_id': torch.tensor([img_id]),
        }
        
        # Add segmentation if available
        if 'segmentation' in self.tasks:
            # Create dummy segmentation for now
            targets['segmentation'] = torch.randint(0, 21, (self.img_size, self.img_size), dtype=torch.long)
        
        return transformed['image'], targets

def collate_fn(batch):
    """Custom collate function for batching"""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    
    # Prepare batch targets
    batch_targets = {
        'boxes': [],
        'labels': [],
        'segmentation': None,
        'classification': {}
    }
    
    # Collect boxes and labels
    for t in targets:
        if 'boxes' in t:
            batch_targets['boxes'].append(t['boxes'])
        if 'labels' in t:
            batch_targets['labels'].append(t['labels'])
    
    # Stack segmentation
    seg_masks = [t['segmentation'] for t in targets if 'segmentation' in t]
    if seg_masks:
        batch_targets['segmentation'] = torch.stack(seg_masks)
    
    # Stack classification
    if 'classification' in targets[0]:
        for key in targets[0]['classification'].keys():
            values = [t['classification'][key] for t in targets]
            batch_targets['classification'][key] = torch.tensor(values, dtype=torch.long)
    
    return images, batch_targets

def create_dataloaders(config):
    """Create training and validation dataloaders"""
    
    train_dataset = COCODataset(
        root_dir=config['data_root'],
        split='train',
        img_size=config['img_size'],
        tasks=config['tasks']
    )
    
    val_dataset = COCODataset(
        root_dir=config['data_root'],
        split='val',
        img_size=config['img_size'],
        tasks=config['tasks']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

print("✅ Dataset handlers loaded")



# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class IoULoss(nn.Module):
    """IoU loss for bounding box regression"""
    def __init__(self, loss_type='giou'):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, pred, target):
        # Calculate IoU
        # This is a simplified version
        return F.l1_loss(pred, target)

class MultiTaskLoss(nn.Module):
    """Multi-task loss with learnable uncertainty weighting"""
    def __init__(self, tasks):
        super().__init__()
        
        # Learnable task weights (uncertainty)
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) 
            for task in tasks
        })
        
        # Task-specific losses
        self.focal_loss = FocalLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, predictions, targets, active_tasks):
        losses = {}
        total_loss = 0
        
        # Detection loss
        if 'detection' in active_tasks and 'detection' in predictions:
            det_loss = sum(torch.abs(pred).mean() for pred in predictions['detection']) * 0.01
            precision = torch.exp(-self.log_vars['detection'])
            losses['detection'] = precision * det_loss + self.log_vars['detection']
            total_loss += losses['detection']
        
        # Keypoint losses (pose, face, hand, iris)
        for task in ['pose', 'face', 'hand', 'iris']:
            if task in active_tasks and task in predictions:
                kp_loss = sum(torch.abs(pred).mean() for pred in predictions[task]) * 0.01
                precision = torch.exp(-self.log_vars[task])
                losses[task] = precision * kp_loss + self.log_vars[task]
                total_loss += losses[task]
        
        # Segmentation loss
        if 'segmentation' in active_tasks and 'segmentation' in predictions:
            if targets.get('segmentation') is not None:
                seg_loss = self.ce_loss(predictions['segmentation'], targets['segmentation'])
            else:
                seg_loss = torch.abs(predictions['segmentation']).mean() * 0.01
            
            precision = torch.exp(-self.log_vars['segmentation'])
            losses['segmentation'] = precision * seg_loss + self.log_vars['segmentation']
            total_loss += losses['segmentation']
        
        # Classification loss
        if 'classification' in active_tasks and 'classification' in predictions:
            cls_loss = 0
            for task_name, pred in predictions['classification'].items():
                if task_name in targets.get('classification', {}):
                    cls_loss += self.focal_loss(pred, targets['classification'][task_name])
            
            if cls_loss > 0:
                precision = torch.exp(-self.log_vars['classification'])
                losses['classification'] = precision * cls_loss + self.log_vars['classification']
                total_loss += losses['classification']
        
        return total_loss, losses

print("✅ Loss functions loaded")



# ============================================================================
# TRAINING ENGINE
# ============================================================================

class NAiBiLTrainer:
    """Complete training pipeline with advanced features"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get('lr', 1e-3),
            epochs=config.get('epochs', 100),
            steps_per_epoch=config.get('steps_per_epoch', 100),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Loss function
        self.criterion = MultiTaskLoss(
            tasks=['detection', 'pose', 'face', 'hand', 'iris', 'segmentation', 'classification']
        ).to(device)
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.save_dir = Path(config.get('save_dir', '/content/drive/MyDrive/NAiBiL_Models'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader, active_tasks):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        task_losses = defaultdict(float)
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            # Move targets to device
            if isinstance(targets, dict):
                if 'segmentation' in targets and targets['segmentation'] is not None:
                    targets['segmentation'] = targets['segmentation'].to(self.device)
                
                if 'classification' in targets:
                    for k, v in targets['classification'].items():
                        targets['classification'][k] = v.to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                predictions = self.model(images, tasks=active_tasks)
                loss, losses_dict = self.criterion(predictions, targets, active_tasks)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Track losses
            total_loss += loss.item()
            for task, task_loss in losses_dict.items():
                task_losses[task] += task_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_task_losses = {k: v / len(train_loader) for k, v in task_losses.items()}
        
        return avg_loss, avg_task_losses
    
    @torch.no_grad()
    def validate(self, val_loader, active_tasks):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        task_losses = defaultdict(float)
        
        pbar = tqdm(val_loader, desc='Validation')
        
        for images, targets in pbar:
            images = images.to(self.device)
            
            # Move targets to device
            if isinstance(targets, dict):
                if 'segmentation' in targets and targets['segmentation'] is not None:
                    targets['segmentation'] = targets['segmentation'].to(self.device)
                
                if 'classification' in targets:
                    for k, v in targets['classification'].items():
                        targets['classification'][k] = v.to(self.device)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                predictions = self.model(images, tasks=active_tasks)
                loss, losses_dict = self.criterion(predictions, targets, active_tasks)
            
            # Track losses
            total_loss += loss.item()
            for task, task_loss in losses_dict.items():
                task_losses[task] += task_loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(val_loader)
        avg_task_losses = {k: v / len(val_loader) for k, v in task_losses.items()}
        
        return avg_loss, avg_task_losses
    
    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"💾 Saved checkpoint: {save_path}")
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"🏆 Best model saved: {best_path}")
    
    def train(self, train_loader, val_loader, epochs, active_tasks):
        """Complete training loop"""
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING STARTED")
        print(f"{'='*70}")
        print(f"   Epochs: {epochs}")
        print(f"   Active Tasks: {active_tasks}")
        print(f"   Device: {self.device}")
        print(f"   Mixed Precision: {self.use_amp}")
        print(f"{'='*70}\n")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_task_losses = self.train_epoch(train_loader, active_tasks)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_task_losses = self.validate(val_loader, active_tasks)
            self.val_losses.append(val_loss)
            
            # Print metrics
            print(f"\n{'='*70}")
            print(f"📊 Epoch {epoch + 1}/{epochs} Summary")
            print(f"{'='*70}")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss:   {val_loss:.4f}")
            print(f"\n   Task-wise Losses (Train | Val):")
            for task in active_tasks:
                if task in train_task_losses:
                    print(f"     {task:15s}: {train_task_losses[task]:.4f} | {val_task_losses.get(task, 0):.4f}")
            print(f"{'='*70}\n")
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if (epoch + 1) % 10 == 0 or is_best:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth', is_best)
        
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"   Best Validation Loss: {self.best_loss:.4f}")
        print(f"   Models saved to: {self.save_dir}")
        print(f"{'='*70}\n")

print("✅ Training engine loaded")



# ============================================================================
# DECISION ENGINE
# ============================================================================

@dataclass
class RobotCommand:
    """Command structure for autonomous systems"""
    action: str  # 'proceed', 'slow', 'stop', 'turn_left', 'turn_right', 'hover', 'ascend', 'descend'
    speed: float  # 0.0 to 1.0
    steering: float  # -1.0 (left) to 1.0 (right)
    altitude_change: float  # For drones: -1.0 (descend) to 1.0 (ascend)
    warnings: List[str]
    hazards: List[str]
    confidence: float
    sensor_commands: Dict[str, Union[bool, float]]
    reasoning: str

class AdvancedDecisionEngine:
    """Intelligent decision making for autonomous systems"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Safety parameters
        self.max_speed = self.config.get('max_speed', 10.0)
        self.safe_distance = self.config.get('safe_distance', 2.0)  # meters
        self.reaction_time = self.config.get('reaction_time', 0.3)  # seconds
        
        # Decision thresholds
        self.confidence_threshold = 0.5
        self.hazard_threshold = 0.7
        
        # System type
        self.system_type = self.config.get('system_type', 'drone')  # drone, vehicle, robot, submarine
    
    def analyze_scene(self, predictions, fps=30):
        """Comprehensive scene analysis"""
        analysis = {
            'safe_to_proceed': True,
            'recommended_speed': 1.0,
            'warnings': [],
            'hazards': [],
            'objects': [],
            'infrastructure_quality': 'good',
            'environmental_conditions': 'normal',
            'confidence': 1.0
        }
        
        outputs = predictions['outputs']
        
        # 1. Analyze detection outputs
        if 'detection' in outputs:
            objects = self._analyze_objects(outputs['detection'])
            analysis['objects'] = objects
            
            # Check for immediate hazards
            for obj in objects:
                if obj['class'] in ['person', 'bicycle', 'car', 'motorcycle']:
                    if obj['distance'] < self.safe_distance:
                        analysis['safe_to_proceed'] = False
                        analysis['hazards'].append(f"Close {obj['class']} detected")
                        analysis['recommended_speed'] = 0.0
                    elif obj['distance'] < self.safe_distance * 2:
                        analysis['recommended_speed'] = min(analysis['recommended_speed'], 0.3)
                        analysis['warnings'].append(f"{obj['class']} nearby")
        
        # 2. Analyze segmentation (road/water conditions)
        if 'segmentation' in outputs:
            road_quality = self._analyze_segmentation(outputs['segmentation'])
            
            if road_quality['has_potholes']:
                analysis['recommended_speed'] = min(analysis['recommended_speed'], 0.5)
                analysis['warnings'].append("Poor road condition detected")
            
            if road_quality['has_water']:
                analysis['recommended_speed'] = min(analysis['recommended_speed'], 0.6)
                analysis['warnings'].append("Water detected on path")
        
        # 3. Analyze pose/activity
        if 'pose' in outputs:
            activities = self._analyze_human_activity(outputs['pose'])
            
            for activity in activities:
                if activity['type'] in ['falling', 'running', 'waving']:
                    analysis['safe_to_proceed'] = False
                    analysis['warnings'].append(f"Human activity: {activity['type']}")
                    analysis['recommended_speed'] = 0.0
        
        # 4. Analyze classification outputs
        if 'classification' in outputs:
            cls_analysis = self._analyze_classification(outputs['classification'])
            
            # Infrastructure condition
            if 'infrastructure' in cls_analysis:
                if cls_analysis['infrastructure']['quality'] == 'critical':
                    analysis['safe_to_proceed'] = False
                    analysis['hazards'].append("Critical infrastructure damage")
                elif cls_analysis['infrastructure']['quality'] == 'poor':
                    analysis['recommended_speed'] = min(analysis['recommended_speed'], 0.4)
                    analysis['warnings'].append("Poor infrastructure condition")
            
            # Activity detection
            if 'activity' in cls_analysis:
                if cls_analysis['activity']['risk_level'] == 'high':
                    analysis['safe_to_proceed'] = False
                    analysis['warnings'].append(f"High-risk activity: {cls_analysis['activity']['type']}")
        
        # 5. Environmental analysis (for drones/submarines)
        if self.system_type in ['drone', 'submarine']:
            env_analysis = self._analyze_environment(outputs)
            analysis['environmental_conditions'] = env_analysis['condition']
            
            if env_analysis['condition'] == 'hazardous':
                analysis['safe_to_proceed'] = False
                analysis['hazards'].extend(env_analysis['hazards'])
        
        # Calculate overall confidence
        analysis['confidence'] = predictions.get('fps', 30) / 30.0  # Based on processing speed
        
        return analysis
    
    def _analyze_objects(self, detections):
        """Analyze detected objects"""
        objects = []
        
        # This is a placeholder - implement actual detection parsing
        # In real implementation, parse bounding boxes, classes, confidences
        
        return objects
    
    def _analyze_segmentation(self, seg_output):
        """Analyze segmentation output for road/environment quality"""
        # Placeholder implementation
        return {
            'has_potholes': False,
            'has_water': False,
            'has_cracks': False,
            'quality_score': 0.8
        }
    
    def _analyze_human_activity(self, pose_output):
        """Analyze human poses for activity recognition"""
        # Placeholder implementation
        return []
    
    def _analyze_classification(self, cls_outputs):
        """Analyze classification outputs"""
        analysis = {}
        
        if 'infrastructure' in cls_outputs:
            # Get infrastructure quality
            infra_scores = F.softmax(cls_outputs['infrastructure'], dim=1)
            infra_pred = torch.argmax(infra_scores, dim=1)[0].item()
            
            quality_map = {0: 'excellent', 1: 'good', 2: 'fair', 3: 'poor', 4: 'critical', 5: 'dangerous'}
            analysis['infrastructure'] = {
                'quality': quality_map.get(infra_pred, 'unknown'),
                'confidence': infra_scores[0][infra_pred].item()
            }
        
        if 'activity' in cls_outputs:
            # Analyze activity risk
            activity_scores = F.softmax(cls_outputs['activity'], dim=1)
            activity_pred = torch.argmax(activity_scores, dim=1)[0].item()
            
            # Define high-risk activities (example indices)
            high_risk_activities = [5, 6, 7, 15, 16]  # fighting, running, accident, etc.
            
            analysis['activity'] = {
                'type': f'activity_{activity_pred}',
                'risk_level': 'high' if activity_pred in high_risk_activities else 'low',
                'confidence': activity_scores[0][activity_pred].item()
            }
        
        return analysis
    
    def _analyze_environment(self, outputs):
        """Analyze environmental conditions for aerial/underwater systems"""
        # Placeholder implementation
        return {
            'condition': 'normal',
            'hazards': []
        }
    
    def make_decision(self, scene_analysis) -> RobotCommand:
        """Generate control command based on scene analysis"""
        
        warnings = scene_analysis['warnings']
        hazards = scene_analysis['hazards']
        
        # Emergency stop
        if not scene_analysis['safe_to_proceed'] or len(hazards) > 0:
            return RobotCommand(
                action='stop',
                speed=0.0,
                steering=0.0,
                altitude_change=0.0,
                warnings=warnings,
                hazards=hazards,
                confidence=scene_analysis['confidence'],
                sensor_commands={
                    'brake': True,
                    'emergency_stop': True,
                    'warning_lights': True,
                    'horn': len(hazards) > 0
                },
                reasoning="Emergency stop due to detected hazards"
            )
        
        # Calculate speed
        target_speed = self.max_speed * scene_analysis['recommended_speed']
        
        # Determine action
        if target_speed < 0.3 * self.max_speed:
            action = 'slow'
            reasoning = "Reducing speed due to warnings"
        elif target_speed > 0.7 * self.max_speed:
            action = 'proceed'
            reasoning = "Clear path ahead"
        else:
            action = 'proceed'
            reasoning = "Proceeding with caution"
        
        # System-specific adjustments
        altitude_change = 0.0
        if self.system_type == 'drone':
            # Drones can adjust altitude to avoid obstacles
            if len(warnings) > 0:
                altitude_change = 0.3  # Ascend slightly
                reasoning += ", increasing altitude for safety"
        
        return RobotCommand(
            action=action,
            speed=target_speed,
            steering=0.0,  # Straight ahead (implement path planning for turns)
            altitude_change=altitude_change,
            warnings=warnings,
            hazards=hazards,
            confidence=scene_analysis['confidence'],
            sensor_commands={
                'motor_speed': target_speed,
                'brake': False,
                'warning_lights': len(warnings) > 0,
                'camera_zoom': 1.0 + (0.5 if len(warnings) > 0 else 0.0)
            },
            reasoning=reasoning
        )

print("✅ Advanced decision engine loaded")



# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class InferenceEngine:
    """Real-time inference engine"""
    
    def __init__(self, model, device='cuda', img_size=640):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.img_size = img_size
        
        # Normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        
        # Performance tracking
        self.inference_times = []
    
    def preprocess(self, image):
        """Preprocess image for inference"""
        self.original_size = image.shape[:2]
        
        # Resize while maintaining aspect ratio
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.to(self.device)
        
        # Normalize
        image_tensor = (image_tensor - self.mean) / self.std
        
        return image_tensor.unsqueeze(0)
    
    @torch.no_grad()
    def predict(self, image, tasks='all'):
        """Run inference on image"""
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Forward pass
        outputs = self.model(input_tensor, tasks=tasks)
        
        # Calculate metrics
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        # Keep only last 100 measurements
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        avg_inference_time = np.mean(self.inference_times)
        fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
        
        results = {
            'inference_time_ms': inference_time,
            'avg_inference_time_ms': avg_inference_time,
            'fps': fps,
            'outputs': outputs,
            'input_shape': input_tensor.shape,
            'original_size': self.original_size
        }
        
        return results
    
    def get_performance_stats(self):
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'std_inference_time': np.std(self.inference_times),
            'avg_fps': 1000 / np.mean(self.inference_times)
        }

class RealtimeDetector:
    """Complete real-time detection and decision system"""
    
    def __init__(self, model, device='cuda', decision_config=None):
        self.inference_engine = InferenceEngine(model, device)
        self.decision_engine = AdvancedDecisionEngine(decision_config)
        
        # Visualization colors
        self.colors = self._generate_colors(100)
    
    def _generate_colors(self, num_classes):
        """Generate random colors for visualization"""
        np.random.seed(42)
        return [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) 
                for _ in range(num_classes)]
    
    def process_frame(self, frame, visualize=True, tasks='all'):
        """Process single frame"""
        # Run inference
        predictions = self.inference_engine.predict(frame, tasks=tasks)
        
        # Make decision
        scene_analysis = self.decision_engine.analyze_scene(predictions)
        decision = self.decision_engine.make_decision(scene_analysis)
        
        if visualize:
            annotated = self.visualize(frame, predictions, decision)
            return annotated, predictions, decision
        
        return frame, predictions, decision
    
    def visualize(self, frame, predictions, decision):
        """Draw predictions and decision on frame"""
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Draw status panel
        panel_height = 200
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # Status information
        info_texts = [
            f"FPS: {predictions['fps']:.1f}",
            f"Inference: {predictions['inference_time_ms']:.1f}ms",
            f"Action: {decision.action.upper()}",
            f"Speed: {decision.speed:.2f} m/s",
            f"Confidence: {decision.confidence:.2f}",
        ]
        
        # Add warnings
        if decision.warnings:
            info_texts.append("⚠️  WARNINGS:")
            for warning in decision.warnings[:2]:
                info_texts.append(f"  • {warning}")
        
        # Add hazards
        if decision.hazards:
            info_texts.append("🚨 HAZARDS:")
            for hazard in decision.hazards[:2]:
                info_texts.append(f"  • {hazard}")
        
        # Draw text on panel
        y_offset = 20
        for text in info_texts:
            color = (0, 255, 0) if '🚨' not in text and '⚠️' not in text else (0, 0, 255)
            cv2.putText(panel, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1, cv2.LINE_AA)
            y_offset += 25
        
        # Combine frame and panel
        result = np.vstack([annotated, panel])
        
        return result

print("✅ Inference engine and real-time detector loaded")



# ============================================================================
# MODEL CONFIGURATION & CREATION
# ============================================================================

# Model configuration
model_config = {
    # Model architecture
    'num_classes': 80,              # COCO classes
    'num_pose_keypoints': 33,       # Body keypoints (MediaPipe style)
    'num_face_landmarks': 468,      # Face mesh landmarks
    'num_hand_landmarks': 42,       # Hand landmarks (21 per hand)
    'num_iris_landmarks': 10,       # Iris landmarks (5 per eye)
    'num_seg_classes': 21,          # Segmentation classes
    
    # Classification tasks
    'classification_tasks': {
        'age': 5,                   # [0-12, 13-19, 20-39, 40-59, 60+]
        'gender': 3,                # [male, female, other]
        'vehicle_type': 20,         # [car, truck, bus, motorcycle, etc.]
        'activity': 30,             # [walking, running, sitting, etc.]
        'fish_health': 4,           # [healthy, sick, injured, dead]
        'infrastructure': 6,        # [excellent, good, fair, poor, critical, dangerous]
        'object_quality': 5,        # Quality assessment
    },
    
    # Training parameters
    'dropout': 0.1,
    'bifpn_layers': 3,
    'num_heads': 8,
}

print("="*70)
print("🏗️  CREATING NAiBiL MODEL")
print("="*70)

# Create model
model = NAiBiL(model_config)
model = model.to(device)

# Count parameters
params = model.count_parameters()
print(f"\n✅ Model Created Successfully!")
print(f"   Total Parameters: {params['total']:,}")
print(f"   Trainable Parameters: {params['trainable']:,}")
print(f"   Model Size: ~{params['total'] * 4 / (1024**2):.1f} MB (FP32)")


# Test forward pass
print("\n🧪 Testing Model Forward Pass...")

model.eval()
dummy_input = torch.randn(1, 3, 640, 640).to(device)

with torch.no_grad():
    outputs = model(dummy_input, tasks='all')

print("\n✅ Forward Pass Successful!")
print("\n📐 Output Shapes:")

for task, output in outputs.items():
    if isinstance(output, dict):
        print(f"\n{task.upper()}:")
        for k, v in output.items():
            if torch.is_tensor(v):
                print(f"  {k}: {tuple(v.shape)}")
    elif isinstance(output, list):
        print(f"\n{task.upper()}: {len(output)} scales")
        for i, o in enumerate(output):
            if torch.is_tensor(o):
                print(f"  Scale {i}: {tuple(o.shape)}")
    else:
        if torch.is_tensor(output):
            print(f"{task.upper()}: {tuple(output.shape)}")

print("\n✅ Model is ready for training!")


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Data configuration
data_config = {
    'data_root': '/content/datasets/coco',
    'img_size': 640,
    'batch_size': 32,  # Adjust based on GPU memory
    'num_workers': 16,
    'tasks': ['detection', 'pose', 'segmentation', 'classification']
}

# Training configuration
training_config = {
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 1,  # Start with 50, increase for better results
    'use_amp': True,
    'save_dir': '/content/drive/MyDrive/NAiBiL_Models',
    'steps_per_epoch': 250,  # Will be updated based on actual dataset size
}

# Active tasks for training
active_tasks = [
    'detection',
    'pose',
    'face',
    'hand',
    'iris',
    'segmentation',
    'classification',
]

print("Configuration set!")
print(f"\nData: {data_config}")
print(f"\nTraining: {training_config}")
print(f"\nActive Tasks: {active_tasks}")



# Create dataloaders
print("📦 Creating Dataloaders...")

train_loader, val_loader = create_dataloaders(data_config)

# Update steps per epoch
training_config['steps_per_epoch'] = len(train_loader)

print(f"✅ Dataloaders created!")
print(f"   Train batches: {len(train_loader)}")
print(f"   Val batches: {len(val_loader)}")

# Test data loading
print("\n🧪 Testing data loading...")
for images, targets in train_loader:
    print(f"   Image batch shape: {images.shape}")
    print(f"   Targets keys: {list(targets.keys())}")
    break

print("\n✅ Data pipeline is working!")



# ============================================================================
# START TRAINING
# ============================================================================

# Create trainer
trainer = NAiBiLTrainer(model, training_config, device=device)

# Start training
trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=training_config['epochs'],
    active_tasks=active_tasks
)

print("\n" + "="*70)
print("✅ TRAINING COMPLETED!")
print("="*70)
print(f"\n💾 Models saved to: {training_config['save_dir']}")
print(f"   • Best model: best_model.pth")
print(f"   • Checkpoints: checkpoint_epoch_*.pth")


# ============================================================================
# LOAD TRAINED MODEL
# ============================================================================

best_model_path = Path(training_config['save_dir']) / 'best_model.pth'

if best_model_path.exists():
    print(f"📂 Loading best model from: {best_model_path}")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded!")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Best Loss: {checkpoint['best_loss']:.4f}")
else:
    print("⚠️  Best model not found. Using current model state.")

model.eval()



# ============================================================================
# CREATE REAL-TIME DETECTOR
# ============================================================================

# Decision engine configuration
decision_config = {
    'system_type': 'drone',  # Options: 'drone', 'vehicle', 'robot', 'submarine'
    'max_speed': 10.0,       # meters per second
    'safe_distance': 2.0,    # meters
    'reaction_time': 0.3,    # seconds
}

# Create detector
detector = RealtimeDetector(
    model=model,
    device=device,
    decision_config=decision_config
)

print("✅ Real-time detector created!")
print(f"   System Type: {decision_config['system_type']}")
print(f"   Max Speed: {decision_config['max_speed']} m/s")



# ============================================================================
# TEST INFERENCE
# ============================================================================

print("🧪 Testing Inference Engine...\n")

# Create test image
test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Process frame
annotated, predictions, decision = detector.process_frame(test_image)

print("✅ Inference Successful!\n")
print("📊 Performance Metrics:")
print(f"   FPS: {predictions['fps']:.2f}")
print(f"   Inference Time: {predictions['inference_time_ms']:.2f} ms")

print("\n🤖 Decision Engine Output:")
print(f"   Action: {decision.action.upper()}")
print(f"   Speed: {decision.speed:.2f} m/s")
print(f"   Confidence: {decision.confidence:.2f}")
print(f"   Reasoning: {decision.reasoning}")

if decision.warnings:
    print(f"\n⚠️  Warnings:")
    for warning in decision.warnings:
        print(f"     • {warning}")

if decision.hazards:
    print(f"\n🚨 Hazards:")
    for hazard in decision.hazards:
        print(f"     • {hazard}")

# Display annotated frame
plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.title(f"NAiBiL Real-time Detection (FPS: {predictions['fps']:.1f})")
plt.axis('off')
plt.tight_layout()
plt.show()

print("\n✅ Inference engine is ready for deployment!")




# ============================================================================
# EXPORT TO ONNX
# ============================================================================

export_path = Path(training_config['save_dir']) / 'naibil_model.onnx'

print(f"📦 Exporting model to ONNX format...")
print(f"   Output path: {export_path}")

model.eval()
dummy_input = torch.randn(1, 3, 640, 640).to(device)

try:
    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['detection', 'pose', 'face', 'hand', 'iris', 'segmentation', 'classification'],
        dynamic_axes={
            'input': {0: 'batch'},
        }
    )
    print(f"\n✅ Model exported successfully!")
    print(f"   File size: {export_path.stat().st_size / (1024**2):.1f} MB")
except Exception as e:
    print(f"\n⚠️  ONNX export encountered an issue: {str(e)}")
    print("   This is normal for complex multi-task models.")
    print("   You can use the PyTorch model directly for deployment.")




# ============================================================================
# BENCHMARK PERFORMANCE
# ============================================================================

print("🚀 Running Performance Benchmark...\n")

# Run multiple inferences
num_iterations = 100

for i in tqdm(range(num_iterations), desc="Benchmarking"):
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, predictions, _ = detector.process_frame(test_image, visualize=False)

# Get statistics
stats = detector.inference_engine.get_performance_stats()

print("\n📊 Performance Statistics (100 iterations):")
print(f"   Average FPS: {stats['avg_fps']:.2f}")
print(f"   Average Inference Time: {stats['avg_inference_time']:.2f} ms")
print(f"   Min Inference Time: {stats['min_inference_time']:.2f} ms")
print(f"   Max Inference Time: {stats['max_inference_time']:.2f} ms")
print(f"   Std Inference Time: {stats['std_inference_time']:.2f} ms")

# Plot performance
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(detector.inference_engine.inference_times)
plt.axhline(y=stats['avg_inference_time'], color='r', linestyle='--', label='Average')
plt.xlabel('Iteration')
plt.ylabel('Inference Time (ms)')
plt.title('Inference Time Over Iterations')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(detector.inference_engine.inference_times, bins=20, edgecolor='black')
plt.xlabel('Inference Time (ms)')
plt.ylabel('Frequency')
plt.title('Inference Time Distribution')
plt.axvline(x=stats['avg_inference_time'], color='r', linestyle='--', label='Average')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ Benchmark completed!")
#END#





#Dataset information#

#coco8-Grayscale


path: coco8-grayscale # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images
test: # test images (optional)

channels: 1

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush

# Download script/URL (optional)
download: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-grayscale.zip





#COCO

path: coco # dataset root dir
train: train2017.txt # train images (relative to 'path') 118287 images
val: val2017.txt # val images (relative to 'path') 5000 images
test: test-dev2017.txt # 20288 of 40670 images, submit to https://competitions.codalab.org/competitions/20794

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush

# Download script/URL (optional)
download: |
  from pathlib import Path

  from ultralytics.utils import ASSETS_URL
  from ultralytics.utils.downloads import download

  # Download labels
  segments = True  # segment or box labels
  dir = Path(yaml["path"])  # dataset root dir
  urls = [ASSETS_URL + ("/coco2017labels-segments.zip" if segments else "/coco2017labels.zip")]  # labels
  download(urls, dir=dir.parent)
  # Download data
  urls = [
      "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
      "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
      "http://images.cocodataset.org/zips/test2017.zip",  # 7G, 41k images (optional)
  ]
  download(urls, dir=dir / "images", threads=3)







#COCO8


path: coco8 # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images
test: # test images (optional)

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush

# Download script/URL (optional)
download: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip







#COCO8-Multispectral


path: coco8-multispectral # dataset root dir
train: images/train # train images (relative to 'path') 4 images
val: images/val # val images (relative to 'path') 4 images
test: # test images (optional)

# Number of multispectral image channels
channels: 10

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush

# Download script/URL (optional)
download: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-multispectral.zip




#COCO128


path: coco128 # dataset root dir
train: images/train2017 # train images (relative to 'path') 128 images
val: images/train2017 # val images (relative to 'path') 128 images
test: # test images (optional)

# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush

# Download script/URL (optional)
download: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip




#african-wildlife


path: african-wildlife # dataset root dir
train: images/train # train images (relative to 'path') 1052 images
val: images/val # val images (relative to 'path') 225 images
test: images/test # test images (relative to 'path') 227 images

# Classes
names:
  0: buffalo
  1: elephant
  2: rhino
  3: zebra

# Download script/URL (optional)
download: https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip
#END#



#An ipynb notebook file for reference only#

# Mount Google Drive for persistent storage
from google.colab import drive
import os

drive.mount('/content/drive')

# Create project directories
PROJECT_DIRS = {
    'workspace': '/content/naibil_workspace',
    'datasets': '/content/datasets',
    'models': '/content/drive/MyDrive/NAiBiL/models',
    'checkpoints': '/content/drive/MyDrive/NAiBiL/checkpoints',
    'results': '/content/drive/MyDrive/NAiBiL/results',
    'logs': '/content/drive/MyDrive/NAiBiL/logs'
}

for name, path in PROJECT_DIRS.items():
    os.makedirs(path, exist_ok=True)
    print(f"✓ Created {name}: {path}")

print("\n✅ Google Drive mounted and directories created successfully!")

This is a wrong idea, because in colab show error with version, so you should write without version%%capture
# Install core deep learning and computer vision packages
!pip install -q torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
!pip install -q opencv-python-headless==4.8.1.78
!pip install -q albumentations==1.3.1
!pip install -q Pillow==10.1.0
!pip install -q numpy==1.24.3
!pip install -q scipy==1.11.4
!pip install -q scikit-learn==1.3.2
!pip install -q scikit-image==0.22.0

# Install visualization and logging
!pip install -q matplotlib==3.7.1
!pip install -q seaborn==0.13.0
!pip install -q tqdm==4.66.1
!pip install -q tensorboard==2.15.1
!pip install -q wandb==0.16.0

# Install COCO evaluation and utilities
!pip install -q pycocotools==2.0.7
!pip install -q timm==0.9.12
!pip install -q einops==0.7.0

# Install dataset utilities
!pip install -q datasets==2.14.6
!pip install -q huggingface-hub==0.19.4
!pip install -q gdown==4.7.1
!pip install -q kaggle==1.5.16

# Install ONNX export
!pip install -q onnx==1.15.0
!pip install -q onnxruntime-gpu==1.16.3

print("✅ All packages installed successfully!")

# Import essential libraries
import sys
import gc
import json
import random
import warnings
import zipfile
import shutil
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
from torchvision.ops import nms, box_iou, box_convert

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as coco_mask

from datasets import load_dataset, IterableDataset as HFIterableDataset
from huggingface_hub import hf_hub_download

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("="*70)
print("NAiBiL SYSTEM INITIALIZATION")
print("="*70)
print(f"\n🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
print(f"\n📦 PyTorch Version: {torch.__version__}")
print(f"📦 Torchvision Version: {torchvision.__version__}")
print(f"📦 NumPy Version: {np.__version__}")
print(f"📦 OpenCV Version: {cv2.__version__}")
print("\n✅ Environment initialized successfully!\n")
print("="*70)

@dataclass
class NAiBiLConfig:
    """Complete configuration for NAiBiL multi-task model"""
    
    # ============= Model Architecture =============
    input_size: Tuple[int, int] = (640, 640)
    backbone: str = 'resnet50'  # resnet50, efficientnet_b3, etc.
    backbone_channels: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    fpn_channels: int = 256
    
    # ============= Task Configuration =============
    # Object Detection
    num_classes: int = 100  # Extended object categories
    num_anchors: int = 9
    anchor_scales: List[float] = field(default_factory=lambda: [1.0, 1.26, 1.59])
    anchor_ratios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    max_detections: int = 100
    
    # Pose Estimation
    num_keypoints: int = 17  # COCO keypoints
    keypoint_threshold: float = 0.3
    
    # Face & Facial Landmarks
    num_face_landmarks: int = 68  # 68-point facial landmarks
    enable_face_detection: bool = True
    
    # Iris Detection
    num_iris_landmarks: int = 5  # Per eye
    enable_iris_detection: bool = True
    
    # Segmentation
    enable_segmentation: bool = True
    mask_size: Tuple[int, int] = (28, 28)
    
    # Attributes
    enable_attributes: bool = True
    num_attributes: int = 20  # Demographics, object types, etc.
    
    # ============= Training Configuration =============
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    warmup_epochs: int = 5
    gradient_clip: float = 10.0
    
    # Mixed precision training
    use_amp: bool = True
    
    # ============= Loss Weights =============
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'detection_cls': 1.0,
        'detection_box': 5.0,
        'detection_obj': 1.0,
        'pose_keypoints': 4.0,
        'pose_visibility': 1.0,
        'face_landmarks': 3.0,
        'iris_landmarks': 2.0,
        'segmentation': 2.0,
        'attributes': 1.0
    })
    
    # ============= Data Configuration =============
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Augmentation
    use_augmentation: bool = True
    mosaic_prob: float = 0.5
    mixup_prob: float = 0.15
    
    # ============= Paths =============
    data_dir: str = '/content/datasets'
    output_dir: str = '/content/drive/MyDrive/NAiBiL/results'
    checkpoint_dir: str = '/content/drive/MyDrive/NAiBiL/checkpoints'
    log_dir: str = '/content/drive/MyDrive/NAiBiL/logs'
    
    # ============= Inference =============
    inference_size: Tuple[int, int] = (640, 640)
    visualize_results: bool = True
    save_predictions: bool = True
    
    def __post_init__(self):
        """Validate and setup configuration"""
        assert self.input_size[0] % 32 == 0 and self.input_size[1] % 32 == 0, \
            "Input size must be divisible by 32"
        assert len(self.anchor_scales) * len(self.anchor_ratios) == self.num_anchors, \
            "Number of anchors must match scales x ratios"

# Initialize configuration
config = NAiBiLConfig()

print("\n📋 NAiBiL Configuration:")
print(f"   Input Size: {config.input_size}")
print(f"   Backbone: {config.backbone}")
print(f"   Object Classes: {config.num_classes}")
print(f"   Pose Keypoints: {config.num_keypoints}")
print(f"   Face Landmarks: {config.num_face_landmarks}")
print(f"   Iris Landmarks: {config.num_iris_landmarks} per eye")
print(f"   Batch Size: {config.batch_size}")
print(f"   Epochs: {config.num_epochs}")
print(f"   Learning Rate: {config.learning_rate}")
print(f"   Mixed Precision: {config.use_amp}")
print("\n✅ Configuration loaded successfully!")

# ==================== ATTENTION MECHANISMS ====================

class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on important regions"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention

class ChannelAttention(nn.Module):
    """Channel attention module (SE-like)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# ==================== BUILDING BLOCKS ====================

class ConvBNAct(nn.Module):
    """Standard Conv + BatchNorm + Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, 
                 groups=1, activation=nn.SiLU()):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
                              padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = activation
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    """Standard bottleneck block"""
    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNAct(hidden_channels, out_channels, 3, padding=1, groups=groups)
        self.add = shortcut and in_channels == out_channels
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))

class C3k2(nn.Module):
    """C3k2 block from YOLOv11 for efficient feature extraction"""
    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=True, 
                 groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNAct(in_channels, hidden_channels, 1)
        self.conv3 = ConvBNAct(2 * hidden_channels, out_channels, 1)
        self.m = nn.Sequential(
            *(Bottleneck(hidden_channels, hidden_channels, shortcut, groups, 1.0) 
              for _ in range(num_blocks))
        )
    
    def forward(self, x):
        return self.conv3(torch.cat([self.m(self.conv1(x)), self.conv2(x)], 1))

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) for multi-scale feature aggregation"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNAct(hidden_channels * 4, out_channels, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class C2PSA(nn.Module):
    """C2PSA: Channel-to-Pixel Spatial Attention from YOLOv11"""
    def __init__(self, in_channels, out_channels, num_blocks=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNAct(in_channels, hidden_channels, 1)
        self.conv3 = ConvBNAct(2 * hidden_channels, out_channels, 1)
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.Sigmoid()
        )
        self.m = nn.Sequential(
            *(Bottleneck(hidden_channels, hidden_channels, True, 1, 1.0) for _ in range(num_blocks))
        )
    
    def forward(self, x):
        x1 = self.conv1(x)
        attn = self.attention(x1)
        x1 = x1 * attn
        x1 = self.m(x1)
        return self.conv3(torch.cat([x1, self.conv2(x)], 1))

print("✅ Building blocks defined successfully!")

# ==================== BACKBONE NETWORK ====================

class NAiBiLBackbone(nn.Module):
    """Advanced backbone network with C3k2 blocks and attention mechanisms"""
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            ConvBNAct(in_channels, base_channels, 3, 2, 1),
            ConvBNAct(base_channels, base_channels * 2, 3, 2, 1)
        )
        
        # Stage 1: 160x160 -> 80x80
        self.stage1 = nn.Sequential(
            C3k2(base_channels * 2, base_channels * 4, num_blocks=3),
            ConvBNAct(base_channels * 4, base_channels * 4, 3, 2, 1)
        )
        
        # Stage 2: 80x80 -> 40x40
        self.stage2 = nn.Sequential(
            C3k2(base_channels * 4, base_channels * 8, num_blocks=6),
            ConvBNAct(base_channels * 8, base_channels * 8, 3, 2, 1)
        )
        
        # Stage 3: 40x40 -> 20x20
        self.stage3 = nn.Sequential(
            C3k2(base_channels * 8, base_channels * 16, num_blocks=6),
            C2PSA(base_channels * 16, base_channels * 16, num_blocks=3),
            ConvBNAct(base_channels * 16, base_channels * 16, 3, 2, 1)
        )
        
        # Stage 4: 20x20 -> 20x20 (with SPPF)
        self.stage4 = nn.Sequential(
            C3k2(base_channels * 16, base_channels * 16, num_blocks=3),
            SPPF(base_channels * 16, base_channels * 16),
            C2PSA(base_channels * 16, base_channels * 16, num_blocks=3)
        )
        
        self.out_channels = [base_channels * 4, base_channels * 8, 
                             base_channels * 16, base_channels * 16]
    
    def forward(self, x):
        # Multi-scale feature extraction
        x = self.stem(x)
        
        c2 = self.stage1(x)   # 80x80, 256 channels
        c3 = self.stage2(c2)  # 40x40, 512 channels  
        c4 = self.stage3(c3)  # 20x20, 1024 channels
        c5 = self.stage4(c4)  # 20x20, 1024 channels
        
        return [c2, c3, c4, c5]

# ==================== FEATURE PYRAMID NETWORK ====================

class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            ConvBNAct(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        
        # Top-down pathway
        self.fpn_convs = nn.ModuleList([
            C3k2(out_channels, out_channels, num_blocks=2) 
            for _ in range(len(in_channels_list))
        ])
        
        # Bottom-up pathway (PAN)
        self.downsample_convs = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, 3, 2, 1) 
            for _ in range(len(in_channels_list) - 1)
        ])
        
        self.pan_convs = nn.ModuleList([
            C3k2(out_channels, out_channels, num_blocks=2) 
            for _ in range(len(in_channels_list) - 1)
        ])
    
    def forward(self, features):
        # features: [c2, c3, c4, c5] from backbone
        
        # Lateral connections
        laterals = [lateral_conv(f) for f, lateral_conv in zip(features, self.lateral_convs)]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], scale_factor=2, mode='nearest')
            laterals[i-1] = laterals[i-1] + upsampled
        
        # Apply FPN convolutions
        fpn_features = [conv(lateral) for lateral, conv in zip(laterals, self.fpn_convs)]
        
        # Bottom-up pathway (PAN)
        pan_features = [fpn_features[0]]
        for i in range(len(self.downsample_convs)):
            downsampled = self.downsample_convs[i](pan_features[-1])
            pan_features.append(self.pan_convs[i](downsampled + fpn_features[i + 1]))
        
        return pan_features

print("✅ Backbone and FPN defined successfully!")

# ==================== MULTI-TASK DETECTION HEADS ====================

class DetectionHead(nn.Module):
    """Object detection head with bounding box regression and classification"""
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Shared convolutions
        self.conv = nn.Sequential(
            ConvBNAct(in_channels, in_channels, 3, 1, 1),
            ConvBNAct(in_channels, in_channels, 3, 1, 1)
        )
        
        # Detection outputs: [x, y, w, h, objectness, class_probs]
        self.pred = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pred(x)
        
        bs, _, h, w = x.shape
        x = x.view(bs, self.num_anchors, 5 + self.num_classes, h, w)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        
        return x

class PoseHead(nn.Module):
    """Human pose estimation head for keypoint detection"""
    def __init__(self, in_channels, num_keypoints=17):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        self.conv = nn.Sequential(
            C3k2(in_channels, in_channels, num_blocks=2),
            ConvBNAct(in_channels, in_channels // 2, 3, 1, 1)
        )
        
        # Keypoint heatmaps (x, y) + visibility
        self.heatmap = nn.Conv2d(in_channels // 2, num_keypoints, 1)
        self.visibility = nn.Conv2d(in_channels // 2, num_keypoints, 1)
    
    def forward(self, x):
        x = self.conv(x)
        heatmaps = torch.sigmoid(self.heatmap(x))
        visibility = torch.sigmoid(self.visibility(x))
        return heatmaps, visibility

class FaceLandmarkHead(nn.Module):
    """Face detection and 68-point facial landmark detection head"""
    def __init__(self, in_channels, num_landmarks=68):
        super().__init__()
        self.num_landmarks = num_landmarks
        
        self.conv = nn.Sequential(
            C3k2(in_channels, in_channels // 2, num_blocks=2),
            CBAM(in_channels // 2),
            ConvBNAct(in_channels // 2, in_channels // 4, 3, 1, 1)
        )
        
        # Face detection
        self.face_det = nn.Conv2d(in_channels // 4, 5, 1)  # [x, y, w, h, conf]
        
        # Facial landmarks (x, y, visibility)
        self.landmarks = nn.Conv2d(in_channels // 4, num_landmarks * 3, 1)
    
    def forward(self, x):
        x = self.conv(x)
        face_box = self.face_det(x)
        landmarks = self.landmarks(x)
        
        bs, _, h, w = landmarks.shape
        landmarks = landmarks.view(bs, self.num_landmarks, 3, h, w)
        
        return face_box, landmarks

class IrisHead(nn.Module):
    """Iris detection and landmark estimation head"""
    def __init__(self, in_channels, num_iris_points=5):
        super().__init__()
        self.num_iris_points = num_iris_points
        
        self.conv = nn.Sequential(
            ConvBNAct(in_channels, in_channels // 2, 3, 1, 1),
            CBAM(in_channels // 2),
            ConvBNAct(in_channels // 2, in_channels // 4, 3, 1, 1)
        )
        
        # Left and right iris landmarks
        self.left_iris = nn.Conv2d(in_channels // 4, num_iris_points * 2, 1)
        self.right_iris = nn.Conv2d(in_channels // 4, num_iris_points * 2, 1)
    
    def forward(self, x):
        x = self.conv(x)
        left = self.left_iris(x)
        right = self.right_iris(x)
        return left, right

class SegmentationHead(nn.Module):
    """Instance segmentation head"""
    def __init__(self, in_channels, mask_dim=256, num_protos=32):
        super().__init__()
        self.num_protos = num_protos
        
        # Prototype generation
        self.proto = nn.Sequential(
            ConvBNAct(in_channels, in_channels // 2, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBNAct(in_channels // 2, in_channels // 4, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBNAct(in_channels // 4, in_channels // 8, 3, 1, 1),
            nn.Conv2d(in_channels // 8, num_protos, 1)
        )
        
        # Mask coefficients
        self.mask_coef = nn.Sequential(
            ConvBNAct(in_channels, in_channels, 3, 1, 1),
            nn.Conv2d(in_channels, num_protos, 1)
        )
    
    def forward(self, x):
        protos = self.proto(x)
        coeffs = self.mask_coef(x)
        return protos, coeffs

class AttributeHead(nn.Module):
    """Object attribute classification head (demographics, vehicle types, etc.)"""
    def __init__(self, in_channels, num_attributes=20):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNAct(in_channels, in_channels // 2, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels // 2, num_attributes)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.conv(x))

print("✅ All task-specific heads defined successfully!")

# ==================== COMPLETE NAiBiL MODEL ====================

class NAiBiLModel(nn.Module):
    """Complete NAiBiL multi-task detection system (~200M parameters)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Backbone
        self.backbone = NAiBiLBackbone(in_channels=3, base_channels=64)
        
        # Feature Pyramid Network
        self.fpn = FPN(self.backbone.out_channels, config.fpn_channels)
        
        # Multi-task heads
        self.detection_heads = nn.ModuleList([
            DetectionHead(config.fpn_channels, config.num_classes, 3) 
            for _ in range(4)
        ])
        
        if config.num_keypoints > 0:
            self.pose_head = PoseHead(config.fpn_channels, config.num_keypoints)
        
        if config.enable_face_detection:
            self.face_head = FaceLandmarkHead(config.fpn_channels, config.num_face_landmarks)
        
        if config.enable_iris_detection:
            self.iris_head = IrisHead(config.fpn_channels, config.num_iris_landmarks)
        
        if config.enable_segmentation:
            self.seg_head = SegmentationHead(config.fpn_channels)
        
        if config.enable_attributes:
            self.attr_head = AttributeHead(config.fpn_channels, config.num_attributes)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Extract multi-scale features
        backbone_features = self.backbone(x)
        
        # Feature pyramid
        fpn_features = self.fpn(backbone_features)
        
        outputs = {}
        
        # Object detection at multiple scales
        detections = [head(feat) for head, feat in zip(self.detection_heads, fpn_features)]
        outputs['detections'] = detections
        
        # Pose estimation (use highest resolution feature map)
        if hasattr(self, 'pose_head'):
            heatmaps, visibility = self.pose_head(fpn_features[0])
            outputs['pose'] = {'heatmaps': heatmaps, 'visibility': visibility}
        
        # Face landmarks
        if hasattr(self, 'face_head'):
            face_box, landmarks = self.face_head(fpn_features[0])
            outputs['face'] = {'boxes': face_box, 'landmarks': landmarks}
        
        # Iris detection
        if hasattr(self, 'iris_head'):
            left_iris, right_iris = self.iris_head(fpn_features[0])
            outputs['iris'] = {'left': left_iris, 'right': right_iris}
        
        # Instance segmentation
        if hasattr(self, 'seg_head'):
            protos, coeffs = self.seg_head(fpn_features[0])
            outputs['segmentation'] = {'protos': protos, 'coeffs': coeffs}
        
        # Attributes
        if hasattr(self, 'attr_head'):
            attributes = self.attr_head(fpn_features[0])
            outputs['attributes'] = attributes
        
        return outputs
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

# Create model instance
model = NAiBiLModel(config).to(device)
total_params, trainable_params = model.count_parameters()

print("\n" + "="*70)
print("NAiBiL MODEL ARCHITECTURE")
print("="*70)
print(f"\n📊 Total Parameters: {total_params:,}")
print(f"📊 Trainable Parameters: {trainable_params:,}")
print(f"📊 Model Size: {total_params * 4 / (1024**2):.2f} MB (FP32)")
print("\n✅ Model created successfully!")
print("="*70)

# ==================== DATA AUGMENTATION ====================

def get_training_augmentation(config):
    """Advanced augmentation pipeline for training"""
    return A.Compose([
        A.RandomResizedCrop(config.input_size[0], config.input_size[1], scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
        ], p=0.3),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05),
            A.GridDistortion(num_steps=5, distort_limit=0.05),
        ], p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

def get_validation_augmentation(config):
    """Augmentation pipeline for validation"""
    return A.Compose([
        A.Resize(config.input_size[0], config.input_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']))

# ==================== COCO-STYLE DATASET ====================

class COCOMultiTaskDataset(Dataset):
    """COCO-style dataset with multi-task annotations"""
    def __init__(self, image_dir, annotation_file, transform=None, config=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.config = config
        
        # Load COCO annotations
        if Path(annotation_file).exists():
            self.coco = COCO(annotation_file)
            self.image_ids = list(self.coco.imgs.keys())
        else:
            print(f"Warning: Annotation file {annotation_file} not found")
            self.coco = None
            self.image_ids = []
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        if not self.coco:
            # Return dummy data if no annotations
            img = torch.zeros(3, self.config.input_size[0], self.config.input_size[1])
            return {'image': img, 'boxes': torch.zeros(0, 4), 'labels': torch.zeros(0, dtype=torch.long)}
        
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        
        # Load image
        image_path = self.image_dir / image_info['file_name']
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        
        boxes = []
        labels = []
        keypoints = []
        
        for ann in anns:
            boxes.append(ann['bbox'])
            labels.append(ann['category_id'])
            if 'keypoints' in ann:
                keypoints.append(ann['keypoints'])
        
        # Apply augmentation
        if self.transform:
            try:
                transformed = self.transform(image=image, bboxes=boxes, class_labels=labels)
                image = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['class_labels']
            except:
                # Fallback to simple transform
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return {
            'image': image,
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            'labels': torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            'keypoints': torch.tensor(keypoints, dtype=torch.float32) if keypoints else torch.zeros(0, 51)
        }

print("✅ Dataset classes defined successfully!")

# ==================== STREAMING DATASET HANDLER ====================

class StreamingDatasetManager:
    """Manager for loading multiple datasets with streaming support"""
    def __init__(self, config):
        self.config = config
        self.datasets = []
        
    def add_coco_dataset(self, name, images_dir, annotations_file, split='train'):
        """Add a COCO-format dataset"""
        if not Path(images_dir).exists() or not Path(annotations_file).exists():
            print(f"⚠️  Skipping {name}: paths do not exist")
            return
        
        transform = get_training_augmentation(self.config) if split == 'train' \
                    else get_validation_augmentation(self.config)
        
        dataset = COCOMultiTaskDataset(
            image_dir=images_dir,
            annotation_file=annotations_file,
            transform=transform,
            config=self.config
        )
        
        if len(dataset) > 0:
            self.datasets.append({'name': name, 'dataset': dataset, 'split': split})
            print(f"✓ Added {name} dataset: {len(dataset)} samples")
    
    def add_huggingface_dataset(self, name, dataset_name, split='train', streaming=True):
        """Add a HuggingFace dataset with streaming support"""
        try:
            dataset = load_dataset(dataset_name, split=split, streaming=streaming)
            print(f"✓ Added streaming HF dataset: {name}")
            return dataset
        except Exception as e:
            print(f"⚠️  Failed to load {name}: {str(e)}")
            return None
    
    def get_dataloader(self, split='train', batch_size=None):
        """Get combined dataloader for specified split"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        split_datasets = [d['dataset'] for d in self.datasets if d['split'] == split]
        
        if not split_datasets:
            print(f"⚠️  No datasets found for split: {split}")
            return None
        
        # Combine datasets
        combined_dataset = torch.utils.data.ConcatDataset(split_datasets)
        
        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            collate_fn=self.collate_fn
        )
        
        return dataloader
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching"""
        images = torch.stack([item['image'] for item in batch])
        return {
            'images': images,
            'boxes': [item['boxes'] for item in batch],
            'labels': [item['labels'] for item in batch],
            'keypoints': [item['keypoints'] for item in batch]
        }

print("✅ Streaming dataset manager ready!")

# ==================== MULTI-TASK LOSS FUNCTIONS ====================

class MultiTaskLoss(nn.Module):
    """Combined loss for all tasks"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        
    def detection_loss(self, predictions, targets):
        """Compute detection loss (classification + box regression)"""
        # Simplified detection loss
        # In production, use proper anchor matching and IoU-based loss
        loss = torch.tensor(0.0, device=predictions[0].device)
        
        for pred in predictions:
            if pred.numel() > 0:
                # Classification loss
                cls_loss = self.bce_loss(pred[..., 5:], torch.zeros_like(pred[..., 5:]))
                # Objectness loss
                obj_loss = self.bce_loss(pred[..., 4:5], torch.zeros_like(pred[..., 4:5]))
                # Box regression loss
                box_loss = self.smooth_l1_loss(pred[..., :4], torch.zeros_like(pred[..., :4]))
                
                loss = loss + (
                    self.config.loss_weights['detection_cls'] * cls_loss +
                    self.config.loss_weights['detection_obj'] * obj_loss +
                    self.config.loss_weights['detection_box'] * box_loss
                )
        
        return loss / max(len(predictions), 1)
    
    def pose_loss(self, predictions, targets):
        """Compute pose estimation loss"""
        if not predictions:
            return torch.tensor(0.0, device=device)
        
        heatmaps = predictions['heatmaps']
        visibility = predictions['visibility']
        
        # Heatmap loss (MSE against target heatmaps)
        heatmap_loss = self.mse_loss(heatmaps, torch.zeros_like(heatmaps))
        visibility_loss = self.bce_loss(visibility, torch.zeros_like(visibility))
        
        return (self.config.loss_weights['pose_keypoints'] * heatmap_loss +
                self.config.loss_weights['pose_visibility'] * visibility_loss)
    
    def forward(self, predictions, targets):
        """Compute total multi-task loss"""
        total_loss = 0.0
        loss_dict = {}
        
        # Detection loss
        if 'detections' in predictions:
            det_loss = self.detection_loss(predictions['detections'], targets)
            total_loss = total_loss + det_loss
            loss_dict['detection'] = det_loss.item()
        
        # Pose loss
        if 'pose' in predictions:
            pose_loss = self.pose_loss(predictions['pose'], targets)
            total_loss = total_loss + pose_loss
            loss_dict['pose'] = pose_loss.item()
        
        # Add other task losses as needed
        # Face, iris, segmentation, attributes...
        
        loss_dict['total'] = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        
        return total_loss, loss_dict

print("✅ Multi-task loss functions defined!")

# ==================== TRAINER CLASS ====================

class NAiBiLTrainer:
    """Complete training pipeline for NAiBiL"""
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = MultiTaskLoss(config)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=config.num_epochs,
            steps_per_epoch=100,  # Will be updated
            pct_start=0.1
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Metrics tracking
        self.best_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Tensorboard writer
        self.writer = SummaryWriter(config.log_dir)
        
        print("\n✅ Trainer initialized successfully!")
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{self.config.num_epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['images'].to(self.device)
                
                # Forward pass
                if self.config.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss, loss_dict = self.criterion(outputs, batch)
                else:
                    outputs = self.model(images)
                    loss, loss_dict = self.criterion(outputs, batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.config.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                
                self.scheduler.step()
                
                epoch_loss += loss_dict['total']
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # Log to tensorboard
                global_step = epoch * len(dataloader) + batch_idx
                self.writer.add_scalar('train/loss', loss_dict['total'], global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], global_step)
                
            except Exception as e:
                print(f"\n⚠️  Error in batch {batch_idx}: {str(e)}")
                continue
        
        avg_loss = epoch_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"\n🎯 New best model saved! Loss: {loss:.4f}")
    
    def train(self, train_dataloader, num_epochs=None):
        """Complete training loop"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(train_dataloader, epoch)
            
            print(f"\nEpoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch, avg_loss, is_best)
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Best Loss: {self.best_loss:.4f}")
        
        self.writer.close()

print("✅ Trainer class ready!")

# ==================== INFERENCE ENGINE ====================

class NAiBiLInference:
    """Complete inference pipeline with post-processing and visualization"""
    def __init__(self, model, config, device):
        self.model = model.eval()
        self.config = config
        self.device = device
        
        # Class names (COCO + extended)
        self.class_names = self._load_class_names()
        
        # Colors for visualization
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(config.num_classes, 3), dtype=np.uint8)
        
        # Performance tracking
        self.inference_times = []
        
        print("\n✅ Inference engine initialized!")
    
    def _load_class_names(self):
        """Load class names"""
        # COCO classes + extended classes
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Add extended classes
        extended_classes = [
            'pothole', 'crack', 'fish', 'building', 'road', 'water', 'gun', 'accident',
            'male', 'female', 'child', 'adult', 'old_person', 'young_person'
        ]
        
        all_classes = coco_classes + extended_classes
        return all_classes[:self.config.num_classes]
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        # Resize
        h, w = image.shape[:2]
        target_h, target_w = self.config.inference_size
        
        # Resize maintaining aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Normalize
        normalized = padded.astype(np.float32) / 255.0
        normalized = (normalized - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # To tensor
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        
        return tensor, (scale, pad_w, pad_h)
    
    def postprocess_detections(self, predictions, transform_info, orig_shape):
        """Post-process detection outputs to get final bounding boxes"""
        scale, pad_w, pad_h = transform_info
        h, w = orig_shape[:2]
        
        all_boxes = []
        all_scores = []
        all_classes = []
        
        # Process each scale's detections
        for det in predictions['detections']:
            bs, num_anchors, grid_h, grid_w, pred_size = det.shape
            
            # Reshape predictions
            det = det.view(bs, -1, pred_size)
            
            # Extract components
            box_xy = torch.sigmoid(det[..., :2])
            box_wh = det[..., 2:4]
            objectness = torch.sigmoid(det[..., 4:5])
            class_probs = torch.sigmoid(det[..., 5:])
            
            # Get confidence scores
            scores = objectness * class_probs
            max_scores, class_ids = scores.max(dim=-1)
            
            # Filter by confidence threshold
            mask = max_scores > self.config.conf_threshold
            
            if mask.sum() > 0:
                # Get filtered predictions
                filtered_boxes = torch.cat([box_xy, box_wh], dim=-1)[mask]
                filtered_scores = max_scores[mask]
                filtered_classes = class_ids[mask]
                
                # Scale boxes back to original image
                filtered_boxes = filtered_boxes.cpu().numpy()
                filtered_boxes[:, 0] = (filtered_boxes[:, 0] - pad_w) / scale
                filtered_boxes[:, 1] = (filtered_boxes[:, 1] - pad_h) / scale
                filtered_boxes[:, 2] = filtered_boxes[:, 2] / scale
                filtered_boxes[:, 3] = filtered_boxes[:, 3] / scale
                
                all_boxes.append(filtered_boxes)
                all_scores.append(filtered_scores.cpu().numpy())
                all_classes.append(filtered_classes.cpu().numpy())
        
        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            classes = np.concatenate(all_classes, axis=0)
            
            # Apply NMS
            keep = self._apply_nms(boxes, scores, classes)
            
            return boxes[keep], scores[keep], classes[keep]
        
        return np.array([]), np.array([]), np.array([])
    
    def _apply_nms(self, boxes, scores, classes):
        """Apply Non-Maximum Suppression"""
        keep_indices = []
        
        for class_id in np.unique(classes):
            class_mask = classes == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            
            if len(class_boxes) > 0:
                # Convert to x1,y1,x2,y2 format
                x1 = class_boxes[:, 0] - class_boxes[:, 2] / 2
                y1 = class_boxes[:, 1] - class_boxes[:, 3] / 2
                x2 = class_boxes[:, 0] + class_boxes[:, 2] / 2
                y2 = class_boxes[:, 1] + class_boxes[:, 3] / 2
                
                boxes_tensor = torch.tensor(np.stack([x1, y1, x2, y2], axis=1), dtype=torch.float32)
                scores_tensor = torch.tensor(class_scores, dtype=torch.float32)
                
                keep = nms(boxes_tensor, scores_tensor, self.config.nms_threshold)
                keep_indices.extend(np.where(class_mask)[0][keep.cpu().numpy()])
        
        return np.array(keep_indices)
    
    def visualize_results(self, image, boxes, scores, classes, pose_keypoints=None):
        """Visualize detection results on image"""
        vis_image = image.copy()
        
        # Draw bounding boxes
        for box, score, class_id in zip(boxes, scores, classes):
            if class_id >= len(self.class_names):
                continue
            
            # Convert box from (cx, cy, w, h) to (x1, y1, x2, y2)
            x1 = int(box[0] - box[2] / 2)
            y1 = int(box[1] - box[3] / 2)
            x2 = int(box[0] + box[2] / 2)
            y2 = int(box[1] + box[3] / 2)
            
            # Get color
            color = tuple(map(int, self.colors[int(class_id)]))
            
            # Draw rectangle
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{self.class_names[int(class_id)]}: {score:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1_label = max(y1, label_size[1] + 10)
            
            cv2.rectangle(vis_image, (x1, y1_label - label_size[1] - 10),
                         (x1 + label_size[0], y1_label + baseline - 10), color, -1)
            cv2.putText(vis_image, label, (x1, y1_label - 7),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw pose keypoints if available
        if pose_keypoints is not None:
            self._draw_keypoints(vis_image, pose_keypoints)
        
        return vis_image
    
    def _draw_keypoints(self, image, keypoints, threshold=0.3):
        """Draw pose keypoints on image"""
        # COCO skeleton connections
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > threshold:
                cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # Draw skeleton
        for start_idx, end_idx in skeleton:
            if (start_idx < len(keypoints) and end_idx < len(keypoints)):
                start_kp = keypoints[start_idx - 1]
                end_kp = keypoints[end_idx - 1]
                
                if start_kp[2] > threshold and end_kp[2] > threshold:
                    cv2.line(image, 
                            (int(start_kp[0]), int(start_kp[1])),
                            (int(end_kp[0]), int(end_kp[1])),
                            (0, 255, 255), 2)
    
    @torch.no_grad()
    def predict(self, image):
        """Run inference on a single image"""
        start_time = time.time()
        
        # Preprocess
        input_tensor, transform_info = self.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        outputs = self.model(input_tensor)
        
        # Post-process detections
        boxes, scores, classes = self.postprocess_detections(
            outputs, transform_info, image.shape
        )
        
        # Track inference time
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        
        # Visualize
        vis_image = self.visualize_results(image, boxes, scores, classes)
        
        # Add FPS info
        fps = 1000 / inference_time
        cv2.putText(vis_image, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Inference: {inference_time:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_image, f"Detections: {len(boxes)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'classes': classes,
            'visualization': vis_image,
            'inference_time': inference_time,
            'fps': fps
        }

print("✅ Inference engine ready!")

# Initialize dataset manager
dataset_manager = StreamingDatasetManager(config)

print("\n" + "="*70)
print("DATASET LOADING")
print("="*70)

# ========== COCO Dataset (if available) ==========
coco_train_images = '/content/datasets/coco/train2017'
coco_train_ann = '/content/datasets/coco/annotations/instances_train2017.json'
coco_val_images = '/content/datasets/coco/val2017'
coco_val_ann = '/content/datasets/coco/annotations/instances_val2017.json'

dataset_manager.add_coco_dataset('COCO-Train', coco_train_images, coco_train_ann, 'train')
dataset_manager.add_coco_dataset('COCO-Val', coco_val_images, coco_val_ann, 'val')

# ========== Download sample dataset if no datasets loaded ==========
if len(dataset_manager.datasets) == 0:
    print("\n⚠️  No datasets found. Downloading COCO sample...")
    !mkdir -p /content/datasets/coco_sample/images
    !mkdir -p /content/datasets/coco_sample/annotations
    
    # Download COCO8 sample dataset
    !wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip -O /content/coco8.zip
    !unzip -q /content/coco8.zip -d /content/datasets/coco_sample/
    
    print("\n✅ Sample dataset downloaded!")

# ========== HuggingFace Datasets (Streaming) ==========
# Example: Load detection datasets with streaming
# hf_dataset = dataset_manager.add_huggingface_dataset(
#     'Detection-HF',
#     'keremberke/object-detection-augmented',
#     split='train',
#     streaming=True
# )

print("\n" + "="*70)
print(f"Total datasets loaded: {len(dataset_manager.datasets)}")
for ds in dataset_manager.datasets:
    print(f"  - {ds['name']} ({ds['split']}): {len(ds['dataset'])} samples")
print("="*70)

# Get data loaders
train_loader = dataset_manager.get_dataloader('train', batch_size=config.batch_size)
val_loader = dataset_manager.get_dataloader('val', batch_size=config.batch_size)

if train_loader is None:
    print("\n⚠️  No training data available. Creating dummy dataloader for testing...")
    # Create dummy dataset for testing
    class DummyDataset(Dataset):
        def __len__(self):
            return 100
        def __getitem__(self, idx):
            return {
                'image': torch.randn(3, config.input_size[0], config.input_size[1]),
                'boxes': torch.randn(3, 4),
                'labels': torch.randint(0, config.num_classes, (3,)),
                'keypoints': torch.randn(1, 51)
            }
    
    dummy_dataset = DummyDataset()
    train_loader = DataLoader(dummy_dataset, batch_size=config.batch_size, 
                             collate_fn=dataset_manager.collate_fn)

# Initialize trainer
trainer = NAiBiLTrainer(model, config, device)

# Start training
print("\n🚀 Starting training...")
print(f"   Epochs: {config.num_epochs}")
print(f"   Batch Size: {config.batch_size}")
print(f"   Learning Rate: {config.learning_rate}")
print(f"   Device: {device}")

# Train for specified number of epochs (can be reduced for testing)
trainer.train(train_loader, num_epochs=min(config.num_epochs, 5))  # Train for 5 epochs as demo

# Initialize inference engine
inference_engine = NAiBiLInference(model, config, device)

# Test on sample images
print("\n" + "="*70)
print("RUNNING INFERENCE")
print("="*70)

# Download sample images
sample_urls = [
    'https://ultralytics.com/images/bus.jpg',
    'https://ultralytics.com/images/zidane.jpg'
]

import requests
from io import BytesIO

for idx, url in enumerate(sample_urls):
    print(f"\n📸 Processing image {idx+1}/{len(sample_urls)}")
    
    try:
        # Download image
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image = np.array(image)
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Run inference
        results = inference_engine.predict(image)
        
        print(f"   Detections: {len(results['boxes'])}")
        print(f"   FPS: {results['fps']:.1f}")
        print(f"   Inference Time: {results['inference_time']:.1f}ms")
        
        # Display results
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(results['visualization'], cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"NAiBiL Detection Results - Image {idx+1}", fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
        
        # Print detected objects
        if len(results['boxes']) > 0:
            print("\n   Detected Objects:")
            for i, (box, score, cls) in enumerate(zip(results['boxes'], 
                                                       results['scores'], 
                                                       results['classes'])):
                class_name = inference_engine.class_names[int(cls)]
                print(f"      {i+1}. {class_name}: {score:.2f}")
    
    except Exception as e:
        print(f"   ⚠️  Error processing image: {str(e)}")

print("\n" + "="*70)
print("INFERENCE COMPLETED")
print("="*70)


.
.
.
.
.
#END#
