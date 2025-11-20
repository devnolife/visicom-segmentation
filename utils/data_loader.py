"""
Data Loader untuk Flood Segmentation
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FloodSegmentationDataset(Dataset):
    """Dataset untuk segmentasi banjir"""
    
    def __init__(self, image_dir, mask_dir=None, transform=None, is_train=True):
        """
        Args:
            image_dir: Direktori berisi gambar
            mask_dir: Direktori berisi mask (label segmentasi)
            transform: Transformasi augmentasi
            is_train: Mode training atau inference
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train
        
        # Daftar file gambar
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load gambar
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.is_train and self.mask_dir:
            # Load mask
            mask_name = self.images[idx].replace('.jpg', '.png').replace('.jpeg', '.png').replace('.webp', '.png')
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                # Jika mask tidak ada, buat mask kosong
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
            # Apply transformations
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            return image, mask.long()
        else:
            # Inference mode
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            return image, self.images[idx]


def get_train_transform(image_size=512):
    """Transformasi untuk training dengan augmentasi"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transform(image_size=512):
    """Transformasi untuk validasi (tanpa augmentasi)"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_inference_transform(image_size=512):
    """Transformasi untuk inference"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def create_dataloaders(image_dir, mask_dir, batch_size=4, image_size=512, val_split=0.2):
    """
    Membuat train dan validation dataloaders
    
    Args:
        image_dir: Direktori gambar
        mask_dir: Direktori mask
        batch_size: Ukuran batch
        image_size: Ukuran gambar setelah resize
        val_split: Proporsi data untuk validasi
    
    Returns:
        train_loader, val_loader
    """
    # Dataset lengkap
    full_dataset = FloodSegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=get_train_transform(image_size),
        is_train=True
    )
    
    # Split train/val
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update transform untuk validation
    val_dataset.dataset.transform = get_val_transform(image_size)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set 0 untuk Windows
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader
