

import os
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class NepalDataset(Dataset):
    def __init__(self, data_path, subset='train', transform=None, test_mode=True, augment=False):
        self.images_dir = os.path.join(data_path, 'images', subset)
        self.masks_dir = os.path.join(data_path, 'masks', subset)
        self.transform = transform
        self.test_mode = test_mode
        self.augment = augment
        self.images = [f for f in os.listdir(self.images_dir) if f.endswith('.tif')]
        self.images.sort()

        # Augmentation for image only
        if self.augment:
            self.aug_transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=10, p=0.3)
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.images_dir, img_filename)

        # Load image
        img = tiff.imread(img_path)
        
        # Ensure image is 3-channel and uint8
        if img.ndim == 2:
            img = np.stack([img]*3, axis=2)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        
        # Ensure uint8 dtype for consistent processing
        if img.dtype != np.uint8:
            if img.dtype == np.float32 or img.dtype == np.float64:
                # If already normalized, denormalize first
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # Apply augmentation (works on uint8)
        if self.augment:
            img = self.aug_transform(image=img)['image']

        # Apply transform (converts to tensor and normalizes to [0,1])
        if self.transform is not None:
            img = self.transform(image=img)['image']

        if self.test_mode:
            return img, img_path

        # Load mask
        mask_path = os.path.join(self.masks_dir, img_filename.replace(".tif","_mask.tif"))
        mask = tiff.imread(mask_path)
        mask = (mask > 0).astype(np.float32)
        
        if mask.ndim == 2:
            mask = mask[np.newaxis, :, :]
        elif mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask.transpose(2, 0, 1)

        return img, torch.from_numpy(mask).float(), img_path, mask_path

def get_transform():
    """
    Transform that properly converts uint8 images to float32 tensors normalized to [0,1]
    """
    return A.Compose([
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2()
    ])


