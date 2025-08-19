# dataset.py
import os
from glob import glob
import torch
from torch.utils.data import DataLoader
from monai.data import (Dataset, list_data_collate)
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    RandFlipd, RandRotate90d, RandZoomd, RandAdjustContrastd, RandGaussianNoised,
    RandGaussianSmoothd, RandShiftIntensityd, RandBiasFieldd, RandGridDistortiond,
    ToTensord, RandCropByPosNegLabeld, EnsureTyped, RepeatChanneld, 
    ScaleIntensityRanged, ThresholdIntensityd, Lambdad
)
import numpy as np

# 全局定义的可序列化函数
def rgb_to_grayscale_safe(x):
    """安全的RGB转灰度函数，可被多进程序列化"""
    if x.ndim == 3 and x.shape[0] == 3:  # RGB图像 (C, H, W)
        # 使用标准的RGB权重转换为灰度
        weights = np.array([0.299, 0.587, 0.114]).reshape(3, 1, 1)
        return np.sum(x * weights, axis=0, keepdims=True)
    elif x.ndim == 3 and x.shape[0] == 1:  # 已经是单通道
        return x
    elif x.ndim == 2:  # 2D灰度图
        return x[np.newaxis, ...]  # 添加通道维度
    else:
        return x

def normalize_mask_safe(x):
    """安全的掩码归一化函数"""
    return x / 255.0 if x.max() > 1 else x

def get_weld_dataset(image_dir, mask_dir, image_size=(512, 512), batch_size=4, num_workers=0):
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    mask_paths = [
        os.path.join(mask_dir, os.path.basename(p).replace(".png", "_mask.png"))
        for p in image_paths
    ]
    data_dicts = [{"image": img, "mask": msk} for img, msk in zip(image_paths, mask_paths)]

    transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        
        # RGB转灰度：使用全局定义的可序列化函数
        Lambdad(keys=["image"], func=rgb_to_grayscale_safe),
        
        # 掩码归一化：使用全局定义的可序列化函数  
        Lambdad(keys=["mask"], func=normalize_mask_safe),
        
        # 图像强度归一化
        ScaleIntensityd(keys=["image"]),

        # 优化的困难负样本挖掘 - 减少采样数量以提高速度
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=(256, 256),
            pos=2, neg=1, num_samples=6,  # 减少到6个样本以提高速度
            image_key="image",
            image_threshold=0
        ),

        # 🔁 核心几何变换增强（保留最有效的）
        RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
        RandRotate90d(keys=["image", "mask"], prob=0.5),
        RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.15),
        # 移除RandGridDistortiond（耗时且效果有限）

        # 🌈 核心强度增强（降低概率以提高速度）
        RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.7, 1.5)),
        RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.05),
        # 移除一些耗时的增强

        ToTensord(keys=["image", "mask"]),
        EnsureTyped(keys=["image", "mask"]),
    ])

    dataset = Dataset(data=data_dicts, transform=transforms)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=list_data_collate
    )

    return dataloader