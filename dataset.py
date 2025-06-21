# dataset.py
import os
from glob import glob
import torch
from torch.utils.data import DataLoader
from monai.data import (Dataset, list_data_collate)
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, Resized,
    RandFlipd, RandRotate90d, RandZoomd, ToTensord, Lambdad, CenterSpatialCropd, RandCropByPosNegLabeld, EnsureTyped,   
)


def get_weld_dataset(image_dir, mask_dir, image_size=(512, 512), batch_size=4, num_workers=0):
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    mask_paths = [
        os.path.join(mask_dir, os.path.basename(p).replace(".png", "_mask.png"))
        for p in image_paths
    ]
    data_dicts = [{"image": img, "mask": msk} for img, msk in zip(image_paths, mask_paths)]

    transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        Lambdad(keys=["image"], func=lambda x: x.mean(axis=-1, keepdims=True) if x.ndim == 3 else x),
        Lambdad(keys=["mask"], func=lambda x: x / 255.0 if x.max() > 1 else x),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityd(keys=["image"]),
        # CenterSpatialCropd(keys=["image", "mask"], roi_size=(512, 512)),
        RandCropByPosNegLabeld(
            keys=["image", "mask"],
            label_key="mask",
            spatial_size=(256, 256),
            pos=2,   # 正样本比例（有掩膜的）
            neg=1,   # 负样本比例
            num_samples=16,  # 每对图像掩膜裁剪几个 patch
            image_key="image",
            image_threshold=0
        ),
        # Resized(keys=["image", "mask"], spatial_size=image_size),
        RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
        RandRotate90d(keys=["image", "mask"], prob=0.5),
        RandZoomd(keys=["image", "mask"], min_zoom=0.9, max_zoom=1.1, prob=0.2),
        ToTensord(keys=["image", "mask"]),
        EnsureTyped(keys=["image", "mask"]),
    ])

    dataset = Dataset(data=data_dicts, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,collate_fn=list_data_collate)

    return dataloader