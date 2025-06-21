# inference_pp.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd,
    CenterSpatialCropd, Resized, ToTensord
)
from monai.data import DataLoader, Dataset
from monai.networks.nets import BasicUNetPlusPlus

# ==================== 参数配置 ====================
MODEL_PATH = "weld_seg_unetpp.pth"
IMAGE_PATH = "dataset/images/1.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURES = (16, 32, 64, 128, 256, 512)

# ==================== 加载图像并保存灰度 ====================
img = Image.open(IMAGE_PATH).convert("L")
img.save("temp_gray_image.png")

# ==================== MONAI 流水线 ====================
test_data = [{"image": "temp_gray_image.png"}]
transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    CenterSpatialCropd(keys=["image"], roi_size=(512, 512)),
    Resized(keys=["image"], spatial_size=(512, 512)),
    ToTensord(keys=["image"]),
])
dataset = Dataset(data=test_data, transform=transforms)
loader = DataLoader(dataset, batch_size=1)

# ==================== 加载模型 ====================
model = BasicUNetPlusPlus(spatial_dims=2, in_channels=1, out_channels=1, features=FEATURES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ==================== 推理 ====================
with torch.no_grad():
    for batch in loader:
        image = batch["image"].to(DEVICE)
        output = model(image)
        if isinstance(output, list):
            output = output[0]
        pred = torch.sigmoid(output).squeeze().cpu().numpy()

# ==================== 后处理 ====================
binary_mask = (pred > 0.5).astype(np.uint8)
corrected_mask = binary_mask.T * 255
final_mask = corrected_mask.copy()

# ==================== 可视化 ====================
original = np.array(Image.open("temp_gray_image.png"))
def center_crop(img, crop_size=512):
    h, w = img.shape[:2]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h:start_h+crop_size, start_w:start_w+crop_size]

cropped_original = center_crop(original, 512)
prob_map = (pred * 255).astype(np.uint8).T

plt.figure(figsize=(12, 12))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.subplot(1, 2, 1)
plt.imshow(cropped_original, cmap="gray")
plt.title("Original Image")
plt.axis("off")

overlay = np.zeros((final_mask.shape[0], final_mask.shape[1], 4), dtype=np.float32)
overlay[..., 0] = 7.0
overlay[..., 1] = 7.0
overlay[..., 3] = 0.3 * final_mask.astype(np.float32)
overlay[..., 3] = (final_mask > 0).astype(np.float32) * 0.25  # Alpha, 半透明
plt.subplot(1, 2, 2)
plt.imshow(cropped_original, cmap="gray")
plt.imshow(overlay, interpolation="none")
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()
