
# inference.py
import torch
from model import get_unet_model
from monai.transforms import (
    Compose, LoadImaged, Resized, ScaleIntensityd, EnsureChannelFirstd, ToTensord, Orientationd, CenterSpatialCropd
)
from monai.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy.ma as ma
# 参数
MODEL_PATH = "weld_seg_unet-Cut.pth"
IMAGE_PATH = "dataset/images/8.png"  # 1 15no 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载灰度图 & 保存为临时文件（PIL） → 确保单通道
img = Image.open(IMAGE_PATH).convert("L")

img.save("temp_gray_image.png")  # 保存为灰度图供 MONAI 读取

# 数据管道，使用 dict 结构
test_data = [{"image": "temp_gray_image.png"}]

transforms = Compose([
    LoadImaged(keys=["image"], image_only=False),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityd(keys=["image"]),
    CenterSpatialCropd(keys=["image"], roi_size=(512, 512)),
    Resized(keys=["image"], spatial_size=(512, 512)),
    ToTensord(keys=["image"]),
])

dataset = Dataset(data=test_data, transform=transforms)
loader = DataLoader(dataset, batch_size=1)

# 加载模型
model = get_unet_model().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 推理
with torch.no_grad():
    for batch in loader:
        image = batch["image"].to(DEVICE)
        output = model(image)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        # 提高置信度阈值以减少假阳性
        binary_mask = (pred > 0.8).astype(np.uint8)
        
# 转置后添加垂直翻转（上下颠倒修正）
corrected_mask = binary_mask.T
corrected_mask = corrected_mask * 255

# 改进的形态学处理 - 使用更大的核
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 先闭运算填充空洞，再开运算去除噪声
closed_mask = cv2.morphologyEx(corrected_mask, cv2.MORPH_CLOSE, kernel)
opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

final_mask = opened_mask

# 使用 connectedComponentsWithStats 进行连通区域分析
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask.astype(np.uint8), connectivity=8)

# 增加最小区域面积阈值以过滤更多噪声
min_area = 50
for i in range(1, num_labels):  # 从1开始，跳过背景（标签0）
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        final_mask[labels == i] = 0  # 将面积小于10的区域设为0
final_mask = (final_mask * 255).astype(np.uint8)
# ======================================
original = np.array(Image.open("temp_gray_image.png"))
def center_crop(img, crop_size=512):
    """ 从图像中心裁剪指定大小 """
    h, w = img.shape[:2]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h:start_h+crop_size, start_w:start_w+crop_size]

# 应用裁剪（假设原图是正方形）
cropped_original = center_crop(original, 512)

# 添加这行用于显示 sigmoid 概率图（归一化到 0-255）
prob_map = (pred * 255).astype(np.uint8)  # 保留置信度
prob_map = prob_map.T
# ================= 可视化 ==================
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
plt.subplot(1, 2, 2)
plt.imshow(cropped_original, cmap="gray")
plt.imshow(overlay, interpolation="none")
plt.title("Overlay")
plt.axis("off")

plt.tight_layout()
plt.show()