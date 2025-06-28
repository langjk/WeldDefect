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
IMAGE_PATH = "fakeshow/60-4.jpg"
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
# 加载完整的原始图像
original_full = np.array(Image.open(IMAGE_PATH).convert("L"))
original_gray = np.array(Image.open("temp_gray_image.png"))

def center_crop(img, crop_size=512):
    h, w = img.shape[:2]
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h:start_h+crop_size, start_w:start_w+crop_size], start_h, start_w

cropped_original, crop_start_h, crop_start_w = center_crop(original_gray, 512)

# 比例尺：120像素 = 1000um
PIXELS_PER_UM = 120 / 1000

# 找到连通域
contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 创建完整原始图像的彩色版本
result_image_full = cv2.cvtColor(original_full, cv2.COLOR_GRAY2RGB)

# 为每个连通域绘制最小外接矩形和标注（映射回原始图像坐标）
for i, contour in enumerate(contours):
    if cv2.contourArea(contour) > 50:  # 过滤小的噪声区域
        # 计算最小外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        
        # 映射回原始图像坐标
        orig_x = x + crop_start_w
        orig_y = y + crop_start_h
        
        # 转换为um单位
        width_um = w / PIXELS_PER_UM
        height_um = h / PIXELS_PER_UM
        
        # 绘制矩形框
        cv2.rectangle(result_image_full, (orig_x, orig_y), (orig_x + w, orig_y + h), (0, 255, 0), 2)
        
        # 添加尺寸标注
        label = f"{width_um:.0f}x{height_um:.0f}um"
        
        # 计算文字位置（矩形右上角）
        label_x = orig_x + w + 5
        label_y = orig_y + 15
        
        # 确保标注不超出图像边界
        if label_x > result_image_full.shape[1] - 100:
            label_x = orig_x - 80
        if label_y < 20:
            label_y = orig_y + h - 5
            
        # 绘制文字背景
        cv2.rectangle(result_image_full, (label_x - 3, label_y - 12), 
                     (label_x + len(label) * 8, label_y + 3), (255, 255, 255), -1)
        
        # 绘制文字
        cv2.putText(result_image_full, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

plt.figure(figsize=(12, 8))
plt.imshow(result_image_full)
plt.axis("off")

plt.tight_layout()
plt.savefig("detection_results.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"检测到 {len(contours)} 个连通域")
print("结果已保存为 detection_results.png")
