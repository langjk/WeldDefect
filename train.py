import os
import torch
from torch.optim import Adam
from monai.metrics import DiceMetric
from monai.inferers import SimpleInferer
from monai.utils import set_determinism
from monai.transforms import AsDiscrete
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from glob import glob
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate

from dataset import get_weld_dataset
from losses import DiceTverskyLoss
from monai.networks.nets import BasicUNetPlusPlus
from torch.cuda.amp import autocast, GradScaler

set_determinism(42)

# -------------------
# 参数设置
# -------------------
image_dir = "dataset/images"
mask_dir = "dataset/masks"
num_epochs = 50
lr = 1e-4
batch_size = 1 
image_size = (512, 512)
num_workers = 0

os.makedirs("viz", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# 数据划分 & 加载
# -------------------
image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
mask_paths = [os.path.join(mask_dir, os.path.basename(p).replace(".png", "_mask.png")) for p in image_paths]
data_dicts = [{"image": img, "mask": msk} for img, msk in zip(image_paths, mask_paths)]
train_dicts, val_dicts = train_test_split(data_dicts, test_size=0.2, random_state=42)

transforms = get_weld_dataset(image_dir, mask_dir, image_size, batch_size).dataset.transform
train_ds = Dataset(data=train_dicts, transform=transforms)
val_ds = Dataset(data=val_dicts, transform=transforms)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=list_data_collate)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)

# -------------------
# 模型 + 损失 + 优化器
# -------------------
model = BasicUNetPlusPlus(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    features = (16, 32, 64, 128, 256, 512)
).to(device)

criterion = DiceTverskyLoss(alpha=0.7, beta=0.3)
optimizer = Adam(model.parameters(), lr=lr)
scaler = GradScaler()
dice_metric = DiceMetric(include_background=False, reduction="mean")
inferer = SimpleInferer()
post_pred = AsDiscrete(threshold=0.5)
post_label = AsDiscrete(threshold=0.5)

# -------------------
# 训练 & 验证循环
# -------------------
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    loop = tqdm(train_loader)

    for batch in loop:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        # 自动混合精度上下文
        with autocast():
            outputs = model(images)
            
            if isinstance(outputs, list):  # 若模型有多输出
                outputs = outputs[0]
            
            loss = criterion(outputs, masks)

        # 反向传播与优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} - Loss: {epoch_loss / len(train_loader):.4f}")

    # 验证
    model.eval()
    dice_metric.reset()
    with torch.no_grad():
        for i, val_batch in enumerate(val_loader):
            val_images = val_batch["image"].to(device)
            val_labels = val_batch["mask"].to(device)
            val_outputs = model(val_images)

            if isinstance(val_outputs, list):
                val_outputs = val_outputs[0]

            val_outputs_bin = post_pred(val_outputs)
            val_labels_bin = post_label(val_labels)
            dice_metric(y_pred=val_outputs_bin, y=val_labels_bin)

            if i == 0:
                img = val_images[0, 0].cpu().numpy()
                pred = val_outputs_bin[0, 0].cpu().numpy()
                label = val_labels_bin[0, 0].cpu().numpy()

                plt.figure(figsize=(12, 4))
                plt.subplot(1, 3, 1)
                plt.imshow(img, cmap='gray')
                plt.title("Image")

                plt.subplot(1, 3, 2)
                plt.imshow(pred, cmap='gray')
                plt.title("Prediction")

                plt.subplot(1, 3, 3)
                plt.imshow(label, cmap='gray')
                plt.title("Ground Truth")

                plt.savefig(f"viz/epoch_{epoch+1:02d}_val_overlay.png")
                plt.close()

    val_dice = dice_metric.aggregate().item()
    print(f"Epoch {epoch+1} - Validation Dice: {val_dice:.4f}")
    torch.cuda.empty_cache()

# -------------------
# 保存模型
# -------------------
torch.save(model.state_dict(), "weld_seg_unetpp.pth")