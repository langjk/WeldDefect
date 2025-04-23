import torch
from torch.optim import Adam
from monai.metrics import DiceMetric
from monai.inferers import SimpleInferer
from monai.utils import set_determinism
from tqdm import tqdm

from dataset import get_weld_dataset
from model import get_unet_model
from losses import DiceFocalLoss

set_determinism(42)

# 参数设置
image_dir = "dataset/images"
mask_dir = "dataset/masks"
num_epochs = 10
lr = 5e-4
batch_size = 4
image_size = (512, 512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
train_loader = get_weld_dataset(image_dir, mask_dir, image_size, batch_size)
for batch in train_loader:
    masks = batch["mask"]
    print("Mean mask value:", masks.mean().item())
    break

# 模型 + 损失 + 优化器
model = get_unet_model().to(device)
criterion = DiceFocalLoss()
optimizer = Adam(model.parameters(), lr=lr)
dice_metric = DiceMetric(include_background=False, reduction="mean")
inferer = SimpleInferer()

# 训练循环
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    loop = tqdm(train_loader)

    for batch in loop:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} - Loss: {epoch_loss / len(train_loader):.4f}")

    # 🎯 验证（可选）
    # model.eval()
    # with torch.no_grad():
    #     for batch in val_loader:
    #         ...
    

# 保存模型
torch.save(model.state_dict(), "weld_seg_unet.pth")