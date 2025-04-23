# debug_loader.py

import matplotlib.pyplot as plt
from dataset import get_weld_dataset


def visualize_batch(sample_batch):
    image_tensor = sample_batch["image"][0][0]  # 第一个样本，第一个通道
    mask_tensor = sample_batch["mask"][0][0]

    image_np = image_tensor.numpy()
    mask_np = mask_tensor.numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_np, cmap="gray")
    plt.imshow(mask_np, cmap="Reds", alpha=0.5)
    plt.title("Image + Mask Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_dir = "dataset/images"
    mask_dir = "dataset/masks"

    dataloader = get_weld_dataset(image_dir, mask_dir, image_size=(512, 512), batch_size=1, num_workers=0)

    batch = next(iter(dataloader))
    visualize_batch(batch)
