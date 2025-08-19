import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate
from monai.transforms import AsDiscrete
from monai.networks.nets import BasicUNetPlusPlus
from sklearn.model_selection import train_test_split
from glob import glob
from dataset import get_weld_dataset
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

def calculate_metrics(pred, target):
    """
    计算Dice、IoU、Recall、Precision指标
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # 计算基本统计量
    tp = np.sum((pred == 1) & (target == 1))  # True Positive
    tn = np.sum((pred == 0) & (target == 0))  # True Negative
    fp = np.sum((pred == 1) & (target == 0))  # False Positive
    fn = np.sum((pred == 0) & (target == 1))  # False Negative
    
    # 避免除零错误
    epsilon = 1e-7
    
    # Dice Score
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    
    # IoU (Intersection over Union)
    iou = tp / (tp + fp + fn + epsilon)
    
    # Recall (Sensitivity)
    recall = tp / (tp + fn + epsilon)
    
    # Precision
    precision = tp / (tp + fp + epsilon)
    
    return dice, iou, recall, precision

def evaluate_model(model_path, data_loader, device):
    """
    评估模型性能
    """
    # 加载模型
    model = BasicUNetPlusPlus(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        features=(16, 32, 64, 128, 256, 512)
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 使用与inference.py相同的阈值设置
    post_pred = AsDiscrete(threshold=0.8)  # 提高阈值减少假阳性
    post_label = AsDiscrete(threshold=0.5)
    
    all_dice = []
    all_iou = []
    all_recall = []
    all_precision = []
    
    print("开始评估模型...")
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # 模型预测
            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[0]
            
            # 后处理
            pred_bin = post_pred(outputs)
            label_bin = post_label(masks)
            
            # 转换为numpy数组
            pred_np = pred_bin[0, 0].cpu().numpy()
            label_np = label_bin[0, 0].cpu().numpy()
            
            # 计算指标
            dice, iou, recall, precision = calculate_metrics(pred_np, label_np)
            
            all_dice.append(dice)
            all_iou.append(iou)
            all_recall.append(recall)
            all_precision.append(precision)
            
            # 保存前5个样本的可视化结果
            if i < 5:
                img_np = images[0, 0].cpu().numpy()
                
                plt.figure(figsize=(15, 5))
                
                plt.subplot(1, 4, 1)
                plt.imshow(img_np, cmap='gray')
                plt.title("Original Image")
                plt.axis('off')
                
                plt.subplot(1, 4, 2)
                plt.imshow(label_np, cmap='gray')
                plt.title("Ground Truth")
                plt.axis('off')
                
                plt.subplot(1, 4, 3)
                plt.imshow(pred_np, cmap='gray')
                plt.title("Prediction")
                plt.axis('off')
                
                # 叠加显示
                plt.subplot(1, 4, 4)
                plt.imshow(img_np, cmap='gray')
                plt.imshow(pred_np, cmap='Reds', alpha=0.3)
                plt.imshow(label_np, cmap='Greens', alpha=0.3)
                plt.title("Overlay (Red: Pred, Green: GT)")
                plt.axis('off')
                
                plt.suptitle(f'Sample {i+1} - Dice: {dice:.4f}, IoU: {iou:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')
                plt.tight_layout()
                plt.savefig(f"eval_sample_{i+1}.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"Sample {i+1}/{len(data_loader)} - Dice: {dice:.4f}, IoU: {iou:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    
    # 计算平均指标
    mean_dice = np.mean(all_dice)
    mean_iou = np.mean(all_iou)
    mean_recall = np.mean(all_recall)
    mean_precision = np.mean(all_precision)
    
    std_dice = np.std(all_dice)
    std_iou = np.std(all_iou)
    std_recall = np.std(all_recall)
    std_precision = np.std(all_precision)
    
    # 打印结果
    print("\n" + "="*60)
    print("评估结果汇总:")
    print("="*60)
    print(f"Dice Score:    {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"IoU:           {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Recall:        {mean_recall:.4f} ± {std_recall:.4f}")
    print(f"Precision:     {mean_precision:.4f} ± {std_precision:.4f}")
    print("="*60)
    
    # 保存结果到文件
    with open("evaluation_results.txt", "w") as f:
        f.write("模型评估结果\n")
        f.write("="*60 + "\n")
        f.write(f"Dice Score:    {mean_dice:.6f} ± {std_dice:.6f}\n")
        f.write(f"IoU:           {mean_iou:.6f} ± {std_iou:.6f}\n")
        f.write(f"Recall:        {mean_recall:.6f} ± {std_recall:.6f}\n")
        f.write(f"Precision:     {mean_precision:.6f} ± {std_precision:.6f}\n")
        f.write("="*60 + "\n\n")
        
        f.write("详细结果:\n")
        for i, (d, iou, r, p) in enumerate(zip(all_dice, all_iou, all_recall, all_precision)):
            f.write(f"Sample {i+1}: Dice={d:.6f}, IoU={iou:.6f}, Recall={r:.6f}, Precision={p:.6f}\n")
    
    return {
        'dice': (mean_dice, std_dice),
        'iou': (mean_iou, std_iou),
        'recall': (mean_recall, std_recall),
        'precision': (mean_precision, std_precision)
    }

def main():
    # 参数设置
    image_dir = "dataset/images"
    mask_dir = "dataset/masks"
    model_path = "weld_seg_unetpp.pth"
    image_size = (512, 512)
    batch_size = 1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return
    
    # 数据加载
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    mask_paths = [os.path.join(mask_dir, os.path.basename(p).replace(".png", "_mask.png")) for p in image_paths]
    data_dicts = [{"image": img, "mask": msk} for img, msk in zip(image_paths, mask_paths)]
    
    # 使用相同的数据划分
    train_dicts, val_dicts = train_test_split(data_dicts, test_size=0.2, random_state=42)
    
    # 创建验证数据集
    transforms = get_weld_dataset(image_dir, mask_dir, image_size, batch_size).dataset.transform
    val_ds = Dataset(data=val_dicts, transform=transforms)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=list_data_collate)
    
    print(f"验证集样本数: {len(val_dicts)}")
    
    # 评估模型
    results = evaluate_model(model_path, val_loader, device)
    
    print(f"\n评估完成! 结果已保存到 evaluation_results.txt")
    print(f"可视化结果已保存为 eval_sample_*.png")

if __name__ == "__main__":
    main()