import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from monai.data import Dataset, list_data_collate
from monai.transforms import AsDiscrete, Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, ToTensord, Lambdad, EnsureTyped
from monai.networks.nets import BasicUNetPlusPlus
from sklearn.model_selection import train_test_split
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def calculate_metrics(pred, target):
    """
    计算Dice、IoU、Recall、Precision指标
    """
    pred = pred.flatten()
    target = target.flatten()
    
    # 计算基本统计量
    tp = np.sum((pred == 1) & (target == 1))  # True Positive
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

def create_full_image_transforms():
    """创建全图评估的数据变换"""
    return Compose([
        LoadImaged(keys=["image", "mask"]),
        Lambdad(keys=["image"], func=lambda x: x.mean(axis=-1, keepdims=True) if x.ndim == 3 else x),
        Lambdad(keys=["mask"], func=lambda x: x / 255.0 if x.max() > 1 else x),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityd(keys=["image"]),
        EnsureTyped(keys=["image", "mask"]),
        ToTensord(keys=["image", "mask"])
    ])

def sliding_window_inference(model, image, window_size=(256, 256), overlap=0.5):
    """滑窗推理"""
    device = image.device
    _, _, H, W = image.shape
    window_h, window_w = window_size
    
    # 计算步长
    stride_h = int(window_h * (1 - overlap))
    stride_w = int(window_w * (1 - overlap))
    
    # 计算需要的窗口数量
    num_h = (H - window_h) // stride_h + 1 if H > window_h else 1
    num_w = (W - window_w) // stride_w + 1 if W > window_w else 1
    
    # 创建输出容器
    prediction = torch.zeros_like(image)
    count_map = torch.zeros_like(image)
    
    # 滑窗推理
    for i in range(num_h):
        for j in range(num_w):
            # 计算窗口位置
            start_h = i * stride_h
            start_w = j * stride_w
            end_h = min(start_h + window_h, H)
            end_w = min(start_w + window_w, W)
            
            # 调整窗口位置确保不超出边界
            start_h = max(0, end_h - window_h)
            start_w = max(0, end_w - window_w)
            
            # 提取窗口
            window = image[:, :, start_h:end_h, start_w:end_w]
            
            # 推理
            with torch.no_grad():
                window_pred = model(window)
                if isinstance(window_pred, list):
                    window_pred = window_pred[0]
            
            # 累加预测结果
            prediction[:, :, start_h:end_h, start_w:end_w] += window_pred
            count_map[:, :, start_h:end_h, start_w:end_w] += 1
    
    # 平均化重叠区域
    prediction = prediction / (count_map + 1e-8)
    return prediction

def evaluate_model(model_path, data_loader, device, use_sliding_window=False):
    """评估模型性能"""
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
    
    method_name = "滑窗推理" if use_sliding_window else "全图推理"
    print(f"开始评估模型 ({method_name})...")
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            # 模型推理
            if use_sliding_window:
                outputs = sliding_window_inference(model, images, window_size=(256, 256), overlap=0.5)
            else:
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
            
            # 保存可视化结果
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
                
                plt.subplot(1, 4, 4)
                plt.imshow(img_np, cmap='gray')
                plt.imshow(pred_np, cmap='Reds', alpha=0.3)
                plt.imshow(label_np, cmap='Greens', alpha=0.3)
                plt.title("Overlay (Red: Pred, Green: GT)")
                plt.axis('off')
                
                method_suffix = "sliding" if use_sliding_window else "fullimage"
                plt.suptitle(f'{method_name} Sample {i+1} - Dice: {dice:.4f}, IoU: {iou:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}')
                plt.tight_layout()
                plt.savefig(f"eval_{method_suffix}_sample_{i+1}.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            print(f"Sample {i+1}/{len(data_loader)} - Dice: {dice:.4f}, IoU: {iou:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    
    # 计算统计指标
    results = {
        'dice': (np.mean(all_dice), np.std(all_dice)),
        'iou': (np.mean(all_iou), np.std(all_iou)),
        'recall': (np.mean(all_recall), np.std(all_recall)),
        'precision': (np.mean(all_precision), np.std(all_precision))
    }
    
    # 打印结果
    print("\n" + "="*60)
    print(f"评估结果汇总 ({method_name}):")
    print("="*60)
    print(f"Dice Score:    {results['dice'][0]:.4f} ± {results['dice'][1]:.4f}")
    print(f"IoU:           {results['iou'][0]:.4f} ± {results['iou'][1]:.4f}")
    print(f"Recall:        {results['recall'][0]:.4f} ± {results['recall'][1]:.4f}")
    print(f"Precision:     {results['precision'][0]:.4f} ± {results['precision'][1]:.4f}")
    print("="*60)
    
    return results, all_dice, all_iou, all_recall, all_precision

def main():
    # 参数设置
    image_dir = "dataset/images"
    mask_dir = "dataset/masks"
    model_path = "weld_seg_unetpp.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return
    
    # 数据加载
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))
    mask_paths = [os.path.join(mask_dir, os.path.basename(p).replace(".png", "_mask.png")) for p in image_paths]
    data_dicts = [{"image": img, "mask": msk} for img, msk in zip(image_paths, mask_paths)]
    
    # 数据划分
    _, val_dicts = train_test_split(data_dicts, test_size=0.2, random_state=42)
    
    # 创建全图评估数据集
    transforms = create_full_image_transforms()
    val_ds = Dataset(data=val_dicts, transform=transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=list_data_collate)
    
    print(f"验证集样本数: {len(val_dicts)}")
    
    # 全图评估
    print("\n=== 全图评估 ===")
    results_full, dice_full, iou_full, recall_full, precision_full = evaluate_model(
        model_path, val_loader, device, use_sliding_window=False
    )
    
    # 滑窗评估
    print("\n=== 滑窗评估 ===")
    results_sliding, dice_sliding, iou_sliding, recall_sliding, precision_sliding = evaluate_model(
        model_path, val_loader, device, use_sliding_window=True
    )
    
    # 保存比较结果
    with open("evaluation_comparison.txt", "w") as f:
        f.write("模型评估结果比较\n")
        f.write("="*80 + "\n\n")
        
        f.write("全图评估结果:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dice Score:    {results_full['dice'][0]:.6f} ± {results_full['dice'][1]:.6f}\n")
        f.write(f"IoU:           {results_full['iou'][0]:.6f} ± {results_full['iou'][1]:.6f}\n")
        f.write(f"Recall:        {results_full['recall'][0]:.6f} ± {results_full['recall'][1]:.6f}\n")
        f.write(f"Precision:     {results_full['precision'][0]:.6f} ± {results_full['precision'][1]:.6f}\n\n")
        
        f.write("滑窗评估结果:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Dice Score:    {results_sliding['dice'][0]:.6f} ± {results_sliding['dice'][1]:.6f}\n")
        f.write(f"IoU:           {results_sliding['iou'][0]:.6f} ± {results_sliding['iou'][1]:.6f}\n")
        f.write(f"Recall:        {results_sliding['recall'][0]:.6f} ± {results_sliding['recall'][1]:.6f}\n")
        f.write(f"Precision:     {results_sliding['precision'][0]:.6f} ± {results_sliding['precision'][1]:.6f}\n\n")
        
        f.write("详细结果:\n")
        f.write("-" * 80 + "\n")
        for i in range(len(dice_full)):
            f.write(f"Sample {i+1}:\n")
            f.write(f"  全图 - Dice: {dice_full[i]:.6f}, IoU: {iou_full[i]:.6f}, Recall: {recall_full[i]:.6f}, Precision: {precision_full[i]:.6f}\n")
            f.write(f"  滑窗 - Dice: {dice_sliding[i]:.6f}, IoU: {iou_sliding[i]:.6f}, Recall: {recall_sliding[i]:.6f}, Precision: {precision_sliding[i]:.6f}\n\n")
    
    print("\n评估完成!")
    print("结果已保存到 evaluation_comparison.txt")
    print("可视化结果已保存为 eval_fullimage_sample_*.png 和 eval_sliding_sample_*.png")

if __name__ == "__main__":
    main()