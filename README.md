# 焊缝气孔缺陷检测系统

基于深度学习的焊缝X射线图像气孔缺陷检测项目，使用UNet和UNet++模型进行语义分割。

## 项目概述

本项目旨在通过深度学习技术自动检测焊缝X射线图像中的气孔缺陷，提高焊接质量检测的效率和准确性。系统采用语义分割方法，能够精确定位和分割气孔缺陷区域。

## 主要功能

- **多模型支持**: 实现UNet和UNet++两种分割网络
- **数据增强**: 包含几何变换、亮度调整、噪声添加等多种增强策略
- **完整训练流程**: 支持模型训练、验证和性能评估
- **实时推理**: 提供单张图像推理功能
- **可视化输出**: 生成训练过程可视化和预测结果叠加图

## 项目结构

```
WeldDefect/
├── model.py              # UNet模型定义
├── dataset.py            # 数据加载和预处理
├── train.py              # 模型训练脚本
├── losses.py             # 损失函数定义
├── inference.py          # 模型推理脚本
├── evaluate.py           # 模型评估脚本
├── evaluate_fullimage.py # 完整图像评估
├── generate_weld_xray.py # 合成X射线图像生成
├── dataset/              # 数据集目录
│   ├── images/           # 原始图像
│   ├── masks/            # 标注掩码
│   └── annotations/      # 标注文件
└── weld_seg_*.pth        # 训练好的模型权重
```

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- MONAI 1.0+
- OpenCV
- scikit-learn
- matplotlib
- numpy

## 安装依赖

```bash
pip install torch torchvision
pip install monai[all]
pip install opencv-python scikit-learn matplotlib
```

## 数据准备

1. 将原始X射线图像放入 `dataset/images/` 目录
2. 将对应的掩码图像放入 `dataset/masks/` 目录
3. 掩码文件命名格式：`{image_name}_mask.png`

## 使用方法

### 训练模型

```bash
python train.py
```

训练参数：
- 图像尺寸: 512x512
- 批次大小: 1
- 学习率: 1e-4
- 训练轮次: 50

### 模型推理

```bash
python inference.py
```

推理脚本会：
- 加载预训练模型 `weld_seg_unet-Cut.pth`
- 对指定图像进行预测
- 应用形态学后处理去除噪声
- 显示原图和预测结果的叠加效果

### 模型评估

```bash
python evaluate.py
```

评估指标包括：
- Dice Score
- IoU (Intersection over Union)  
- Recall (召回率)
- Precision (精确率)

## 模型架构

### UNet
- 编码器-解码器结构
- 跳跃连接保留细节信息
- 特征通道：(32, 64, 128, 256, 512)

### UNet++
- 嵌套U形结构
- 密集跳跃连接
- 特征通道：(16, 32, 64, 128, 256, 512)

## 损失函数

采用组合损失函数：
- **Dice Loss**: 处理类别不平衡
- **Tversky Loss**: 调节假正例和假负例权重 (α=0.7, β=0.3)

## 数据增强策略

- 随机翻转和旋转
- 缩放变换 (0.9-1.1倍)
- 对比度调整
- 高斯噪声和平滑
- 网格变形
- 强度偏移和偏置场

## 性能指标

当前模型在验证集上的表现：
- Dice Score: 0.181 ± 0.152
- IoU: 0.107 ± 0.094
- Recall: 0.165 ± 0.160
- Precision: 0.304 ± 0.314

## 后处理

预测结果经过以下后处理步骤：
1. Sigmoid激活和阈值化 (阈值=0.9)
2. 形态学开运算去除小噪声
3. 形态学闭运算填充空洞
4. 连通域分析过滤小区域 (面积<20像素)

## 文件说明

- `debug_loader.py`: 数据加载器调试工具
- `pngChange.py`: 图像格式转换工具
- `pngHandle.py`: 图像预处理工具
- `inferenceRe.py`: 重构版推理脚本

## 注意事项

1. 模型在训练时使用了随机裁剪策略，推理时需要相应的预处理
2. 图像需要转换为灰度格式并进行强度归一化
3. 模型权重文件较大，需要确保足够的存储空间
4. 建议使用GPU进行训练和推理以提高效率

## 参考资料

项目基于以下技术栈开发：
- [MONAI](https://monai.io/): 医学影像深度学习框架
- [PyTorch](https://pytorch.org/): 深度学习框架
- [UNet](https://arxiv.org/abs/1505.04597): 语义分割网络
- [UNet++](https://arxiv.org/abs/1807.10165): 改进的UNet架构