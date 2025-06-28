import json
import numpy as np
import cv2
from PIL import Image
import os
import glob
import re

def load_label_data(json_file):
    """加载标注数据"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def natural_sort_key(filename):
    """自然排序的键函数，正确处理数字顺序"""
    # 提取文件名中的数字和文字部分
    parts = re.split(r'(\d+)', filename)
    # 将数字部分转换为整数，文字部分保持字符串
    return [int(part) if part.isdigit() else part for part in parts]

def get_image_files():
    """获取fakeshow文件夹中的图片文件列表，按自然顺序排序"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(f"fakeshow/{ext}"))
    
    # 使用自然排序确保与JSON中的ID顺序对应
    image_files.sort(key=natural_sort_key)
    return image_files

def visualize_annotations(data, image_files, output_dir="annotated_results"):
    """可视化标注结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, item in enumerate(data):
        if i >= len(image_files):
            print(f"警告: 标注数据项 {i+1} 超过了图片文件数量")
            break
            
        image_path = image_files[i]
        
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            print(f"图片文件不存在: {image_path}")
            continue
            
        # 加载图片
        try:
            if image_path.lower().endswith('.tif'):
                # 对于TIF文件，使用cv2读取
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"无法读取图片: {image_path}")
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                # 对于其他格式，使用PIL
                image = np.array(Image.open(image_path).convert("RGB"))
        except Exception as e:
            print(f"读取图片 {image_path} 时出错: {e}")
            continue
        
        # 获取图片尺寸
        img_height, img_width = image.shape[:2]
        
        # 绘制标注框
        result_image = image.copy()
        
        # 检查是否有标注
        if 'annotations' in item and len(item['annotations']) > 0:
            annotations = item['annotations'][0]['result']  # 取第一个标注结果
            
            for annotation in annotations:
                if annotation['type'] == 'rectanglelabels':
                    # 获取边界框坐标 (百分比形式)
                    x_percent = annotation['value']['x']
                    y_percent = annotation['value']['y'] 
                    w_percent = annotation['value']['width']
                    h_percent = annotation['value']['height']
                    
                    # 转换为像素坐标
                    x = int(x_percent * img_width / 100)
                    y = int(y_percent * img_height / 100)
                    w = int(w_percent * img_width / 100)
                    h = int(h_percent * img_height / 100)
                    
                    # 绘制绿色矩形框
                    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 直接保存原图尺寸的结果
        output_filename = f"{output_dir}/annotated_{item['id']:03d}_{os.path.basename(image_path).split('.')[0]}.png"
        cv2.imwrite(output_filename, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        print(f"已处理: {image_path} -> {output_filename}")
        
        # 输出标注信息
        if 'annotations' in item and len(item['annotations']) > 0:
            annotations = item['annotations'][0]['result']
            print(f"  检测到 {len(annotations)} 个标注框")
        else:
            print("  无标注数据")

def main():
    # 加载标注数据
    try:
        label_data = load_label_data('label2.json')
        print(f"加载了 {len(label_data)} 个标注项")
    except Exception as e:
        print(f"加载标注文件时出错: {e}")
        return
    
    # 获取图片文件列表
    image_files = get_image_files()
    print(f"找到 {len(image_files)} 个图片文件")
    
    if len(image_files) == 0:
        print("未找到图片文件!")
        return
    
    # 显示前几个文件的对应关系
    print("\n文件对应关系:")
    for i in range(min(5, len(label_data), len(image_files))):
        print(f"  ID {label_data[i]['id']}: {image_files[i]}")
    
    # 可视化标注
    visualize_annotations(label_data, image_files)

if __name__ == "__main__":
    main()