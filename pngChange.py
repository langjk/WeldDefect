import os
from PIL import Image

# 设置你的目标目录路径
directory = "./renew"  # 替换为你的实际路径

# 初始化计数器
counter = 1

for filename in os.listdir(directory):
    if filename.lower().endswith(".jpeg"):
        # 获取文件路径
        old_path = os.path.join(directory, filename)
        
        # 打开jpeg文件
        with Image.open(old_path) as img:
            # 按顺序命名新的文件
            new_filename = f"{counter}.png"
            new_path = os.path.join(directory, new_filename)
            
            # 保存为PNG格式
            img.save(new_path, "PNG")
            
            # 删除原来的jpeg文件（可选）
            os.remove(old_path)
            
            # 打印转换信息
            print(f"Converted: {filename} -> {new_filename}")
            
            # 递增计数器
            counter += 1
