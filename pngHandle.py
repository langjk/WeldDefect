import os
import re

# 设置你的目标目录路径
directory = "./dataset/masks"  # 替换为你的实际路径

for filename in os.listdir(directory):
    match = re.match(r"task-(\d+)-.+", filename)
    if match:
        number = match.group(1)
        new_name = f"{number}_mask"
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_name)
        
        # 如果有扩展名，保留原扩展名
        if '.' in filename:
            ext = os.path.splitext(filename)[1]
            new_path += ext
        
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {os.path.basename(new_path)}")
