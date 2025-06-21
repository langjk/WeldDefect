import numpy as np
import cv2

def generate_weld_xray(width=1000, height=1000, pipe_width=800):
    # 创建基础图像（全黑背景）
    image = np.zeros((height, width), dtype=np.uint8)
    
    # 计算管道在整个图像中的起始和结束位置（左右）
    start_x = (width - pipe_width) // 2
    end_x = start_x + pipe_width
    
    # 生成管道基本形状（两侧亮，中间暗）
    x = np.linspace(0, pipe_width-1, pipe_width)
    gradient = np.exp(-((x - pipe_width/2)**2) / (2 * (pipe_width/3)**2))  # 调整渐变范围
    gradient = 1 - gradient * 0.5  # 增加对比度
    
    # 将渐变应用到管道区域
    for i in range(height):
        image[i, start_x:end_x] = gradient * 255
    
    # 添加管道凹陷效果
    center_y = height // 2
    indent_height = 150  # 增加凹陷范围
    for y in range(height):
        factor = np.exp(-((y - center_y)**2) / (2 * (indent_height)**2))
        image[y, start_x:end_x] = image[y, start_x:end_x] * (1 - factor * 0.3)
    
    # 创建焊缝效果
    weld_mask = np.zeros_like(image)
    weld_center = center_y
    weld_width = 100
    
    # 生成基础焊缝
    for x in range(start_x, end_x):
        offset = np.random.normal(0, 2)
        y_center = weld_center + int(offset)
        # 创建不规则的焊缝形状
        height_var = np.random.randint(15, 25)
        cv2.line(weld_mask, (x, y_center-height_var), (x, y_center+height_var), 150, 1)
    
    # 添加焊缝缺陷（气孔）
    num_defects = np.random.randint(30, 50)
    for _ in range(num_defects):
        x = np.random.randint(start_x, end_x)
        y = weld_center + np.random.normal(0, 10)
        radius = np.random.randint(2, 6)
        cv2.circle(weld_mask, (int(x), int(y)), radius, 50, -1)  # 较暗的缺陷
    
    # 对焊缝进行模糊处理使其更自然
    weld_mask = cv2.GaussianBlur(weld_mask, (5, 5), 1.0)
    
    # 将焊缝与原图混合
    image = cv2.addWeighted(image, 1, weld_mask, 0.4, 0)
    
    # 添加两侧明亮的竖线效果
    edge_width = 40  # 增加边缘宽度
    edge_gradient = np.exp(-np.linspace(0, 4, edge_width)**2)
    
    # 在管道边缘添加亮线
    for i in range(edge_width):
        # 左边缘
        x_pos = start_x + i
        image[:, x_pos] = image[:, x_pos] * (1 + edge_gradient[i] * 0.8)
        
        # 右边缘
        x_pos = end_x - i - 1
        image[:, x_pos] = image[:, x_pos] * (1 + edge_gradient[i] * 0.8)
    
    # 添加细腻的噪声纹理
    # 首先添加高斯噪声
    noise1 = np.random.normal(0, 2, image.shape).astype(np.uint8)
    image = cv2.add(image, noise1)
    
    # 添加泊松噪声
    noise2 = np.random.poisson(lam=1, size=image.shape).astype(np.uint8)
    image = cv2.add(image, noise2)
    
    # 添加细小的颗粒纹理
    texture = np.zeros_like(image)
    for _ in range(1000000):  # 增加大量的小点
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        texture[y, x] = np.random.randint(0, 3)  # 很小的强度变化
    
    # 对纹理进行轻微模糊
    texture = cv2.GaussianBlur(texture, (3, 3), 0.5)
    
    # 将纹理叠加到图像上
    image = cv2.add(image, texture)
    
    # 最终的高斯模糊使整体更自然
    image = cv2.GaussianBlur(image, (3, 3), 0.5)
    
    # 确保值在有效范围内
    image = np.clip(image, 0, 255)
    
    return image.astype(np.uint8)

if __name__ == "__main__":
    # 生成图像
    result = generate_weld_xray()
    
    # 保存图像
    cv2.imwrite("simulated_weld_xray.png", result)
    
    # 显示图像
    cv2.imshow("Simulated Weld X-ray", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 