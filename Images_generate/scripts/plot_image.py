import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# 配置参数
num_iterations = 5
steps_per_iteration = 8
image_dir = '../images/derma/process'  # 图像文件存放路径
output_image = '../images/derma/process/evolution_grid.png'  # 最终输出的合成图像路径
target_size = 32  # 目标尺寸

# 定义网格的大小
rows = num_iterations
cols = steps_per_iteration

# 创建一个空的画布，调整figsize使图像紧密排列
fig, axes = plt.subplots(rows, cols, figsize=(cols, rows))

# 函数：将图像填充到指定大小
def pad_to_size(image, target_size):
    h, w = image.shape[:2]
    top = (target_size - h) // 2
    bottom = target_size - h - top
    left = (target_size - w) // 2
    right = target_size - w - left
    color = [0, 0, 0]  # 黑色填充
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

# 加载并显示图像
for i in range(num_iterations):
    for j in range(steps_per_iteration):
        image_path = os.path.join(image_dir, f'derma_50k_test_image_iteration_{i + 1}_step_{j + 1}.png')
        img = cv2.imread(image_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
            img_padded = pad_to_size(img_rgb, target_size)  # 填充到指定大小
            axes[i, j].imshow(img_padded)
        axes[i, j].axis('off')  # 关闭坐标轴

# 去掉子图之间的空白
plt.subplots_adjust(wspace=0, hspace=0)

# 保存最终的合成图像
plt.savefig(output_image, bbox_inches='tight', pad_inches=0)
plt.show()
