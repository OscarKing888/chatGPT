# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt

# # 读取图像
# img = Image.open('vangogh2.png')

# # 将图像转换为numpy数组
# img_data = np.array(img)

# # 提取颜色特征
# colors, counts = np.unique(img_data.reshape(-1, img_data.shape[2]), axis=0, return_counts=True)

# # 取数量最多的前5种颜色
# top_colors = colors[np.argsort(counts)[-5:]]
# top_counts = counts[np.argsort(counts)[-5:]]

# # 绘制颜色特征
# plt.figure(figsize=(8, 6))
# plt.title('Color Feature')
# plt.xlabel('Color')
# plt.ylabel('Count')
# for i in range(len(top_colors)):
#     plt.bar(i, top_counts[i], color=[tuple(c/255 for c in top_colors[i])])
# plt.xticks(range(len(top_colors)), [f"Color {i}" for i in range(len(top_colors))])
# plt.show()

import os
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def round_color(color, color_step=32):
    return tuple([min(255, ((x + color_step // 2) // color_step) * color_step) for x in color])

def get_color_distribution(image_path, resize_width=None, resize_height=None, color_step=32):
    # 打开图片
    img = Image.open(image_path)

    # 如果需要，调整图片大小
    if resize_width and resize_height:
        img = img.resize((resize_width, resize_height))

    # 获取像素数据
    pixels = img.load()

    # 统计颜色占比
    color_distribution = defaultdict(int)
    total_pixels = img.size[0] * img.size[1]
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            rounded_color = round_color(pixels[x, y], color_step)
            color_distribution[rounded_color] += 1

    # 将像素计数转换为占比
    for color in color_distribution:
        color_distribution[color] /= total_pixels

    return color_distribution

def plot_color_distribution(color_distribution, top_colors=5):
    # 只保留前top_colors种数量最多的颜色
    top_color_distribution = dict(sorted(color_distribution.items(), key=lambda x: x[1], reverse=True)[:top_colors])

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制颜色占比图
    start = 0
    for color, proportion in top_color_distribution.items():
        normalized_color = tuple([x/255 for x in color])  # 将颜色值转换为0-1范围
        ax.add_patch(plt.Rectangle((start, 0), proportion, 1, color=normalized_color))
        start += proportion

    # 隐藏坐标轴
    plt.axis('off')

    # 显示图像
    plt.show()




def plot_color_blocks(hex_colors):
    fig, ax = plt.subplots()
    
    num_colors = len(hex_colors)
    width = 1 / num_colors
    
    for i, color in enumerate(hex_colors):
        rect = patches.Rectangle((i * width, 0), width, 1, linewidth=1, edgecolor='k', facecolor=color)
        ax.add_patch(rect)
        
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.show()

# 蓝色：#0b3d91（深蓝），#1e88e5（亮蓝）
# 黄色：#fdd835
# 白色：#ffffff
# 橙色：#ff9f00
# 绿色：#2e7d32
# 黑色：#000000

hex_colors = ['#0b3d91', '#1e88e5', '#fdd835', '#ffffff', '#ff9f00', '#2e7d32', '#000000']
plot_color_blocks(hex_colors)


# def main():
    
#     image_path = "./vangogh2.png"  # 替换为你的图片路径
#     #image_path = "./test.png"  # 替换为你的图片路径
#     #resize_width, resize_height = 100, 100  # 调整图片大小以加快处理速度，可根据需要调整
#     resize_width, resize_height = None, None
#     color_distribution = get_color_distribution(image_path, resize_width, resize_height, 64)
#     plot_color_distribution(color_distribution, 10)

# if __name__ == '__main__':
#     main()

    