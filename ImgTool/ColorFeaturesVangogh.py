import os
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import numpy as np
from sklearn.cluster import KMeans


def plot_color_blocks(hex_colors, title, algorithm):
    #plt.figure()
    fig, ax = plt.subplots()
    
    num_colors = len(hex_colors)
    width = 1 / num_colors
    
    for i, color in enumerate(hex_colors):
        rect = patches.Rectangle((i * width, 0), width, 1, linewidth=1, edgecolor='k', facecolor=color)
        ax.add_patch(rect)
        
    plt.title(f'{title}_{algorithm}')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.savefig(f'./output/{title}_{algorithm}.png')
    plt.show()
    plt.close()
    


def extract_hex_colors1(image_path, num_colors):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixels = image.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_

    hex_colors = []
    for color in colors:
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
        hex_colors.append(hex_color)
    
    return hex_colors


def extract_hex_colors_labcolor(image_path, num_colors):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    pixels = image.reshape(-1, 3)
    pixels_lab = image_lab.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(pixels_lab)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    colors = np.zeros((num_colors, 3))
    for i in range(num_colors):
        indices = np.where(labels == i)
        closest_index = indices[0][np.argmin(np.linalg.norm(pixels_lab[indices] - cluster_centers[i], axis=1))]
        colors[i] = pixels[closest_index]

    hex_colors = []
    for color in colors:
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
        hex_colors.append(hex_color)

    return hex_colors


def extract_hex_colors_hsv(image_path, num_colors):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    pixels = image.reshape(-1, 3)
    pixels_hsv = image_hsv.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, n_init=10)
    kmeans.fit(pixels_hsv)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    colors = np.zeros((num_colors, 3))
    for i in range(num_colors):
        indices = np.where(labels == i)
        closest_index = indices[0][np.argmin(np.linalg.norm(pixels_hsv[indices] - cluster_centers[i], axis=1))]
        colors[i] = pixels[closest_index]

    hex_colors = []
    for color in colors:
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
        hex_colors.append(hex_color)

    return hex_colors


if __name__ == '__main__':
    
    # 蓝色：#0b3d91（深蓝），#1e88e5（亮蓝）
    # 黄色：#fdd835
    # 白色：#ffffff
    # 橙色：#ff9f00
    # 绿色：#2e7d32
    # 黑色：#000000
    hex_colors = ['#0b3d91', '#1e88e5', '#fdd835', '#ffffff', '#ff9f00', '#2e7d32', '#000000']
    #plot_color_blocks(hex_colors)

    image_paths = ['test.png', 'VanGogh.jpg', 'VanGogh.png', 'vangogh1.png', 'vangogh2.png']
    num_colors = 10

    for image_path in image_paths:
        hex_colors = extract_hex_colors_hsv(f'./input/{image_path}', num_colors)
        print(hex_colors)
        plot_color_blocks(hex_colors, os.path.basename(image_path), 'hsv')
