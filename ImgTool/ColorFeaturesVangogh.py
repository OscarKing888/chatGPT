import os
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import numpy as np
from sklearn.cluster import KMeans
import colorsys


def plot_color_blocks(hex_colors, title, algorithm):
    #plt.figure()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    num_colors = len(hex_colors)
    width = 1 / num_colors
    height = 1 / num_colors
    
    for i, color in enumerate(hex_colors):
        rect = patches.Rectangle((i, 0), 1, 1, color=color)
        #rect = patches.Rectangle((i * width, 0), width, 1, linewidth=1, edgecolor='k', facecolor=color)
        ax.add_patch(rect)
        ax.text(i + 0.5, 0, str(i), ha='center', va='bottom', fontsize=10)
        
    plt.title(f'{title}_{algorithm}')
    plt.xlim(0, num_colors)
    plt.ylim(0, 1)
    plt.xticks(range(num_colors))
    plt.yticks([])
    #plt.axis('off')
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
    # for color in colors:
    #     hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
    #     hex_colors.append(hex_color)

    hex_colors = sort_colors_by_brightness(colors)
    return hex_colors


def sort_colors_by_brightness(colors):
    hsv_colors = []
    for color in colors:
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        hsv_color = colorsys.rgb_to_hsv(r, g, b)
        hsv_colors.append((hsv_color, color))

    hsv_colors.sort(key=lambda x: x[0][2])
    sorted_colors = [x[1] for x in hsv_colors]

    sorted_hex_colors = []
    for color in sorted_colors:
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
        sorted_hex_colors.append(hex_color)

    return sorted_hex_colors


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
    # for color in colors:
    #     hex_color = '#{:02x}{:02x}{:02x}'.format(int(color[0]), int(color[1]), int(color[2]))
    #     hex_colors.append(hex_color)
    
    hex_colors = sort_colors_by_brightness(colors)

    return hex_colors


def hex_to_rgb(hex_colors):
    rgb_colors = []
    for hex_color in hex_colors:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        rgb_colors.append((r, g, b))
    return rgb_colors


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

    hex_colors = hex_to_rgb(hex_colors)
    hex_colors = sort_colors_by_brightness(hex_colors)
    print(f'chatGPT:{hex_colors}')
    plot_color_blocks(hex_colors, 'chatGPT', 'hsv')

    for image_path in image_paths:
        hex_colors = extract_hex_colors_hsv(f'./input/{image_path}', num_colors)
        print(f'{image_path}:{hex_colors}')
        plot_color_blocks(hex_colors, os.path.basename(image_path), 'hsv')

#%APPDATA%\Microsoft\Templates\Document Themes\Theme Colors