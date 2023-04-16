import os
from PIL import Image

def resize_images(input_dir, size=(32, 32), suffix="32"):
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    
    # 如果不存在，则创建子目录
    output_dir = os.path.join(input_dir, "32x32")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 筛选支持的图像文件
    image_filenames = [f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in supported_extensions)]

    for image_filename in image_filenames:
        input_image_path = os.path.join(input_dir, image_filename)
        image = Image.open(input_image_path)
        
        # 调整图像大小
        resized_image = image.resize(size)

        # 在文件名中添加后缀
        output_filename = os.path.splitext(image_filename)[0] + suffix + os.path.splitext(image_filename)[1]
        output_image_path = os.path.join(output_dir, output_filename)

        # 保存调整大小的图像
        resized_image.save(output_image_path)

    print(f"Resized images have been saved to {output_dir}.")

# give me a main function

def main():
    # 从命令行读取目录
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing images to resize")
    args = parser.parse_args()

    # 调整图像大小
    resize_images(args.input_dir)


if __name__ == "__main__":
    main()