import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import unicodedata
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import time
from prettytable import PrettyTable
from PIL import Image

# 数据集名称映射
dataset_mapping = {
    "CIFAR10": {
        "class_names": ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        "train_transform": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        "test_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        "loader": datasets.CIFAR10,
    },
    "STL10": {
        "class_names": ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'),
        "train_transform": transforms.Compose([
            transforms.Resize(96),
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        "test_transform": transforms.Compose([
            transforms.Resize(96),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]),
        "loader": datasets.STL10,
    },
}

# 模型名称映射
model_mapping = {
    "ResNet18": models.resnet18,
    "ResNet50": models.resnet50,
    "ResNet101": models.resnet101,
}

def create_model(model_name, num_classes):
    if model_name in model_mapping:
        model = model_mapping[model_name](pretrained=False)
    else:
        raise ValueError("Invalid model name. Choose from: " + ", ".join(model_mapping.keys()))

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 参数梯度置零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 深度复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


def get_display_width(s):
    width = 0
    for c in s:
        if unicodedata.east_asian_width(c) in ['F', 'W', 'A']:
            width += 2
        else:
            width += 1
    return width


def truncate_filename(filename, max_width):
    display_width = get_display_width(filename)
    if display_width > max_width:
        truncated = ""
        width = 0
        for c in filename:
            c_width = 2 if unicodedata.east_asian_width(c) in ['F', 'W', 'A'] else 1
            if width + c_width > max_width - 3:
                break
            truncated += c
            width += c_width
        return truncated + "..."
    return filename



def save_resized_image(input_path, output_path, size):
    image = Image.open(input_path)
    resized_image = image.resize(size, Image.ANTIALIAS)
    resized_image.save(output_path)


def predict_all_images(image_folder, model, device, class_names, test_transform):
    model.eval()
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_filenames = [f for f in os.listdir(image_folder) if f.lower().endswith(supported_extensions)]

    # Prepare to store resized error images
    error_folder = os.path.join(os.path.dirname(image_folder), "err")
    os.makedirs(error_folder, exist_ok=True)

    # Initialize progress bar
    progress_bar = tqdm(image_filenames, desc="Predicting", ncols=100)

    # Prepare results table
    table = PrettyTable(["File", "Predicted Class", "Class Name"])

    # Process each image
    for image_filename in progress_bar:
        file_path = os.path.join(image_folder, image_filename)

        # Convert the image to RGB
        image = Image.open(file_path).convert("RGB")
        
        image_tensor = test_transform(image).unsqueeze(0).to(device)


        # Predict the class
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)

        predicted_class_name = class_names[predicted_class.item()]

        # Extract correct class from file name if available
        correct_class = "?"
        for name in class_names:
            if name.lower() in image_filename.lower():
                correct_class = name
                break

        # Save error images as resized 32x32 versions
        if predicted_class_name.lower() != correct_class.lower():
            save_resized_image(file_path, os.path.join(error_folder, f"{image_filename[:-4]}_32{image_filename[-4:]}"), (32, 32))

        # Update the results table
        table.add_row([image_filename, predicted_class.item(), predicted_class_name])

    # Print the results table
    print(table)


def generate_model_filename(dataset_name, model_name):
    return f"{model_name}_{dataset_name}_best.pth"

def main():
    parser = argparse.ArgumentParser(description='PyTorch ResNet Training and Prediction')
    parser.add_argument('--mode', default='train', type=str, help='Mode: train or predict (default: train)')
    parser.add_argument('--dataset', default='STL10', choices=['CIFAR10', 'STL10'], help='Dataset')
    parser.add_argument('--model', default='ResNet101', choices=['ResNet18', 'ResNet50', 'ResNet101'], help='Model')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--image', default='./test', type=str, help='Path to the folder containing images for prediction')
    parser.add_argument('--resize', default=False, action='store_true', help='Resize images in folder to 32x32')
    args = parser.parse_args()

    # 打印当前命令行参数值
    print("Command line arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("\n")

    # Select dataset
    dataset_config = dataset_mapping[args.dataset]
    train_transform = dataset_config["train_transform"]
    test_transform = dataset_config["test_transform"]
    class_names = dataset_config["class_names"]

    # Prepare data
    trainset = dataset_config["loader"](root='./data', split='train', download=True, transform=train_transform)
    testset = dataset_config["loader"](root='./data', split='test', download=True, transform=test_transform)
    #trainset = dataset_config["loader"](root='./data', train=True, download=True, transform=train_transform)
    #testset = dataset_config["loader"](root='./data', train=False, download=True, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)    
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    dataloaders = {'train': trainloader, 'val':    testloader}
    dataset_sizes = {'train': len(trainset), 'val': len(testset)}

    # Create model
    model = create_model(args.model, num_classes=len(class_names))

    # Check if GPU is available
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    model_filename = generate_model_filename(args.dataset, args.model)

    print(f"Mode: {'Training' if args.mode == 'train' else 'Predicting'}, "
        f"Dataset: {args.dataset}, Model: {args.model}, "
        f"Model file: {model_filename}")


    if args.mode == 'train':
        # Train the model
        best_model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=args.epochs)

        # Save the best model
        torch.save(best_model.state_dict(), model_filename)
    elif args.mode == 'predict':
        # Load the best model
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file not found: {model_filename}")
        
        model.load_state_dict(torch.load(model_filename))

        # Predict for images in folder
        predict_all_images(args.image, model, device, class_names, test_transform)
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'predict'.")



if __name__ == '__main__':
    # 在训练或预测开始前
    start_time = time.time()
    main()

    # 在训练或预测结束后
    end_time = time.time()

    # 计算并输出所用时间
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")




