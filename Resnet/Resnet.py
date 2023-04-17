import os,sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import time
from prettytable import PrettyTable
import unicodedata
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import webbrowser
import matplotlib.pyplot as plt

tensorboard_log_dir = 'runs/tensorboard_log'
tensorboard_log_dir_pred = 'runs/tensorboard_log_predition'
max_filename_length = 30
used_model_name = ""
used_dataset_name = ""
show_plot = False

# 数据集名称映射
dataset_mapping = {
    "CIFAR10": {
        "class_names": ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        "image_size": (32, 32),
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
        "image_size": (96, 96),
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


all_train_loss = []
all_test_loss = []
all_train_acc = []
all_test_acc = []

def plot_train_result():
    global show_plot
    global used_dataset_name
    global used_model_name

    # 绘制训练和测试损失的变化曲线
    plt.plot(all_train_loss, label='train_loss')
    plt.plot(all_test_loss, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.figure()
    plt.savefig(f"{used_model_name}_{used_dataset_name}_train_test_loss.png")
    if show_plot:    
        plt.show()
    

    # 绘制训练和测试准确率的变化曲线
    plt.plot(all_train_acc, label='train_acc')
    plt.plot(all_test_acc, label='test_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.figure()
    plt.savefig(f"{used_model_name}_{used_dataset_name}_train_test_acc.png")
    if show_plot:
        plt.show()    


def train(model, dataloader, criterion, optimizer, scheduler, device, writer, epoch):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    running_corrects = 0

    # 迭代数据
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 参数梯度置零
        optimizer.zero_grad()

        # 前向传播
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    scheduler.step()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    # 记录训练损失和准确率的变化情况
    all_train_loss.append(epoch_loss)
    all_train_acc.append(epoch_acc.item())

    # 将训练损失和准确率写入TensorBoard
    writer.add_scalar('train_loss', epoch_loss, epoch)
    writer.add_scalar('train_acc', epoch_acc, epoch)

    return epoch_loss, epoch_acc



def test(model, dataloader, criterion, device, writer, epoch):
    model.eval()   # 设置模型为评估模式
    running_loss = 0.0
    running_corrects = 0

    # 迭代数据
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 前向传播
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # 统计
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    # 记录测试损失和准确率的变化情况
    all_test_loss.append(epoch_loss)
    all_test_acc.append(epoch_acc.item())

    # 将验证损失和准确率写入TensorBoard
    writer.add_scalar('test_loss', epoch_loss, epoch)
    writer.add_scalar('test_acc', epoch_acc, epoch)

    return epoch_loss, epoch_acc


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=25, batch_size=100, test_size=0.2):
    since = time.time()

    # 划分训练集和验证集
    #train_data, val_data = train_test_split(dataset, test_size=test_size)    

    # 创建数据加载器
    #train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    #val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    train_loader = dataloaders['train']
    val_loader = dataloaders['test']

    best_model_wts = model.state_dict()
    best_acc = 0.0

    # 创建TensorBoard的SummaryWriter对象
    writer = SummaryWriter(tensorboard_log_dir)

    stat_table = PrettyTable(["Epoch", "Train Loss", "Train Acc", "Test Loss", "Test ACC"])

    for epoch in range(num_epochs):
        epoch_str = 'Epoch {}/{}'.format(epoch + 1, num_epochs)
        print(epoch_str)
        print('-' * 10)

        # 训练和验证
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device, writer, epoch)
        test_loss, test_acc = test(model, val_loader, criterion, device, writer, epoch)

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_loss, train_acc))
        print('Test Loss: {:.4f} Acc: {:.4f}'.format(test_loss, test_acc))

        stat_table.add_row([epoch + 1, train_loss, train_acc.item(), test_loss, test_acc.item()])

        # 深度复制模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 关闭SummaryWriter对象
    writer.close()
    print(stat_table)

    return model


def save_resized_image(input_path, output_path, size):
    image = Image.open(input_path)
    resized_image = image.resize(size, Image.ANTIALIAS)
    resized_image.save(output_path)



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


def predict_all_images(image_folder, model, device, class_names, test_transform, img_size=(32, 32)):
    model.eval()
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_filenames = [f for f in os.listdir(image_folder) if f.lower().endswith(supported_extensions)]

    global used_dataset_name
    global used_model_name
    
    # Prepare to store resized error images
    print(f"used_model_name:{used_model_name} used_dataset_name:{used_dataset_name}")
    
    error_folder = os.path.join(os.path.dirname(image_folder), f"err_{used_model_name}_{used_dataset_name}")
    os.makedirs(error_folder, exist_ok=True)

    # Initialize progress bar
    progress_bar = tqdm(image_filenames, desc="Predicting", ncols=100)

    # Prepare results table
    table = PrettyTable(["Image Name", "Predicted Class", "Class Name", "Confidence"])

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(tensorboard_log_dir_pred)
    
    # Process each image
    for idx, image_filename in enumerate(progress_bar):
        file_path = os.path.join(image_folder, image_filename)
        # Convert the image to RGB
        image = Image.open(file_path).convert("RGB")        
        image_tensor = test_transform(image).unsqueeze(0).to(device)

        # Predict the class
        with torch.no_grad():
            output = model(image_tensor)
            prob, predicted_class = torch.max(nn.functional.softmax(output, dim=1), 1)

        predicted_class_idx = predicted_class.item()
        predicted_class_name = class_names[predicted_class_idx]

        # 检查类名是否在文件名中
        confidence = "✓" if predicted_class_name.lower() in image_filename.lower() else "?"

        # 将数据添加到表格中
        table.add_row([truncate_filename(image_filename, max_filename_length), predicted_class_idx, predicted_class_name, confidence])

        # Save error images as resized versions
        if confidence == "?":
            save_resized_image(file_path, os.path.join(error_folder, f"{predicted_class_name}_{image_filename}"), img_size)

        # Write to TensorBoard
        writer.add_scalar('Prediction/class_index', predicted_class_idx, idx)
        writer.add_scalar('Prediction/probability', prob.item(), idx)

       
    # Close the TensorBoard SummaryWriter
    writer.close()

    # Print the results table
    print(table)


def generate_model_filename(dataset_name, model_name):
    return f"{model_name}_{dataset_name}_best.pth"

def main():

    sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description='PyTorch ResNet Training and Prediction')
    parser.add_argument('--mode', default='train', type=str, help='Mode: train or predict (default: train)')
    parser.add_argument('--dataset', default='STL10', choices=['CIFAR10', 'STL10'], help='Dataset')
    parser.add_argument('--model', default='ResNet101', choices=['ResNet18', 'ResNet50', 'ResNet101'], help='Model')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='Weight decay')
    #parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--image', default='./test', type=str, help='Path to the folder containing images for prediction')    
    parser.add_argument('--showplot', default=False, action='store_true', help='Show plot of training and validation loss')

    # 检查命令行参数中是否包含-h或--help参数
    if '-h' in sys.argv or '--help' in sys.argv:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()

    # 打印当前命令行参数值
    print("Command line arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("\n")

    webbrowser.open_new_tab('http://localhost:6006/')
    
    global used_dataset_name
    global used_model_name

    used_dataset_name = args.dataset
    used_model_name = args.model

    # Select dataset
    dataset_config = dataset_mapping[args.dataset]
    train_transform = dataset_config["train_transform"]
    test_transform = dataset_config["test_transform"]
    class_names = dataset_config["class_names"]
    image_size = dataset_config["image_size"]

    trainset = []
    testset = []
    trainloader = ()
    testloader = ()

    if args.dataset == "CIFAR10":
        trainset = dataset_config["loader"](root='./data', train=True, download=True, transform=train_transform)
        testset = dataset_config["loader"](root='./data', train=False, download=True, transform=test_transform)

    elif args.dataset == "STL10":        
        trainset = dataset_config["loader"](root='./data', split='train', download=True, transform=train_transform)
        testset = dataset_config["loader"](root='./data', split='test', download=True, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    dataloaders = {'train': trainloader, 'test':    testloader}
    dataset_sizes = {'train': len(trainset), 'test': len(testset)}

    # Create model
    model = create_model(args.model, num_classes=len(class_names))

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda")
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

        plot_train_result()

    elif args.mode == 'predict':
        # Load the best model
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file not found: {model_filename}")
        
        model.load_state_dict(torch.load(model_filename))

        # Predict for images in folder
        predict_all_images(args.image, model, device, class_names, test_transform, image_size)
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'predict'.")



if __name__ == '__main__':
    print("Python version: ", sys.version)
    # 在训练或预测开始前
    start_time = time.time()
    main()

    # 在训练或预测结束后
    end_time = time.time()

    # 计算并输出所用时间
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds")