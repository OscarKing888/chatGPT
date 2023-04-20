import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import PIL.Image as Image
import os
import unicodedata
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


tensorboard_log_dir = 'runs/tensorboard_log18m'
tensorboard_log_dir_pred = 'runs/tensorboard_log18m_predition'

# 定义超参数
model_name = "ResNet18_M"
used_dataset_name = ""
dataset_config = None

learning_rate = 0.1
num_epochs = 100
batch_size = 512
momentum = 0.9
weight_decay = 5e-4


# 数据集名称映射
dataset_mapping = {
    "CIFAR10": {
        "batch_size" : 512,
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
        "batch_size" : 100,
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

def print_gpu_info():
    #print(f'GPU count: {torch.cuda.device_count()}')
    #print(f'GPU name: {torch.cuda.get_device_name(0)}')
    #print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024}MB')
    # 查看当前GPU内存的使用情况
    print(f'Memory allocated 1:{torch.cuda.memory_allocated() / 1024 / 1024}MB')
    #torch.cuda.empty_cache()
    

def generate_model_filename(dataset_name, model_name, epoch, is_best=False):
    if is_best:
        return f"{model_name}_{dataset_name}_best.pth"
    else:
        return f"{model_name}_{dataset_name}_epoch{epoch}.pth"


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock, [2, 2, 2, 2])


all_train_loss = []
all_test_loss = []
all_train_acc = []
all_test_acc = []


def plot_train_result(epoch):
    global show_plot
    global used_dataset_name
    global used_model_name

    # 绘制训练和测试损失的变化曲线
    plt.figure()
    plt.plot(all_train_loss, label='train_loss')
    plt.plot(all_test_loss, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.savefig(f"Resnet18M_train_test_loss_[{epoch}].png")
    plt.close()
    #if show_plot:    
    #    plt.show()
    

    # 绘制训练和测试准确率的变化曲线
    plt.figure()
    plt.plot(all_train_acc, label='train_acc')
    plt.plot(all_test_acc, label='test_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.savefig(f"Resnet18M_train_test_acc_[{epoch}].png")
    plt.close()
    #if show_plot:
    #    plt.show()    


# 定义训练函数
def train(epoch, model, dataloader, criterion, optimizer, scheduler, device, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    scheduler.step()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100.0 * correct / total#running_corrects.double() / len(dataloader.dataset)

    # 将训练损失和准确率写入TensorBoard
    writer.add_scalar('train_loss', epoch_loss, epoch)
    writer.add_scalar('train_acc', epoch_acc, epoch)

    #return running_loss / len(dataloader), 100.0 * correct / total
    return epoch_loss, epoch_acc


# 定义测试/验证函数
def test(epoch, model, dataloader, criterion, device, writer):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    #return running_loss / len(dataloader), 100.0 * correct / total
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100.0 * correct / total#running_corrects.double() / len(dataloader.dataset)

    # 将验证损失和准确率写入TensorBoard
    writer.add_scalar('test_loss', epoch_loss, epoch)
    writer.add_scalar('test_acc', epoch_acc, epoch)

    return epoch_loss, epoch_acc



def create_model():
    # # 准备CIFAR-10数据集
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # ])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 创建模型
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    model = ResNet18().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)

    return device, model, criterion, scheduler, optimizer


def create_dataset_loader(dataset_name, batch_size=512, num_workers=4, pin_memory=True):
    print(f"batch_size:{batch_size}, num_workers:{num_workers}, pin_memory:{pin_memory}")

    # 准备CIFAR-10数据集
    # transform_train = transforms.Compose([
    #     transforms.Resize(96),
    #     transforms.RandomCrop(96, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.Resize(96),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])

    # trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
    # trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform_test)
    # testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    global dataset_config
    #dataset_config = dataset_mapping[dataset_name]

    train_transform = dataset_config["train_transform"]
    test_transform = dataset_config["test_transform"]

    if dataset_name == "CIFAR10":
        trainset = dataset_config["loader"](root='./data', train=True, download=True, transform=train_transform)
        testset = dataset_config["loader"](root='./data', train=False, download=True, transform=test_transform)

    elif dataset_name == "STL10":        
        trainset = dataset_config["loader"](root='./data', split='train', download=True, transform=train_transform)
        testset = dataset_config["loader"](root='./data', split='test', download=True, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
    
    dataloaders = {'train': trainloader, 'test':    testloader}
    dataset_sizes = {'train': len(trainset), 'test': len(testset)}

    print(f"batch_size:{batch_size}")

    return dataloaders, dataset_sizes


# 预测函数
def predict(model, image, device):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        _, prediction = torch.max(output, 1)
        return prediction.item()

# 加载一张图片进行预测


# image_path = 'path/to/your/image.jpg'
# image = Image.open(image_path).resize((32, 32))
# image = transform_test(image).unsqueeze(0)

# # 执行预测
# prediction = predict(model, image, device)
# print(f'Predicted class: {prediction}')


# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x[:3, :, :]),  # 只保留前三个通道（RGB）
#     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
# ])


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


def predict_all_images(test_dir, model, device):
    # 支持的图像格式
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp']

    # 筛选支持的图像文件
    image_filenames = [f for f in os.listdir(test_dir) if any(f.lower().endswith(ext) for ext in supported_extensions)]

    # CIFAR-10 类别名称
    # class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    global dataset_config
    class_names = dataset_config["class_names"]
    image_size = dataset_config["image_size"]

    predictions = []

    test_transform = dataset_config["test_transform"]

    for image_filename in tqdm(image_filenames, desc="Processing images", ncols=80):
        image_path = os.path.join(test_dir, image_filename)
        image = Image.open(image_path).resize(image_size).convert('RGB')
        image = test_transform(image).unsqueeze(0).to(device)

        prediction = predict(model, image, device)
        predictions.append((image_filename, prediction, class_names[prediction]))

    # 创建子目录
    err_dir = "err"
    if not os.path.exists(err_dir):
        os.makedirs(err_dir)

    # 输出报表
    max_filename_length = 30
    print("\nPrediction Report:")
    print(f"{'Image Filename':<{max_filename_length}}{'Predicted Class':<20}{'Class Name':<20}{'Hit':<20}")

    predictions_new = []

    for prediction in predictions:
        truncated_filename = truncate_filename(prediction[0], max_filename_length)
        
        # 从文件名中查找类别名称
        true_label = "?"
        for class_name in class_names:
            if class_name.lower() in prediction[0].lower():
                true_label = class_name
                break
        
        # 将真实类别添加到预测结果中
        prediction_new = prediction + (true_label,)
        predictions_new.append(prediction_new)

        print(f"{truncated_filename:<{max_filename_length - get_display_width(truncated_filename) + len(truncated_filename)}}{prediction[1]:<20}{prediction[2]:<20}{true_label:<20}")
        
        # 如果文件名中没有找到类别名称，则将其保存到err目录
        if prediction_new[2] != prediction_new[3]:
            image_path = os.path.join(test_dir, prediction[0])
            image = Image.open(image_path).resize(image_size)
            err_image_path = os.path.join(err_dir, f"{prediction[2]}_{prediction[0]}")
            image.save(err_image_path)

    
    #print(predictions_new)
    print("\nPrediction Report of err?:")
    print(f"{'Image Filename':<{max_filename_length}}{'Predicted Class':<20}{'Class Name':<20}{'Hit':<20}")    
    for prediction_err in predictions_new:
        if prediction_err[2] != prediction_err[3]:            
            truncated_filename = truncate_filename(prediction_err[0], max_filename_length)
            print(f"{truncated_filename:<{max_filename_length - get_display_width(truncated_filename) + len(truncated_filename)}}{prediction_err[1]:<20}{prediction_err[2]:<20}{prediction_err[3]:<20}")



def train_data(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device):

   # 训练和测试
    best_acc = 0.0    
    train_results = []

    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    # 创建TensorBoard的SummaryWriter对象
    writer = SummaryWriter(tensorboard_log_dir)
    #writer = None
    print_gpu_info()
    
    #num_epochs = 5
    for epoch in tqdm(range(num_epochs), desc=f"Training", ncols=80):
        print(f"========= Train: {epoch + 1} =========")
        train_loss, train_acc = train(epoch, model, train_loader, criterion, optimizer, scheduler, device, writer)

        print(f"========= Test: {epoch + 1} =========")
        test_loss, test_acc = test(epoch, model, test_loader, criterion, device, writer)

        

        # 记录训练损失和准确率的变化情况
        all_train_loss.append(train_loss)
        all_train_acc.append(train_acc)

        # 记录测试损失和准确率的变化情况
        all_test_loss.append(test_loss)
        all_test_acc.append(test_acc)

        train_results.append((epoch + 1, train_loss, train_acc, test_loss, test_acc))

        if test_acc > best_acc:
            best_acc = test_acc
            print(f'========= New Best: {epoch + 1}, Train Loss: {test_loss:.4f}, Train Acc: {test_acc:.2f}%')
            torch.save(model.state_dict(), f'{model_name}_{used_dataset_name}_{epoch + 1}_{test_loss:.4f}_{test_acc:.2f}.pth')

            best_model_filename = generate_model_filename(used_dataset_name, model_name, epoch + 1, True)
            torch.save(model.state_dict(), best_model_filename)

        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')        
        plot_train_result(epoch)
        print_gpu_info()

    # 输出报表数据
    print("\nTraining Report:")
    print(f"{'Epoch':<10}{'Train Loss':<15}{'Train Acc':<15}{'Test Loss':<15}{'Test Acc':<15}")
    print("---------------------------------------------------------------")
    for result in train_results:
        test_acc_temp = result[4]
        mark = '*' if test_acc_temp == best_acc else ''
        print(f"{result[0]:<10}{result[1]:<15.4f}{result[2]:<15.2f}{result[3]:<15.4f}{result[4]:<15.2f}{mark}")

    # 关闭SummaryWriter对象
    writer.close()
    

def main():
    parser = argparse.ArgumentParser(description='ResNet18 for CIFAR-10 | STL10')
    parser.add_argument('--mode', default='predict', type=str, help='Mode: train or predict (default: train)')
    parser.add_argument('--dataset', default='STL10', choices=['CIFAR10', 'STL10'], help='Dataset')
    parser.add_argument('--image', default='./test', type=str, help='Path to the folder containing images for prediction')    
    
    # 检查命令行参数中是否包含-h或--help参数
    if '-h' in sys.argv or '--help' in sys.argv:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()
        
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=256"

    # 查看当前GPU内存的使用情况    
    torch.cuda.empty_cache()
    print(f'Memory allocated 1:{torch.cuda.memory_allocated() / 1024}MB')
    #print(f'Memory allocated 2:{torch.cuda.memory_allocated() / 1024}MB')

    global used_dataset_name
    used_dataset_name = args.dataset
    
    global dataset_config
    dataset_config = dataset_mapping[used_dataset_name]

    batch_size = dataset_config['batch_size']
    device, model, criterion, scheduler, optimizer = create_model()
    dataloaders, dataset_sizes = create_dataset_loader(used_dataset_name, batch_size, 2, True)

    print_gpu_info()

    if args.mode == 'predict':
        model_filename = generate_model_filename(used_dataset_name, model_name, 0, True)
        # Load the best model
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Model file not found: {model_filename}")
        
        # 加载模型
        model.load_state_dict(torch.load(model_filename))
        model.eval()

        # 预测并输出结果
        predict_all_images(args.image, model, device)        

    elif args.mode == 'train':
        train_data(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device)

if __name__ == '__main__':
    main()