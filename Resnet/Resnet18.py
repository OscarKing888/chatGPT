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

# 定义训练函数
def train(model, dataloader, criterion, optimizer, device):
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

    return running_loss / len(dataloader), 100.0 * correct / total

# 定义测试/验证函数
def test(model, dataloader, criterion, device):
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

    return running_loss / len(dataloader), 100.0 * correct / total

# 设置超参数
learning_rate = 0.1
num_epochs = 100
batch_size = 512
momentum = 0.9
weight_decay = 5e-4

# 准备CIFAR-10数据集
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# 创建模型
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

model = ResNet18().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

# # 训练和测试
# for epoch in range(num_epochs):
#     train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
#     test_loss, test_acc = test(model, testloader, criterion, device)

#     print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# # 保存模型
# torch.save(model.state_dict(), 'resnet18_cifar10.pth')

# # 加载模型
# model.load_state_dict(torch.load('resnet18_cifar10.pth'))
# model.eval()

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


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :]),  # 只保留前三个通道（RGB）
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])


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
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    predictions = []

    for image_filename in tqdm(image_filenames, desc="Processing images", ncols=80):
        image_path = os.path.join(test_dir, image_filename)
        image = Image.open(image_path).resize((32, 32)).convert('RGBA')
        image = transform_test(image).unsqueeze(0)

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
            image = Image.open(image_path).resize((32, 32))
            err_image_path = os.path.join(err_dir, f"{prediction[2]}_{prediction[0]}")
            image.save(err_image_path)

    
    #print(predictions_new)
    print("\nPrediction Report of err?:")
    print(f"{'Image Filename':<{max_filename_length}}{'Predicted Class':<20}{'Class Name':<20}{'Hit':<20}")    
    for prediction_err in predictions_new:
        if prediction_err[2] != prediction_err[3]:            
            truncated_filename = truncate_filename(prediction_err[0], max_filename_length)
            print(f"{truncated_filename:<{max_filename_length - get_display_width(truncated_filename) + len(truncated_filename)}}{prediction_err[1]:<20}{prediction_err[2]:<20}{prediction_err[3]:<20}")



def train_data():
   # 训练和测试
    best_acc = 0.0    
    train_results = []

    #num_epochs = 5
    for epoch in tqdm(range(num_epochs), desc=f"Training", ncols=80):
        print(f"========= Train: {epoch + 1} =========")
        train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
        print(f"========= Test: {epoch + 1} =========")
        test_loss, test_acc = test(model, testloader, criterion, device)

        train_results.append((epoch + 1, train_loss, train_acc, test_loss, test_acc))

        if test_acc > best_acc:
            best_acc = test_acc
            print(f'========= New Best: {epoch + 1}, Train Loss: {test_loss:.4f}, Train Acc: {test_acc:.2f}%')
            torch.save(model.state_dict(), f'resnet18_cifar10_{epoch + 1}_{test_loss:.4f}_{test_acc:.2f}.pth')
            torch.save(model.state_dict(), 'resnet18_cifar10_bestm.pth')

        print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # 输出报表数据
    print("\nTraining Report:")
    print(f"{'Epoch':<10}{'Train Loss':<15}{'Train Acc':<15}{'Test Loss':<15}{'Test Acc':<15}")
    print("---------------------------------------------------------------")
    for result in train_results:
        test_acc_temp = result[4]
        mark = '*' if test_acc_temp == best_acc else ''
        print(f"{result[0]:<10}{result[1]:<15.4f}{result[2]:<15.2f}{result[3]:<15.4f}{result[4]:<15.2f}{mark}")


def main():
    parser = argparse.ArgumentParser(description='ResNet18 for CIFAR-10')
    parser.add_argument('--image', type=str, help='Path to the folder containing images for prediction')
    args = parser.parse_args()

    if args.image:
        # 加载模型
        model.load_state_dict(torch.load('resnet18_cifar10_bestm.pth'))
        model.eval()

        # 预测并输出结果
        predict_all_images(args.image, model, device)        

        # # 加载模型
        # model.load_state_dict(torch.load('resnet18_cifar10.pth'))
        # model.eval()

        # # 加载一张图片进行预测
        # image_path = args.image
        # image = Image.open(image_path).resize((32, 32))
        # image = transform_test(image).unsqueeze(0)

        # # 执行预测
        # prediction = predict(model, image, device)
        # print(f'Predicted class: {prediction}')
    else:
        train_data()
        # # 训练和测试
        # for epoch in range(num_epochs):
        #     train_loss, train_acc = train(model, trainloader, criterion, optimizer, device)
        #     test_loss, test_acc = test(model, testloader, criterion, device)

        #     print(f'Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        # # 保存模型
        # torch.save(model.state_dict(), 'resnet18_cifar10.pth')

if __name__ == '__main__':
    main()