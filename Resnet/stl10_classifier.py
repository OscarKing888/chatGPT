import os
import argparse
from pathlib import Path

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision.models import resnet18
import PIL.Image as Image
from tqdm import tqdm
import unicodedata
import matplotlib.pyplot as plt

all_train_loss = []
all_test_loss = []
all_train_acc = []
all_test_acc = []

def load_data(batch_size, data_dir='./data'):
    transform = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = STL10(root=data_dir, split='train', download=True, transform=transform)
    test_set = STL10(root=data_dir, split='test', download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def create_model():
    model = resnet18(weights=None, num_classes=10)
    return model

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    # 将验证损失和准确率写入TensorBoard
    #writer.add_scalar('test_loss', epoch_loss, epoch)
    #writer.add_scalar('test_acc', epoch_acc, epoch)

    return epoch_loss, epoch_acc


def test(model, test_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def plot_train_result(epoch):
    # 绘制训练和测试损失的变化曲线
    plt.figure()
    plt.plot(all_train_loss, label='train_loss')
    plt.plot(all_test_loss, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.savefig(f"stl10_loss_[{epoch}].png")
    plt.close()
    

    # 绘制训练和测试准确率的变化曲线
    plt.figure()
    plt.plot(all_train_acc, label='train_acc')
    plt.plot(all_test_acc, label='test_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.savefig(f"stl10_acc_[{epoch}].png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="STL10 Classifier")
    parser.add_argument('--mode', default='train', choices=['train', 'predict'], help="Mode: train or predict")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training/testing")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train")
    parser.add_argument('--input_dir', type=str, help="Directory containing images for prediction")
    parser.add_argument('--model_path', type=str, default='./stl10_model.pth', help="Path to save/load model")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    if args.mode == 'train':
        train_loader, test_loader = load_data(args.batch_size)
        
        best_accuracy = 0.0

        for epoch in tqdm(range(args.epochs), desc=f"Training", ncols=80):            
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc = test(model, test_loader, device)
            scheduler.step()

            # 记录训练损失和准确率的变化情况
            all_train_loss.append(train_loss)
            all_train_acc.append(train_acc)

            # 记录测试损失和准确率的变化情况
            all_test_loss.append(test_loss)
            all_test_acc.append(test_acc)

            print(f'Epoch {epoch + 1}/{args.epochs}, Loss: {train_loss:.4f}, Accuracy: {test_acc:.2f}%')
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(model.state_dict(), args.model_path)
                print(f'New best model saved at {args.model_path}')        

        plot_train_result(args.epochs)        

    elif args.mode == 'predict':
        if not args.input_dir:
            print("Please provide an input directory for prediction.")
            exit(1)

        model.load_state_dict(torch.load(args.model_path))
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(96),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        input_dir = Path(args.input_dir)
        for img_path in input_dir.glob("*.jpg"):
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            _, predicted = torch.max(output.data, 1)
            print(f"Image: {img_path}, Prediction: {predicted.item()}")

if __name__ == "__main__":
    main()
