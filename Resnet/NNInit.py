import os
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image
from tabulate import tabulate
import torchvision.transforms as transforms
from torch import nn

nn_logs_folder = './logs/'
nn_pth_folder = './pth/'
nn_plot_folder = './plots/'
nn_err_img_folder = './err/'


def nn_init():
    folders = [nn_logs_folder, nn_pth_folder, nn_plot_folder, nn_err_img_folder]    
    #if path not exist then create it
    for folder in folders:
        print(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)


def nn_args(description='Image Classifier'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--mode', default='train', choices=['train', 'predict'], help="Mode: train or predict")
    parser.add_argument('--dataset', default='STL10', choices=['CIFAR10', 'STL10'], help='Dataset')
    parser.add_argument('--batchsize', type=int, default=32, help="Batch size for training/testing")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train")
    parser.add_argument('--inputdir', type=str, default='./test', help="Directory containing images for prediction")    
    parser.add_argument('--model_file', type=str, default='best_model.pth', help="File to save/load model")
    parser.add_argument('--scheduler', default=True, action='store_true', help='Use or not use scheduler (default: True)')
    return parser

def nn_print_args(args):
    print(tabulate(vars(args).items(), ['Arg', 'Value'], tablefmt="grid"))


def nn_get_logs_path(file_name):
    return os.path.join(nn_logs_folder, file_name)


def nn_get_pth_path(file_name):
    return os.path.join(nn_pth_folder, file_name)


def nn_get_plot_path(file_name):
    return os.path.join(nn_plot_folder, file_name)


def nn_get_err_img_path(file_name):
    return os.path.join(nn_err_img_folder, file_name)


def nn_plot_result(file_name, data1, data2, data1_label='train_acc', data2_label='test_acc', title='Training and Testing Accuracy', x_label='Epoch', y_label='Accuracy', show=False):
    plt.figure()
    
    #plt.annotate(title[0], xy=(0.5, title[1]), xycoords='axes fraction', fontsize=12, ha='center', va='center', alpha=0.5)

    plt.plot(data1, label=data1_label)
    plt.plot(data2, label=data2_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    save_path = nn_get_plot_path(file_name)
    plt.savefig(save_path)
    plt.close()
    if show:
        plt.show()
    print("Plot saved to: ", save_path)


def nn_print_gpu_info():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    #print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024}MB')
    # 查看当前GPU内存的使用情况
    print(f'Memory allocated 1:{torch.cuda.memory_allocated() / 1024 / 1024}MB')
    #torch.cuda.empty_cache()


def nn_print_hyperparameters():
    nn_print_gpu_info()
    for name, obj in vars(torch.optim).items():
        if isinstance(obj, type) and issubclass(obj, torch.optim.Optimizer):
            if hasattr(obj, 'defaults'):
                print(f"Optimizer: {name}")
                for key, value in obj.defaults.items():
                    print(f"\t{key}: {value}")



def nn_save_image_as(src_path, dest_file_name, image_size=(96, 96)):
    image = Image.open(src_path).resize(image_size)
    save_path = nn_get_err_img_path(dest_file_name)    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    print("Image saved to: ", dest_file_name)


def nn_print_table(table, headers, title=None):
    if title:
        print('-' * len(title))
        print(title)
        print('-' * len(title))

    print(tabulate(table, headers, tablefmt="grid"))


def nn_save_model(model, file_name):    
    save_path = nn_get_pth_path(file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print("Model saved to: ", save_path)


def nn_load_model(model, file_name):
    load_path = nn_get_pth_path(file_name)
    model.load_state_dict(torch.load(load_path))
    print("Model loaded from: ", load_path)



def nn_extract_dataset_images(dataset, target_dir, prefix):
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍历数据集中的所有图像
    for i, (image, label) in enumerate(dataset):        
        # 将张量转换为PIL图像对象
        image = transforms.ToPILImage()(image)

        # 构造文件名
        filename = f"{prefix}_[{label}]{i+1:05d}.png"
        filepath = os.path.join(target_dir, filename)
        
        # 将图像保存到文件
        image.save(filepath)


def nn_print_model_summary(model, show_hidden_layers=False):
    # 遍历模型的每一层，并打印每一层的名称和输出形状
    layers = []
    idx = 0
    idx_all = 0
    for name, module in model.named_modules():
        layer_type = type(module).__name__
        if isinstance(module, torch.nn.Conv2d):
            out_channels = module.out_channels
            in_channels = module.in_channels
            kernel_size = module.kernel_size
            layer_neurons = out_channels * kernel_size[0] * kernel_size[1] * in_channels
            layer_neurons_str = f'{out_channels} x {kernel_size[0]} x {kernel_size[1]} x {in_channels}'
            layers.append((idx_all, idx, name, layer_type, layer_neurons_str, layer_neurons))
            idx += 1

        elif isinstance(module, torch.nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            layer_neurons = in_features * out_features
            layer_neurons_str = f'{in_features} x {out_features}'
            layers.append((idx_all, idx, name, layer_type, layer_neurons_str, layer_neurons))
            idx += 1

        elif show_hidden_layers:
            layers.append((idx_all, idx, name, layer_type, "", 0))
            #idx += 1
        
        idx_all += 1

    nn_print_table(layers, ['#', '#' 'Layer', 'Type', 'Output Shape', 'Neurons'], title='Model Summary')


# 比较两个神经网络
def nn_print_model_diff(net1, net2):
    # 比较两个神经网络
    diff1 = []
    diff2 = []
    for i, (layer1, layer2) in enumerate(zip(net1, net2)):
        if type(layer1) != type(layer2):
            if type(layer1) < type(layer2):
                diff1.append((i, layer1))
            else:
                diff2.append((i, layer2))
        elif layer1.weight.shape != layer2.weight.shape:
            if layer1.weight.shape < layer2.weight.shape:
                diff1.append((i, layer1))
            else:
                diff2.append((i, layer2))

    # 输出比较结果
    table = []
    for i, (layer1, layer2) in enumerate(zip(net1, net2)):
        if i in [x[0] for x in diff1]:
            table.append([i, str(diff1[[x[0] for x in diff1].index(i)][1]), ''])
        elif i in [x[0] for x in diff2]:
            table.append([i, '', str(diff2[[x[0] for x in diff2].index(i)][1])])
        else:
            table.append([i, '', ''])
    print(tabulate(table, headers=['layer', 'net1_diff', 'net2_diff']))


#def nn_get_model_layers_count(model):
#    return sum(1 for _ in model.modules() if isinstance(_, torch.nn.Conv2d))
from collections import defaultdict
def nn_stat_modules(model, idx=0, indent=1):
    module_counts = defaultdict(int)
    layer_tree = ""
    for module in model.children():
        module_name = type(module).__name__
        module_counts[module_name] += 1
        layer_tree += f"[{idx}]{'-' * indent}{module_name}\n"
        
        idx += 1
        
        module_counts_child, child_tree = nn_stat_modules(module, idx, indent + 1)
        layer_tree += child_tree

        for module_type, count in module_counts_child.items():
            module_counts[module_type] += count

    return module_counts, layer_tree
    

# Resnet18
# 一个7x7的卷积层（64个卷积核） Conv2D
# 一个批归一化层 BatchNormalization
# 一个ReLU激活函数 ReLU
# 一个3x3的最大池化层 MaxPooling2D
# 4个残差块，每个块包含2个卷积层和1个跳跃连接 ResidualBlock
# 一个全局平均池化层 GlobalAveragePooling2D
# 一个全连接层 Dense
def nn_get_model_layers_count(model: nn.Module) -> int:

    table, tree_dat = nn_stat_modules(model)
    table = [(module_type, count) for module_type, count in table.items()]        
    print(tabulate(table, headers=["Module Type", "Count"]))
    print(tree_dat)

    total_index = 0
    def _count_layers(module: nn.Module, is_child=False) -> int:
        nonlocal total_index
        layer_count = 0
        for submodule in module.children():
            # print submodule name
            #print(submodule.__class__.__name__)

            if isinstance(submodule, (nn.Sequential)):
                children_count = len(list(submodule.named_modules()))
                for chmodule in submodule.children():
                    layer_count += _count_layers(chmodule, True)
                #print("children_count", children_count)
                #layer_count += children_count

            #elif isinstance(submodule, (nn.ModuleList, nn.ModuleDict)):
                #layer_count += _count_layers(submodule)
            elif not is_child and isinstance(submodule, (nn.ReLU, nn.BatchNorm2d, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                layer_count += 1
                total_index += 1
                print(total_index, submodule.__class__.__name__)
            elif isinstance(submodule, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                layer_count += 1
                total_index += 1
                print(total_index, submodule.__class__.__name__)

            #elif not isinstance(submodule, (nn.ReLU, nn.BatchNorm2d, nn.Dropout, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.AvgPool2d)):
            #elif not isinstance(submodule, (nn.Dropout)):
                #layer_count += 1

        return layer_count

    return _count_layers(model)
    # def _count_layers(module: nn.Module) -> int:
    #     layer_count = 0
    #     for submodule in module.children():
    #         if isinstance(submodule, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
    #             layer_count += _count_layers(submodule)
    #         elif isinstance(submodule, nn.Conv2d) or isinstance(submodule, nn.Linear):
    #             layer_count += 1
    #     return layer_count

    # return _count_layers(model)

    # idx = 0
    # for name, module in model.named_modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         idx += 1
    #     #elif isinstance(module, torch.nn.Linear):
    #     #    idx += 1
    # return idx


def nn_get_model_name(model):
    return f'{type(model).__name__}{nn_get_model_layers_count(model)}'