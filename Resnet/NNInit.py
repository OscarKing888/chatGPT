import os
import torch
import argparse
import matplotlib.pyplot as plt
import PIL.Image as Image
from tabulate import tabulate

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


def nn_args(desc='Image Classifier'):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--mode', default='train', choices=['train', 'predict'], help="Mode: train or predict")
    parser.add_argument('--batchsize', type=int, default=32, help="Batch size for training/testing")
    parser.add_argument('--epochs', type=int, default=60, help="Number of epochs to train")
    parser.add_argument('--input_dir', type=str, default='./test', help="Directory containing images for prediction")    
    parser.add_argument('--model_file', type=str, default='best_model.pth', help="File to save/load model")
    parser.add_argument('--scheduler', default=False, action='store_true', help='Use or not use scheduler (default: True)')
    return parser


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



def nn_save_image_as(src_path, dest_path, image_size=(96, 96)):
    image = Image.open(src_path).resize(image_size)
    save_path = nn_get_err_img_path(dest_path)    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)
    print("Image saved to: ", dest_path)


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
