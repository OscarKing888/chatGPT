@echo on
mkdir logsp

set python_file=Resnet.py
set mode=predict
rem set scheduler=--scheduler
set "batchs=32 64 128 256 512"

if "%*" neq "*--dataset*" (
    set dataset=CIFAR10
)

if "%*" neq "*--inputdir*" (
    set inputdir=.\test
)

call proc_all_template.bat --model ResNet18


@REM echo predicting ResNet18 on CIFAR10
@REM python Resnet.py --mode predict --dataset CIFAR10 --model ResNet18 > logsp\ResNet18_CIFAR10_predict.log

@REM echo predicting ResNet101 on CIFAR10
@REM python Resnet.py --mode predict --dataset CIFAR10 --model ResNet101 > logsp\ResNet101_CIFAR10_predict.log

@REM echo predicting ResNet50 on CIFAR10
@REM python Resnet.py --mode predict --dataset CIFAR10 --model ResNet50 > logsp\ResNet50_CIFAR10_predict.log

@REM echo predicting ResNet18 on STL10
@REM python Resnet.py --mode predict --dataset STL10 --model ResNet18 > logsp\ResNet18_STL10_predict.log

@REM echo predicting ResNet101 on STL10
@REM python Resnet.py --mode predict --dataset STL10 --model ResNet101 > logsp\ResNet101_STL10_predict.log

@REM echo predicting ResNet50 on STL10
@REM python Resnet.py --mode predict --dataset STL10 --model ResNet50 > logsp\ResNet50_STL10_predict.log

echo All predictings completed!
