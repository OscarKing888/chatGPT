@echo on

set python_file=Resnet.py
set mode=train

if "%*" neq "*--dataset*" (
    set dataset=CIFAR10
)

if "%*" neq "*--inputdir*" (
    set inputdir=.\test
)

set "batchs=32 64 128 256 512"

call proc_all_template.bat

@REM mkdir logs

@REM echo Training ResNet18 on CIFAR10
@REM python Resnet.py --mode train --dataset CIFAR10 --model ResNet18 > logs\ResNet18_CIFAR10_train.log

@REM echo Training ResNet101 on CIFAR10
@REM python Resnet.py --mode train --dataset CIFAR10 --model ResNet101 > logs\ResNet101_CIFAR10_train.log

@REM echo Training ResNet50 on CIFAR10
@REM python Resnet.py --mode train --dataset CIFAR10 --model ResNet50 > logs\ResNet50_CIFAR10_train.log

@REM echo Training ResNet18 on STL10
@REM python Resnet.py --mode train --dataset STL10 --model ResNet18 > logs\ResNet18_STL10_train.log

@REM echo Training ResNet101 on STL10
@REM python Resnet.py --mode train --dataset STL10 --model ResNet101 > logs\ResNet101_STL10_train.log

@REM echo Training ResNet50 on STL10
@REM python Resnet.py --mode train --dataset STL10 --model ResNet50 > logs\ResNet50_STL10_train.log

@REM echo All trainings completed!
