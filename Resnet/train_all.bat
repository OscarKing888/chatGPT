@echo on
mkdir logs

echo Training ResNet18 on CIFAR10
python Resnet.py --mode train --dataset CIFAR10 --model ResNet18 > logs\ResNet18_CIFAR10_train.log

echo Training ResNet101 on CIFAR10
python Resnet.py --mode train --dataset CIFAR10 --model ResNet101 > logs\ResNet101_CIFAR10_train.log

echo Training ResNet50 on CIFAR10
python Resnet.py --mode train --dataset CIFAR10 --model ResNet50 > logs\ResNet50_CIFAR10_train.log

echo Training ResNet18 on STL10
python Resnet.py --mode train --dataset STL10 --model ResNet18 > logs\ResNet18_STL10_train.log

echo Training ResNet101 on STL10
python Resnet.py --mode train --dataset STL10 --model ResNet101 > logs\ResNet101_STL10_train.log

echo Training ResNet50 on STL10
python Resnet.py --mode train --dataset STL10 --model ResNet50 > logs\ResNet50_STL10_train.log

echo All trainings completed!
