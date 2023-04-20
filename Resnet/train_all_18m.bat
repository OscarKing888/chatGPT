@echo on
mkdir logs18m

echo Training ResNet18M on CIFAR10
python Resnet18.py --mode train --dataset CIFAR10 > logs18m\ResNet18_CIFAR10_train.log

echo Training ResNet18M on STL10
python Resnet18.py --mode train --dataset STL10 > logs18m\ResNet18_STL10_train.log

echo All trainings completed!
