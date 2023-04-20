@echo on
mkdir logsp18m

echo predicting ResNet18 on CIFAR10
python Resnet18.py --mode predict --dataset CIFAR10 > logsp18m\ResNet18M_CIFAR10_predict.log

echo predicting ResNet101 on CIFAR10
python Resnet18.py --mode predict --dataset STL10 > logsp18m\ResNet18M_STL10_predict.log

echo All predictings completed!
