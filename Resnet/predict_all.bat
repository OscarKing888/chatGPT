@echo on
mkdir logsp

echo predicting ResNet18 on CIFAR10
python Resnet.py --mode predict --dataset CIFAR10 --model ResNet18 > logsp\ResNet18_CIFAR10_predict.log

echo predicting ResNet101 on CIFAR10
python Resnet.py --mode predict --dataset CIFAR10 --model ResNet101 > logsp\ResNet101_CIFAR10_predict.log

echo predicting ResNet50 on CIFAR10
python Resnet.py --mode predict --dataset CIFAR10 --model ResNet50 > logsp\ResNet50_CIFAR10_predict.log

echo predicting ResNet18 on STL10
python Resnet.py --mode predict --dataset STL10 --model ResNet18 > logsp\ResNet18_STL10_predict.log

echo predicting ResNet101 on STL10
python Resnet.py --mode predict --dataset STL10 --model ResNet101 > logsp\ResNet101_STL10_predict.log

echo predicting ResNet50 on STL10
python Resnet.py --mode predict --dataset STL10 --model ResNet50 > logsp\ResNet50_STL10_predict.log

echo All predictings completed!
