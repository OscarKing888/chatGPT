@echo on
set logdir=.\logs\logs_stl10_train
mkdir %logdir%


set dataset=STL10
set scheduler=--scheduler

set batchsize=32
echo Training %dataset% batch_size:%batchsize% scheduler:%scheduler%
call train_stl10.bat --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

set batchsize=64
echo Training %dataset% batch_size:%batchsize%
call train_stl10.bat --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

set batchsize=100
echo Training %dataset% batch_size:%batchsize%
call train_stl10.bat --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

echo All trainings completed!
