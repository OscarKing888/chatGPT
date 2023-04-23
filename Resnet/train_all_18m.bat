@echo on

set dataset=CIFAR10

set logdir=logs\18m_%dataset%
mkdir %logdir%


set scheduler=--scheduler

set batchsize=32
echo Training %dataset% batch_size:%batchsize% scheduler:%scheduler%
call train18m.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

set batchsize=64
echo Training %dataset% batch_size:%batchsize%
call train18m.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

set batchsize=128
echo Training %dataset% batch_size:%batchsize%
call train18m.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log


set batchsize=256
echo Training %dataset% batch_size:%batchsize%
call train18m.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log


set batchsize=512
echo Training %dataset% batch_size:%batchsize%
call train18m.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log


set scheduler=""

set batchsize=32
echo Training %dataset% batch_size:%batchsize% scheduler:%scheduler%
call train18m.bat --dataset %dataset% --batchsize %batchsize% > %logdir%\%dataset%_%batchsize%.log

set batchsize=64
echo Training %dataset% batch_size:%batchsize%
call train18m.bat --dataset %dataset% --batchsize %batchsize% > %logdir%\%dataset%_%batchsize%.log

set batchsize=128
echo Training %dataset% batch_size:%batchsize%
call train18m.bat --dataset %dataset% --batchsize %batchsize% > %logdir%\%dataset%_%batchsize%.log


set batchsize=256
echo Training %dataset% batch_size:%batchsize%
call train18m.bat --dataset %dataset% --batchsize %batchsize% > %logdir%\%dataset%_%batchsize%.log


set batchsize=512
echo Training %dataset% batch_size:%batchsize%
call train18m.bat --dataset %dataset% --batchsize %batchsize% > %logdir%\%dataset%_%batchsize%.log

echo All trainings completed!
