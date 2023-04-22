@echo on
set logdir=logs18mstl10
mkdir %logdir%


set dataset=STL10
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

echo All trainings completed!
