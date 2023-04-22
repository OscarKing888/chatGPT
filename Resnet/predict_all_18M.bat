@echo on
set logdir=logs18mstl10predict
mkdir %logdir%


set dataset=STL10
set scheduler=--scheduler

set batchsize=32
echo Predict %dataset% batch_size:%batchsize% scheduler:%scheduler%
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

set batchsize=64
echo Predict %dataset% batch_size:%batchsize%
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

set batchsize=128
echo Predict %dataset% batch_size:%batchsize%
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

echo All Predicts completed!
