@echo on
set logdir=.\logs\logs_stl10_predict
mkdir %logdir%


set dataset=STL10
set scheduler=--scheduler

set batchsize=32
echo Predict %dataset% batch_size:%batchsize% scheduler:%scheduler%
call predict_stl10.bat --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

set batchsize=64
echo Predict %dataset% batch_size:%batchsize%
call predict_stl10.bat --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

set batchsize=100
echo Predict %dataset% batch_size:%batchsize%
call predict_stl10.bat --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

echo All Predicts completed!
