@echo on
set logdir=.\logs\logs_stl10_train
mkdir %logdir%


set dataset=STL10
set scheduler=--scheduler
set log_file=%logdir%\%dataset%_%batchsize%.log

set batchsize=32
call train_stl10.bat --batchsize %batchsize% %scheduler% > %log_file%

set batchsize=64
call train_stl10.bat --batchsize %batchsize% %scheduler% > %log_file%

set batchsize=100
call train_stl10.bat --batchsize %batchsize% %scheduler% > %log_file%

set batchsize=128
call train_stl10.bat --batchsize %batchsize% %scheduler% > %log_file%

echo All trainings completed!

rem shutdown /t 10 /s /f /c "Training completed. Shutting down in 10 seconds."