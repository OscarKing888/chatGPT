@echo on

set dataset=CIFAR10

set logdir=logs18m%dataset%predict
mkdir %logdir%

set scheduler=--scheduler
set scheduler=

set log_file=%logdir%\%dataset%_%batchsize%_noscheduler.log

set batchsize=32
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %*  %scheduler% > %log_file%

set batchsize=64
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %*  %scheduler% > %log_file%

set batchsize=128
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %*  %scheduler% > %log_file%

set batchsize=256
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %*  %scheduler% > %log_file%

set batchsize=512
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %*  %scheduler% > %log_file%


echo All Predicts completed!
