@echo on


set dataset=CIFAR10

set logdir=logs18m%dataset%predict
mkdir %logdir%

set scheduler=--scheduler
set scheduler=

set batchsize=32
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

set batchsize=64
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

set batchsize=128
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

set batchsize=256
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log

set batchsize=512
call predict18M.bat --dataset %dataset% --batchsize %batchsize% %scheduler% > %logdir%\%dataset%_%batchsize%.log


echo All Predicts completed!
