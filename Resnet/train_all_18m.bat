@echo on

set dataset=CIFAR10

set logdir=logs\18m_%dataset%
mkdir %logdir%


set scheduler=--scheduler
set log_file=%logdir%\%dataset%_%batchsize%_scheduler.log

setlocal enabledelayedexpansion

set "nums=32 64 128 256 512"

for %%i in (%nums%) do (
    echo batchsize:%%i
    set batchsize=%%i
    call train18m.bat --dataset %dataset% --batchsize %batchsize% %scheduler% %* > %log_file%
)

set scheduler=""
set log_file=%logdir%\%dataset%_%batchsize%_noscheduler.log

for %%i in (%nums%) do (
    echo batchsize:%%i
    set batchsize=%%i
    call train18m.bat --dataset %dataset% --batchsize %batchsize% %scheduler% %* > %log_file%
)

endlocal

echo All Predicts completed!