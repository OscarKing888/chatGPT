@echo on

set dataset=CIFAR10

set logdir=logs\18m_%dataset%
mkdir %logdir%


set scheduler=--scheduler
set log_file=%logdir%\%dataset%_%batchsize%_scheduler.log

set batchsize=32
call train18m.bat --dataset %dataset% --batchsize %batchsize% %* %scheduler% > %log_file%

set batchsize=64
call train18m.bat --dataset %dataset% --batchsize %batchsize% %* %scheduler% > %log_file%

set batchsize=128
call train18m.bat --dataset %dataset% --batchsize %batchsize% %* %scheduler% > %log_file%


set batchsize=256
call train18m.bat --dataset %dataset% --batchsize %batchsize% %* %scheduler% > %log_file%


set batchsize=512
call train18m.bat --dataset %dataset% --batchsize %batchsize% %* %scheduler% > %log_file%


set scheduler=""
set log_file=%logdir%\%dataset%_%batchsize%_noscheduler.log

set batchsize=32
call train18m.bat --dataset %dataset% --batchsize %batchsize% %* > %log_file%

set batchsize=64
call train18m.bat --dataset %dataset% --batchsize %batchsize% %* > %log_file%

set batchsize=128
call train18m.bat --dataset %dataset% --batchsize %batchsize% %* > %log_file%


set batchsize=256
call train18m.bat --dataset %dataset% --batchsize %batchsize% %* > %log_file%


set batchsize=512
call train18m.bat --dataset %dataset% --batchsize %batchsize% %* > %log_file%

echo All trainings completed!
