@echo on
set python_file=Resnet18.py
set mode=train
@REM set scheduler=--scheduler

if "%*" neq "*--dataset*" (
    set dataset=CIFAR10
)

if "%*" neq "*--inputdir*" (
    set inputdir=.\test
)

set "batchs=32 64 128 256 512"

call proc_all_template.bat

@REM set dataset=CIFAR10

@REM set logdir=logs\18m_%dataset%
@REM mkdir %logdir%


@REM set scheduler=--scheduler
@REM set log_file=%logdir%\%dataset%_%batchsize%_scheduler.log

@REM setlocal enabledelayedexpansion

@REM set "nums=32 64 128 256 512"

@REM for %%i in (%nums%) do (
@REM     echo batchsize:%%i
@REM     set batchsize=%%i
@REM     call train18m.bat --dataset %dataset% --batchsize %batchsize% %scheduler% %* > %log_file%
@REM )

@REM set scheduler=""
@REM set log_file=%logdir%\%dataset%_%batchsize%_noscheduler.log

@REM for %%i in (%nums%) do (
@REM     echo batchsize:%%i
@REM     set batchsize=%%i
@REM     call train18m.bat --dataset %dataset% --batchsize %batchsize% %scheduler% %* > %log_file%
@REM )

@REM endlocal

@REM echo All Predicts completed!