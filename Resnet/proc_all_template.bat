@echo off

rem 这些参数在外面调用前设置好，这里不用设置了
rem set python_file=stl10_classifier.py
rem set mode=predict
rem set dataset=STL10
rem set "batchs=32 64 100 128"
rem set scheduler=--scheduler
rem set inputdir=.\test

rem set

rem setlocal enabledelayedexpansion

IF not defined dataset (
    @ECHO dataset is not defined.
)

IF not defined batchs (
    @ECHO batchs is not defined.
)

set scheduler_str=scheduler
IF NOT defined scheduler (
    set scheduler_str=noscheduler
)


set logdir=.\logs\%mode%_%dataset%
@echo %logdir%

if not exist "%logdir%" (
    mkdir %logdir%
)



for %%i in (%batchs%) do (
    @echo start batch:%%i
    set cur_batch=%%i
    set log_file=!logdir!\!dataset!_%%i_!scheduler_str!.log
    @echo save to log: !log_file!

    call python "%python_file%" --mode %mode% --dataset %dataset% --batchsize %%i %scheduler% --inputdir "%inputdir%" %* > "!log_file!"
)

rem endlocal

echo All Predicts completed!
