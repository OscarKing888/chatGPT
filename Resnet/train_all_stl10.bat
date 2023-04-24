@echo on


set python_file=stl10_classifier.py
set mode=train

if "%*" neq "*--dataset*" (
    set dataset=STL10
)

if "%*" neq "*--inputdir*" (
    set inputdir=.\test2
)

set "batchs=32 64 100 128"

call proc_all_template.bat



@REM set logdir=.\logs\logs_stl10_train
@REM mkdir %logdir%


@REM set dataset=STL10
@REM set scheduler=--scheduler
@REM set log_file=%logdir%\%dataset%_%batchsize%.log

@REM set batchsize=32
@REM call train_stl10.bat --batchsize %batchsize% %scheduler% > %log_file%

@REM set batchsize=64
@REM call train_stl10.bat --batchsize %batchsize% %scheduler% > %log_file%

@REM set batchsize=100
@REM call train_stl10.bat --batchsize %batchsize% %scheduler% > %log_file%

@REM set batchsize=128
@REM call train_stl10.bat --batchsize %batchsize% %scheduler% > %log_file%

@REM echo All trainings completed!

rem shutdown /t 10 /s /f /c "Training completed. Shutting down in 10 seconds."