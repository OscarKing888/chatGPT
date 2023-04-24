@echo on


set dataset=STL10
set logdir=.\logs\logs_%dataset%_predict
mkdir %logdir%

set scheduler=--scheduler
set log_file=%logdir%\%dataset%_%batchsize%_scheduler.log

setlocal enabledelayedexpansion

set "nums=32 64 100 128"

for %%i in (%nums%) do (
    echo %%i
    set batchsize=%%i
    call predict_stl10.bat --batchsize %batchsize% %scheduler% %* > %log_file%
)

endlocal

echo All Predicts completed!
