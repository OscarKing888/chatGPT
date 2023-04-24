@echo on

set python_file=Resnet18.py
set mode=predict
rem set scheduler=--scheduler
set "batchs=32 64 128 256 512"

if "%*" neq "*--dataset*" (
    set dataset=CIFAR10
)

if "%*" neq "*--inputdir*" (
    set inputdir=.\test
)

call proc_all_template.bat
