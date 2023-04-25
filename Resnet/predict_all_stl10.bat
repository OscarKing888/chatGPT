@echo on

set python_file=stl10_classifier.py
set mode=predict


if "%*" neq "*--dataset*" (
    set dataset=STL10
)

if "%*" neq "*--inputdir*" (
    set inputdir=.\test
)

set "batchs=32 64 100 128"

call proc_all_template.bat

