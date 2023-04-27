
@echo on

REM Set cache directory
if exist X:\pycache (
	set CACHE_DIR= X:\pycache
) else (
	set CACHE_DIR= c:\pycache
)

@echo cache dir:%CACHE_DIR%

set env_name=.venv

REM Check if the virtual environment exists
if exist %env_name%\Scripts\activate (
    echo Virtual environment '%env_name%' already exists.
) else (
    REM Create a virtual environment named 'resnet'
    echo Creating virtual environment '%env_name%'...
    python -m venv %env_name%
)

REM Activate the virtual environment
call %env_name%\Scripts\activate

python.exe -m pip install --upgrade pip

REM Install requirements
pip install --cache-dir %CACHE_DIR% -r requirements.txt

echo Done!
