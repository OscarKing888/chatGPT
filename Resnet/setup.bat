
@echo on

REM Set cache directory
if exist X:\pycache (
	set CACHE_DIR= X:\pycache
) else (
	set CACHE_DIR= .\pycache
)

@echo cache dir:%CACHE_DIR%

REM Check if the virtual environment exists
if exist resnet\Scripts\activate (
    echo Virtual environment 'resnet' already exists.
) else (
    REM Create a virtual environment named 'resnet'
    echo Creating virtual environment 'resnet'...
    python -m venv resnet
)

REM Activate the virtual environment
call resnet\Scripts\activate

REM Install PyTorch 2.0.0 with CUDA 11.1 and torchvision, torchaudio
REM conda install pytorch=2.0.0 torchvision torchaudio -c pytorch -c conda-forge -y
pip3 install --cache-dir %CACHE_DIR% torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install requirements
pip install --cache-dir %CACHE_DIR% -r requirements.txt
pip install --cache-dir %CACHE_DIR%  -U tensorboard-plugin-profile
pip install --cache-dir %CACHE_DIR%  tabulate
pip install --cache-dir %CACHE_DIR%  torchsummary

echo Done!
