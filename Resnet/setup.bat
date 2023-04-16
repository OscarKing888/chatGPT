@echo off
echo Creating the "resnet" conda environment...
conda create -n resnet python=3.10.6 -y
echo.

echo Activating the "resnet" environment...
call conda activate resnet
echo.

echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
echo.


echo Installing PyTorch 2.0.0 (CUDA 11.8 version)...
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.

conda install pytorch torchvision -c pytorch


echo Environment setup completed.
pause
