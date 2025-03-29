@echo off
setlocal enabledelayedexpansion

:: 检查是否提供了足够的参数
if "%~1"=="" (
    echo 使用方法：
    echo 按章节切分：split_pptx.bat 输入文件路径 输出目录路径
    echo 按页面范围切分：split_pptx.bat 输入文件路径 输出目录路径 页面范围
    echo.
    echo 示例：
    echo split_pptx.bat input.pptx output
    echo split_pptx.bat input.pptx output "1-3,5-8,9-10"
    exit /b 1
)

:: 设置参数
set "input_file=%~1"
set "output_dir=%~2"
set "page_ranges=%~3"

:: 显示接收到的参数
echo 输入文件: !input_file!
echo 输出目录: !output_dir!
echo 页面范围: !page_ranges!

:: 检查输入文件是否存在
if not exist "!input_file!" (
    echo 错误：输入文件 "!input_file!" 不存在
    exit /b 1
)

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到Python，请确保已安装Python并添加到系统环境变量中
    exit /b 1
)

:: 检查依赖是否安装
python -c "import pptx" >nul 2>&1
if errorlevel 1 (
    echo 正在安装必要的依赖...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 错误：安装依赖失败
        exit /b 1
    )
)

:: 调用Python脚本
if "!page_ranges!"=="" (
    echo 正在按章节切分PPTX文件...
    python split_pptx.py "!input_file!" "!output_dir!"
) else (
    echo 正在按指定页面范围切分PPTX文件...
    python split_pptx.py "!input_file!" "!output_dir!" --page-ranges="!page_ranges!"
)

if errorlevel 1 (
    echo 错误：处理PPTX文件时发生错误
    exit /b 1
)

echo 处理完成！
pause 