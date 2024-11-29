@echo off
setlocal

if "%~1"=="" (
    echo 请提供源目录路径。
    exit /b 1
)

set "SourceDir=%~1"
set "TargetDir=%~dp1%~n1_tmp"

rem 创建目标目录
mkdir "%TargetDir%"

rem 使用 robocopy 移动指定类型文件并保持目录结构
robocopy "%SourceDir%" "%TargetDir%" "._*" /MT:128 /S /MOV /NFL /NDL /NJH /NJS >nul 2>nul

echo 文件已移动至 "%TargetDir%"。
