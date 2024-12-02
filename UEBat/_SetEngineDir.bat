@echo off
rem 设置引擎目录
set EngineDir=%~dp0..\Engine

rem 设置是否为 UE5（0 表示 UE4，1 表示 UE5）
set IsUE5=1

rem 游戏和编辑器启动的其它参数
set GameParams=-log -Game -Windowed -ResX=1920 -ResY=1080
set EditorParams=-log

rem 生成游戏工程参数 -engine带引擎源码
set GenGamePrjParams=-game -rocket

rem 根据 IsUE5 的值设置引擎可执行文件名称
if "%IsUE5%"=="1" (
    set EngineExe=UnrealEditor.exe
    set EngineExeDebug=UnrealEditor-Win64-Debug.exe
) else (
    set EngineExe=UE4Editor.exe
    set EngineExeDebug=UE4Editor-Win64-Debug.exe
)

rem 获取传入的参数
set "prj=%~1"

rem 检查参数是否为空
color 0C
if "%prj%"=="" (    
    echo 参数为空，查找当前目录中的第一个 .uproject 文件...
    
    rem 使用 for 循环查找第一个 .uproject 文件并设置 prj 变量
    for %%f in (*.uproject) do (
        set "prj=%%~ff"
        goto :found
    )
    echo 当前目录中未找到 .uproject 文件。
    exit /b 1
)

:found

rem 输出结果以供验证
color 0A
echo 设置参数为：
echo ====================================================================
echo 引擎目录:        !EngineDir!
echo 引擎可执行文件:  !EngineExe!
echo 调试可执行文件:  !EngineExeDebug!
echo 设置 prj 为:     !prj!
echo ====================================================================
color 0F