@echo off

rem 设置引擎目录
set EngineDir=%~dp0..\Engine

rem 设置是否为 UE5（0 表示 UE4，1 表示 UE5）
set IsUE5=0

rem 其它参数
set GameParams="-log -Game -Windowed -ResX=1920 -ResY=1080"
set EditorParams="-log"

rem 根据 IsUE5 的值设置引擎可执行文件名称
if "%IsUE5%"=="1" (
    set EngineExe=UnrealEditor.exe
    set EngineExeDebug=UnrealEditor-Win64-Debug.exe
) else (
    set EngineExe=UE4Editor.exe
    set EngineExeDebug=UE4Editor-Win64-Debug.exe
)

rem 输出结果以供验证
echo 引擎目录: %EngineDir%
echo 引擎可执行文件: %EngineExe%
echo 调试可执行文件: %EngineExeDebug%
