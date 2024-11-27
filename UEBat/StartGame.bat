@ECHO OFF

PUSHD "%~dp0"
..\Engine\Binaries\Win64\UnrealEditor.exe %1.uproject -Game -Windowed -ResX=1920 -ResY=1080
POPD