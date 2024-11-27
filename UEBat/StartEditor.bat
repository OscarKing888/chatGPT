@ECHO OFF

PUSHD "%~dp0"
..\Engine\Binaries\Win64\UnrealEditor.exe %1.uproject -log
POPD