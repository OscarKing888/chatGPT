call _SetEngineDir.bat
"%EngineDir%\Binaries\DotNET\UnrealBuildTool.exe" -projectfiles -project=%prj% %GenGamePrjParams% -progress
pause