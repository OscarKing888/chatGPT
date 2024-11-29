call _SetEngineDir.bat
"%EngineDir%\Binaries\DotNET\UnrealBuildTool.exe" -projectfiles -project=%prj% %GenGamePrjParams% -progress -2022
pause