setlocal enabledelayedexpansion
 
call _SetEngineDir.bat %1
@echo on
"%EngineDir%\GenerateProjectFiles.bat" -projectfiles -project=%prj% %GenGamePrjParams% -progress -2017 -platform=Win64 -SkipPlatformModules=PS5

endlocal
pause