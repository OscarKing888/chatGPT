setlocal enabledelayedexpansion
 
call _SetEngineDir.bat %1
@echo on
"%EngineDir%\GenerateProjectFiles.bat" -projectfiles -project=%prj% %GenGamePrjParams% -progress -2022

endlocal
pause