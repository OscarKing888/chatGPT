setlocal enabledelayedexpansion
 
call _SetEngineDir.bat %1
@echo on
"%EngineDir%\GenerateProjectFiles.bat" -projectfiles -project="%prj%" %GenGamePrjParams% -progress

endlocal
pause