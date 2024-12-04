setlocal enabledelayedexpansion
 
call _SetEngineDir.bat %1
@echo on
"%EngineDir%\Build\BatchFiles\Build.bat" -projectfiles -project="%prj%" %GenGamePrjParams% -progress -vscode

if "%PauseCmd%"=="" (
    timeout /t 5 >nul
) else (
    pause
)


endlocal

