setlocal enabledelayedexpansion
 
call _SetEngineDir.bat %1
@echo on
"%EngineDir%\Build\BatchFiles\Build.bat" -projectfiles -project="%prj%" %GenGamePrjParams% -progress -2022

if "%PauseCmd%"=="" (
    timeout /t 5 >nul
) else (
    pause
)


endlocal

