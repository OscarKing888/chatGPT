setlocal enabledelayedexpansion
 
call _SetEngineDir.bat %1
@echo on
"%EngineDir%\Binaries\Win64\%EngineExe%" %prj% %EditorParams%

if "%PauseCmd%"=="" (
    timeout /t 5 >nul
) else (
    pause
)


endlocal
