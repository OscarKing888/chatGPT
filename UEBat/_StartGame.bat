setlocal enabledelayedexpansion
 
call _SetEngineDir.bat %1
@echo on
start %EngineDir%\Engine\Binaries\Win64\%EngineExe% %prj% %GameParams%
endlocal
pause