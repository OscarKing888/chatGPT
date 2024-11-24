mkdir "%CD%__empty"
robocopy "%CD%__empty" "%1" /mt:128 /r:0 /w:0 /MIR /PURGE /NJH /NJS /NC /NS /NDL /XF * >nul 2>&1
rmdir /s /q "%CD%__empty"
rmdir /s /q %1
pause