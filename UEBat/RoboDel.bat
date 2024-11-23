mkdir __empty
robocopy __empty %1 /r:0 /w:0 /MIR /PURGE /NJH /NJS /NC /NS /NDL /XF * >nul 2>&1
rmdir /s /q __empty
rmdir /s /q %1
pause