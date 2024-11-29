set CachePath=x:\UESetupCache418\
if exist "%CachePath%" (
    Setup.bat --cache="%CachePath%"
)
pause