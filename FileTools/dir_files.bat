@echo off
setlocal enabledelayedexpansion

:: Save the directory where the script is run from
set "scriptdir=%CD%"

:: Check if a directory path is provided as a parameter
if "%~1"=="" (
  echo Please provide a directory path as a parameter.
  echo Example: %~nx0 "\\192.168.20.5\Psoul\FromSoftData\Design\Documents"
  goto :eof
)

:: Save the provided directory path to variable basepath
set "basepath=%~1"

:: Try to switch to the specified directory, supports network paths
pushd "%basepath%" || (
  echo Unable to access directory: %basepath%
  goto :eof
)

:: Get the current directory, save to variable currentdir
set "currentdir=%CD%"

:: Recursively list all files, generate relative paths
(for /f "delims=" %%i in ('dir /b /s /a:-d') do (
  set "filepath=%%i"
  set "relativepath=!filepath:%currentdir%\=!"
  echo !relativepath!
)) > "%scriptdir%\list.txt"

:: Return to the previous directory
popd

echo File list saved to: %scriptdir%\list.txt
