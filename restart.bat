@echo off
setlocal
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
set "HDC=hdc"
if exist "%SCRIPT_DIR%hdc.exe" (
    set "HDC=%SCRIPT_DIR%hdc.exe"
) else (
    where hdc >nul 2>nul
    if errorlevel 1 (
        echo hdc was not found. Put hdc.exe next to restart.bat or add hdc to PATH.
        exit /b 1
    )
)

echo Rebooting device...
"%HDC%" shell reboot -D
if errorlevel 1 exit /b 1

echo Waiting for device...
"%HDC%" wait
exit /b %errorlevel%
