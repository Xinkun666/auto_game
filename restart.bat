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
        set "EXIT_CODE=1"
        goto :fail
    )
)

echo Rebooting device...
"%HDC%" shell reboot -D
if errorlevel 1 (
    set "EXIT_CODE=%errorlevel%"
    echo Failed to reboot device with hdc.
    goto :fail
)

echo Waiting for device...
"%HDC%" wait
if errorlevel 1 (
    set "EXIT_CODE=%errorlevel%"
    echo Failed while waiting for device with hdc.
    goto :fail
)

exit /b 0

:fail
echo.
echo restart.bat failed with exit code %EXIT_CODE%.
echo Please review the output above.
pause
exit /b %EXIT_CODE%
