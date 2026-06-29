@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

if defined AUTOGAME_RELEASE_PYTHON (
    "%AUTOGAME_RELEASE_PYTHON%" "%~dp0build_release.py" %*
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 "%~dp0build_release.py" %*
    ) else (
        python "%~dp0build_release.py" %*
    )
)

set "RC=%errorlevel%"
echo.
if "%RC%"=="0" (
    echo build_release finished.
) else (
    echo build_release failed, exit_code=%RC%.
)
pause
exit /b %RC%
