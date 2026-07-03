@echo off
chcp 65001
set is_obfucate=""
set "is_obfucate=%2"
set python_path=""
set "python_path=%1"
if "%python_path%"=="" (
    echo "please provide python path e.g:D:/python3.11.9/"
) else (
    echo "python path：%python_path%"
    if "%is_obfucate%"=="o" (
        echo "混淆"
        %python_path%python.exe build.py -p %python_path% -o
    ) else (
        echo "不混淆"
        %python_path%python.exe build.py -p %python_path%
    )
)
