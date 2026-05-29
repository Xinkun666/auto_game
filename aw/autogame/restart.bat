hdc shell reboot -D

hdc wait

:wait_boot
for /f "tokens=*" %%i in ('hdc shell param get sys.boot_completed') do set boot=%%i
if not "%boot%"=="1" (
    timeout /t 2 /nobreak >nul
    goto wait_boot
)

hdc shell setenforce 0
hdc fport tcp:12345 tcp:12345
