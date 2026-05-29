hdc shell reboot -D

hdc wait

hdc shell setenforce 0
hdc fport tcp:12345 tcp:12345
