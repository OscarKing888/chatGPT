
netsh advfirewall firewall add rule name="Allow Port %1" dir=in action=allow protocol=TCP localport=%1
pause