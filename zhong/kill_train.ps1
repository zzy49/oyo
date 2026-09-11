$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*train_md2_sup*' }
if ($procs) {
    foreach ($p in $procs) {
        Write-Host "Killing PID $($p.ProcessId)"
        Stop-Process -Id $p.ProcessId -Force
    }
} else {
    Write-Host "No train_md2_sup process found"
}
