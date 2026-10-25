$dirs = Get-ChildItem 'C:\ProgramData\Microsoft\Windows\WER\ReportArchive' -Directory -Filter 'Kernel*' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime
foreach ($d in $dirs) {
    $wer = Join-Path $d.FullName 'Report.wer'
    if (-not (Test-Path $wer -ErrorAction SilentlyContinue)) {
        Write-Output ("[{0:yyyy-MM-dd HH:mm}] {1} (no Report.wer)" -f $d.LastWriteTime, $d.Name)
        continue
    }
    $lines = @(Get-Content $wer -ErrorAction SilentlyContinue)
    $sig0 = ($lines | Where-Object { $_ -match '^Sig\[0\]\.Value=' } | Select-Object -First 1) -replace '^Sig\[0\]\.Value=', ''
    $sig1 = ($lines | Where-Object { $_ -match '^Sig\[1\]\.Value=' } | Select-Object -First 1) -replace '^Sig\[1\]\.Value=', ''
    Write-Output ("[{0:yyyy-MM-dd HH:mm}] {1} | {2} | {3}" -f $d.LastWriteTime, $sig0, $sig1, $sig2)
}
