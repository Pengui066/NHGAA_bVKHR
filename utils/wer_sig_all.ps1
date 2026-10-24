$dirs = Get-ChildItem 'C:\ProgramData\Microsoft\Windows\WER\ReportArchive' -Directory -Filter 'AppCrash_vkhr*' -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending
foreach ($d in $dirs) {
    $wer = Join-Path $d.FullName 'Report.wer'
    if (-not (Test-Path $wer -ErrorAction SilentlyContinue)) { continue }
    $lines = @(Get-Content $wer -ErrorAction SilentlyContinue)
    $get = { param($n) ($lines | Where-Object { $_ -match "^Sig\[$n\]\.Value=" } | Select-Object -First 1) -replace "^Sig\[$n\]\.Value=", "" }
    $module = & $get 3
    $code   = & $get 6
    $offset = & $get 7
    $eventtime = ($lines | Where-Object { $_ -match '^EventTime=' } | Select-Object -First 1) -replace '^EventTime=', ''
    $dt = [DateTime]::FromFileTime([int64]$eventtime)
    Write-Output ("[{0:yyyy-MM-dd HH:mm:ss}] module={1} code={2} offset={3}" -f $dt, $module, $code, $offset)
}
