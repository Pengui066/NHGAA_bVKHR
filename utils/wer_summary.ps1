$dirs = Get-ChildItem 'C:\ProgramData\Microsoft\Windows\WER\ReportArchive' -Directory -Filter 'AppCrash_vkhr*' |
    Sort-Object LastWriteTime -Descending | Select-Object -First 6
foreach ($d in $dirs) {
    Write-Output ("== " + $d.Name + "  [" + $d.LastWriteTime + "]")
    Get-ChildItem $d.FullName | ForEach-Object { Write-Output ("    file: " + $_.Name) }
    $wer = Join-Path $d.FullName 'Report.wer'
    if (Test-Path $wer) {
        Get-Content $wer | Select-String 'ExceptionCode=|FaultingModule=|FaultingOffset=|AppPath=' |
            Select-Object -ExpandProperty Line -Unique | ForEach-Object { Write-Output ("    " + $_) }
    }
    Write-Output ""
}
