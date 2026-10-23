# =====================================================================
#  Table 1 shaded-domain re-run (after the 5-line position fix)  v2
#  Per style: export recon -> offline shading -> metrics
#  IMPORTANT: each style needs ITS OWN .vkhr scene file for shading.
#  The script finds it in this order:
#    1) the $MODELS map below (if you fill it)
#    2) your PowerShell history (a past command containing "<style>_eval")
#    3) a file named "<style>.vkhr" under common asset folders
#  If no scene file is found: export still runs (it does not need one),
#  shading+metrics for that style are skipped and listed at the end.
#  Run again later with -SkipExport to only do the skipped shading.
# =====================================================================

param([switch]$SkipExport)

# conda MKL numpy + pip torch each bundle a libiomp5md.dll; without this the
# python steps abort at OpenMP init (OMP Error #15).
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

# ---------------- config ----------------
$ROOT = "D:\pengui000\Denoise\vkhr"
$STYLES = @("ponytail", "bear", "wcurly", "wstraight", "wwavy")

# Optional but BEST: fill your scene files here, one per style.
# Example:
$MODELS = @{
    ponytail  = "share\scenes\ponytail.vkhr"
    bear      = "share\scenes\bear.vkhr"
    wcurly    = "share\scenes\wcurly.vkhr"
    wstraight = "share\scenes\wstraight.vkhr"
    wwavy     = "share\scenes\wwavy.vkhr"
  }
# $MODELS = @{}

# vkhr.exe path - auto-detected from history if left empty.
$VKHR_EXE = ".\bin\vkhr.exe"
# ----------------------------------------

Set-Location $ROOT
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$start = Get-Date
$log   = "shade_rerun_$stamp.log"
Start-Transcript -Path $log | Out-Null

# ---- load PowerShell history once ----
$script:HIST = @()
try {
    $histPath = (Get-PSReadlineOption).HistorySavePath
    if (Test-Path $histPath) { $script:HIST = Get-Content $histPath }
} catch {}

# ---- auto-detect vkhr.exe (history first, then common build dirs) ----
if (-not $VKHR_EXE) {
    $hit = @($script:HIST | Where-Object { $_ -match '\S*vkhr(\.exe)?\b' -and $_ -match '--model' })
    if ($hit.Count -gt 0 -and $hit[-1] -match '("[^"]+"|\S*vkhr(\.exe)?)') {
        $VKHR_EXE = $Matches[1] -replace '"', ''
    }
}
if (-not $VKHR_EXE) {
    foreach ($c in @(".\build\Release\vkhr.exe", ".\build\Debug\vkhr.exe", ".\build\vkhr.exe", ".\vkhr.exe")) {
        if (Test-Path $c) { $VKHR_EXE = $c; break }
    }
}
if (-not $VKHR_EXE) {
    Write-Host "ERROR: cannot find vkhr.exe. Fill `$VKHR_EXE at the top of the script."
    Stop-Transcript; exit 1
}
Write-Host "vkhr exe : $VKHR_EXE"

# ---- resolve the scene file for one style ----
function Get-ModelForStyle([string]$style) {
    if ($MODELS.Count -gt 0 -and $MODELS[$style]) { return $MODELS[$style] }

    # 2) history: a past command mentioning "<style>_eval" with --model
    $tag = "${style}_eval"
    $lines = @($script:HIST | Where-Object { $_ -match [regex]::Escape($tag) -and $_ -match '--model' })
    if ($lines.Count -gt 0 -and $lines[-1] -match '--model\s+("[^"]+"|\S+)') {
        return ($Matches[1] -replace '"', '')
    }

    # 3) same-named scene file under common asset folders
    foreach ($d in @("asset", "assets", "models", "scenes", "data")) {
        $dir = Join-Path $ROOT $d
        if (Test-Path $dir) {
            $f = Get-ChildItem $dir -Recurse -Filter "$style.vkhr" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($f) { return $f.FullName }
        }
    }
    return $null
}

$done = @(); $failed = @(); $skipped = @()

foreach ($s in $STYLES) {
    $dump = "dumps\dataset_full\${s}_eval"
    if (-not (Test-Path $dump)) {
        Write-Host "SKIP [$s]: dump dir not found: $dump"; $skipped += $s; continue
    }
    Write-Host ""
    Write-Host "================ [$s] ================"

    # ---- 1/3 export recon G-buffer (no scene file needed) ----
    if ($SkipExport) {
        Write-Host "---- [$s] 1/3 export skipped (-SkipExport)"
    } else {
        Write-Host "---- [$s] 1/3 export recon"
        python nn\export_reconstructed.py --data dumps\prepared_full --dump $dump
        if ($LASTEXITCODE -ne 0) { Write-Host "FAIL export [$s]"; $failed += $s; continue }
    }

    # ---- 2/3 offline shading with THIS style's scene ----
    $model = Get-ModelForStyle $s
    if (-not $model) {
        Write-Host "SKIP shading [$s]: no .vkhr scene found."
        Write-Host "       Fill `$MODELS['$s'] in the script, then re-run with -SkipExport."
        $skipped += "${s}(no scene)"; continue
    }
    Write-Host "---- [$s] 2/3 shading (scene: $model)"
    # NOTE: vkhr has no --model flag. The scene file is a POSITIONAL argument,
    # and parse() STOPS at the first non-flag token - so every flag must come
    # BEFORE the scene path. Boolean flags CONSUME the next token as their
    # value ("yes"/"no") - a bare "--shade" swallows "--shade-dir" and
    # silently disables shading. AND vkhr.exe is a GUI-subsystem binary:
    # PowerShell's "&" returns IMMEDIATELY without waiting for it, so use
    # Start-Process -Wait (bit us twice on 2026-10-12).
    $shadeOut = "shade_${s}.out.log"; $shadeErr = "shade_${s}.err.log"
    $proc = Start-Process -FilePath $VKHR_EXE `
        -ArgumentList "--shade yes --shade-dir $dump --shade-source recon $model" `
        -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $shadeOut -RedirectStandardError $shadeErr
    if ($proc.ExitCode -ne 0) { Write-Host "FAIL shade [$s] (exit $($proc.ExitCode))"; $failed += $s; continue }
    # Guard against a silent no-op: shaded files must be fresher than this run.
    $freshShaded = Get-ChildItem (Join-Path $dump "recon") -Filter "*_shaded_offline.f32" -ErrorAction SilentlyContinue |
                   Where-Object { $_.LastWriteTime -gt $start }
    if (-not $freshShaded) {
        Write-Host "FAIL shade [$s]: vkhr wrote no fresh *_shaded_offline.f32"
        $failed += $s; continue
    }

    # ---- 2b/3 re-shade the INPUT and GT references with the SAME
    # calibrated alpha (the dump-time *_shaded.f32 files use the legacy
    # alpha and must not be mixed with the mode-2 reconstruction) ----
    foreach ($src in @("input", "gt")) {
        Write-Host "---- [$s] shading $src reference"
        $proc = Start-Process -FilePath $VKHR_EXE `
            -ArgumentList "--shade yes --shade-dir $dump --shade-source $src $model" `
            -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput "shade_${s}_${src}.out.log" -RedirectStandardError "shade_${s}_${src}.err.log"
        if ($proc.ExitCode -ne 0) { Write-Host "FAIL shade $src [$s] (exit $($proc.ExitCode))"; $failed += $s; continue }
        $freshRef = Get-ChildItem (Join-Path $dump $src) -Filter "*_shaded_offline.f32" -ErrorAction SilentlyContinue |
                    Where-Object { $_.LastWriteTime -gt $start }
        if (-not $freshRef) {
            Write-Host "FAIL shade $src [$s]: no fresh *_shaded_offline.f32"
            $failed += $s; continue
        }
    }

    # ---- 3/3 shaded-domain metrics (Table 1 style) ----
    Write-Host "---- [$s] 3/3 metrics"
    python nn\metrics.py --dump $dump --recon-source recon --require-offline
    if ($LASTEXITCODE -ne 0) { Write-Host "FAIL metrics [$s]"; $failed += $s; continue }

    $done += $s
}

Write-Host ""
Write-Host "================ summary ================"
Write-Host "done   : $($done -join ', ')"
Write-Host "failed : $($failed -join ', ')"
Write-Host "skipped: $($skipped -join ', ')"
Write-Host "log    : $ROOT\$log"
Write-Host "Send the log back to Koko for the comparison read."
Stop-Transcript
