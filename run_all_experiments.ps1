# Run all thesis experiments sequentially with auto-push to GitHub.
# Logs everything to results/run_log_<timestamp>.txt.
# Stops on first error (so later experiments don't run with broken state).
#
# Usage:
#   .\run_all_experiments.ps1                # full sequence (~5-15 hours)
#   .\run_all_experiments.ps1 -SkipAsync     # skip the long async runs

param(
    [switch]$SkipAsync = $false
)

$ErrorActionPreference = "Continue"   # don't abort whole script on a single non-zero exit
Set-Location $PSScriptRoot

# Timestamped log file
$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
$logFile   = "results\run_log_$timestamp.txt"
New-Item -ItemType Directory -Path "results" -Force | Out-Null

function Run-Step {
    param(
        [string]$Name,
        [string]$Script,
        [string]$CommitMsg
    )
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor Cyan
    Write-Host "  STEP: $Name" -ForegroundColor Cyan
    Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor Cyan

    "`n$('=' * 70)`n  STEP: $Name`n  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n$('=' * 70)" | Out-File -Append $logFile

    & python -u $Script 2>&1 | Tee-Object -FilePath $logFile -Append
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  FAILED with exit code $LASTEXITCODE" -ForegroundColor Red
        return $false
    }

    Write-Host "  Pushing to GitHub: '$CommitMsg'" -ForegroundColor Green
    & .\backup.ps1 $CommitMsg 2>&1 | Tee-Object -FilePath $logFile -Append
    return $true
}

# Clean caches once at the start
Write-Host "Cleaning Python caches ..." -ForegroundColor Yellow
Get-ChildItem -Path . -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Initial backup
& .\backup.ps1 "Starting full experiment sweep $timestamp" 2>&1 | Tee-Object -FilePath $logFile -Append

# Run the short ones first
$steps = @(
    @{Name="6.3.1 WS graph stats (~2 min)";       Script="run_ws_graph_stats.py";      Msg="Table 6 ws_graph_stats DONE"}
    @{Name="6.3.2 Sync p=0.1 (~10 min)";          Script="run_sync_p01.py";            Msg="Table 7 sync + Figure 3 DONE"}
    @{Name="6.2 Distributed sync 4 baselines (~10-30 min)"; Script="run_sync_regular.py"; Msg="Tables 2-3 distributed sync DONE"}
    @{Name="6.4 Sync sensitivity (~30-60 min)";   Script="run_sync_sensitivity.py";    Msg="Section 6.4 sync sensitivity DONE"}
)

# Add the long async ones unless --SkipAsync
if (-not $SkipAsync) {
    $steps += @(
        @{Name="6.2 Async on 4 baselines (~30-90 min)";    Script="run_async_regular.py";      Msg="Tables 4-5 async on 4 baselines DONE"}
        @{Name="6.3.2 Async p=0.1 (~50-100 min)";          Script="run_async_p01.py";          Msg="Table 7 async + Figure 4 DONE"}
        @{Name="6.4 Async sensitivity (~5-10 hours)";      Script="run_async_sensitivity.py";  Msg="Section 6.4 async sensitivity DONE"}
    )
}

$startTime = Get-Date

foreach ($step in $steps) {
    $ok = Run-Step -Name $step.Name -Script $step.Script -CommitMsg $step.Msg
    if (-not $ok) {
        Write-Host "ABORTING: stopped after failure in $($step.Name)" -ForegroundColor Red
        break
    }
}

$elapsed = (Get-Date) - $startTime
Write-Host ""
Write-Host ("=" * 70) -ForegroundColor Cyan
Write-Host "  ALL DONE - total time: $($elapsed.Hours)h $($elapsed.Minutes)m" -ForegroundColor Green
Write-Host "  Log: $logFile" -ForegroundColor Green
Write-Host ("=" * 70) -ForegroundColor Cyan
