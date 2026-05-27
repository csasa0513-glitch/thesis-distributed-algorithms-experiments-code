# Quick backup of thesis-code to GitHub.
# Usage:  .\backup.ps1 "your commit message"
# Or:     .\backup.ps1     (uses default timestamped message)

param(
    [string]$Message = "Auto-backup at $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
)

Set-Location $PSScriptRoot

git add .
$status = git status --porcelain
if ([string]::IsNullOrWhiteSpace($status)) {
    Write-Host "Nothing to commit." -ForegroundColor Yellow
    exit 0
}

git commit -m $Message
if ($LASTEXITCODE -ne 0) {
    Write-Host "Commit failed." -ForegroundColor Red
    exit 1
}

git push
if ($LASTEXITCODE -eq 0) {
    Write-Host "Backup pushed to GitHub: $Message" -ForegroundColor Green
} else {
    Write-Host "Push failed. Check 'git status' / network." -ForegroundColor Red
}
