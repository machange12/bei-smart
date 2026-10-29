# Monthly KAMIS pull from a Kenyan machine, pushed before the 5th-of-month CI run.
# kamis.kilimo.go.ke refuses GitHub runners, so CI can't scrape it itself.
# Registered as the "bei-smart KAMIS pull" scheduled task (4th, 10:00); run by hand any time:
#   powershell -ExecutionPolicy Bypass -File scripts\kamis_local_push.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo
Start-Transcript -Path "$env:TEMP\bei_smart_kamis.log" -Append | Out-Null
try {
    $env:PYTHONIOENCODING = "utf-8"
    git pull --rebase --autostash origin main
    & "$repo\venv\Scripts\python.exe" scripts\scrape_kamis.py --discover --months-back 3 --workers 6 --max-minutes 30
    if ($LASTEXITCODE -ne 0) { throw "scrape failed ($LASTEXITCODE)" }
    git add data/raw/kamis_prices.csv
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) { "no new KAMIS rows"; return }
    git commit -m "data: KAMIS pull $(Get-Date -Format yyyy-MM-dd) (local)"
    git push origin main
} finally {
    Stop-Transcript | Out-Null
}
