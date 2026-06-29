param(
    [int]$RefreshSeconds = 30,
    [string[]]$Datasets = @("Data_set_3", "Data_set_4")
)

$ROOT      = $PSScriptRoot
$LOGS_BASE = Join-Path $ROOT "Benchmark_Framework\outputs\logs"

while ($true) {
    Clear-Host
    $ts  = Get-Date -Format "HH:mm:ss"
    $sep = "=" * 82
    Write-Host $sep -ForegroundColor DarkCyan
    Write-Host "  BENCHMARK MONITOR   $ts   refresh=${RefreshSeconds}s   Ctrl+C = exit" -ForegroundColor Cyan
    Write-Host $sep -ForegroundColor DarkCyan

    $allDone = $true

    foreach ($dsName in $Datasets) {
        Write-Host ""

        $logDir = Join-Path $LOGS_BASE $dsName
        if (-not (Test-Path $logDir)) {
            Write-Host "  [$dsName]  (log dir not found)" -ForegroundColor DarkGray
            $allDone = $false
            continue
        }

        $logFiles = @(Get-ChildItem -Path $logDir -Filter "*.log" -ErrorAction SilentlyContinue | Sort-Object Name)

        # Decide dataset status from log files: done only if every log has LANE DONE
        $dsDone = ($logFiles.Count -gt 0) -and (($logFiles | ForEach-Object {
            $c = @(Get-Content -Path $_.FullName -ErrorAction SilentlyContinue)
            ($c | Select-String '^\[NVIDIA LANE DONE\]').Count -gt 0
        }) -notcontains $false)

        if (-not $dsDone) { $allDone = $false }

        $dsStatus = if ($dsDone) { "DONE" } else { "RUNNING" }
        $hdrColor = if ($dsDone) { "Green" } else { "Yellow" }

        Write-Host "  [$dsName]  $dsStatus" -ForegroundColor $hdrColor
        Write-Host ("  " + "-" * 78) -ForegroundColor DarkGray

        if ($logFiles.Count -eq 0) {
            Write-Host "  (no log files yet)" -ForegroundColor DarkGray
            continue
        }

        Write-Host ("  {0,-40} {1,-10} {2,-7} {3,-7} {4}" -f "Model","Progress","Solved","Failed","Last action") -ForegroundColor DarkGray

        foreach ($lf in $logFiles) {
            $model = $lf.BaseName
            $lines = @(Get-Content -Path $lf.FullName -ErrorAction SilentlyContinue)

            if ($lines.Count -eq 0) {
                Write-Host ("  {0,-40} {1}" -f $model, "starting...") -ForegroundColor DarkGray
                continue
            }

            $doneMatches  = @($lines | Select-String '^\[(\d+)/(\d+)\] DONE')
            $startMatches = @($lines | Select-String '^\[(\d+)/(\d+)\] START')

            $completed = 0; $total = 0
            if ($doneMatches.Count -gt 0) {
                $m = $doneMatches[-1].Matches[0]
                $completed = [int]$m.Groups[1].Value
                $total     = [int]$m.Groups[2].Value
            } elseif ($startMatches.Count -gt 0) {
                $m     = $startMatches[-1].Matches[0]
                $total = [int]$m.Groups[2].Value
            }
            $progress = if ($total -gt 0) { "$completed/$total" } else { "?/?" }

            $solvedCount = ($lines | Select-String 'DONE.*solved=True').Count
            $failedCount = ($lines | Select-String 'DONE.*solved=False').Count

            $lastLine = $lines[-1]
            $laneDone = ($lines | Select-String '^\[NVIDIA LANE DONE\]').Count -gt 0

            $actionTag = ""
            if ($laneDone) {
                $actionTag = "LANE DONE"
            } elseif ($lastLine -match '^\[\d+/\d+\] ([A-Z]+)') {
                $actionTag = $Matches[1]
            } elseif ($lastLine -match '^\[([A-Z][A-Z _]+)\]') {
                $actionTag = $Matches[1].Trim()
            }

            $col = "White"
            if    ($laneDone)                             { $col = "Green" }
            elseif ($lastLine -match 'GEN ERROR|ERROR')  { $col = "Red" }
            elseif ($lastLine -match 'TIMEOUT')           { $col = "DarkYellow" }
            elseif ($lastLine -match 'GEN START')         { $col = "Cyan" }
            elseif ($lastLine -match 'VALIDATE DONE')     { $col = "Magenta" }

            Write-Host ("  {0,-40} {1,-10} {2,-7} {3,-7} {4}" -f $model, $progress, $solvedCount, $failedCount, $actionTag) -ForegroundColor $col
        }
    }

    if ($allDone) {
        Write-Host ""
        Write-Host ("=" * 82) -ForegroundColor Green
        Write-Host "  All datasets complete. Monitor exiting." -ForegroundColor Green
        Write-Host ("=" * 82) -ForegroundColor Green
        break
    }

    Write-Host ""
    Write-Host "  Next refresh in ${RefreshSeconds}s ..." -ForegroundColor DarkGray
    Start-Sleep -Seconds $RefreshSeconds
}
