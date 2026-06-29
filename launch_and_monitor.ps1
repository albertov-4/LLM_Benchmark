param(
    [int]$RefreshSeconds = 30
)

$ROOT      = $PSScriptRoot
$LOGS_BASE = Join-Path $ROOT "Benchmark_Framework\outputs\logs"
$BASE_ARGS = "Benchmark_Framework\run_benchmark.py --adapter nvidia_api --protocol-id iterative_repair --parallel-nvidia-models --max-concurrent-nvidia-models 3 --use-real-validator --preflight-tasks"

Write-Host ""
Write-Host "==> Launching Data_set_3 ..." -ForegroundColor Cyan
$proc3 = Start-Process -FilePath "python" `
    -ArgumentList "$BASE_ARGS --run-id Data_set_3" `
    -WorkingDirectory $ROOT `
    -WindowStyle Hidden `
    -PassThru
Write-Host "    PID $($proc3.Id)" -ForegroundColor DarkGray

Write-Host "==> Launching Data_set_4 ..." -ForegroundColor Cyan
$proc4 = Start-Process -FilePath "python" `
    -ArgumentList "$BASE_ARGS --run-id Data_set_4" `
    -WorkingDirectory $ROOT `
    -WindowStyle Hidden `
    -PassThru
Write-Host "    PID $($proc4.Id)" -ForegroundColor DarkGray

Write-Host ""
Write-Host "Both runs started. Entering monitor (Ctrl+C quits monitor only)." -ForegroundColor Green
Start-Sleep -Seconds 4

$RUNS = @(
    [PSCustomObject]@{ Name = "Data_set_3"; Proc = $proc3 },
    [PSCustomObject]@{ Name = "Data_set_4"; Proc = $proc4 }
)

while ($true) {
    Clear-Host
    $ts  = Get-Date -Format "HH:mm:ss"
    $sep = "=" * 82
    Write-Host $sep -ForegroundColor DarkCyan
    Write-Host "  BENCHMARK MONITOR   $ts   refresh=${RefreshSeconds}s   Ctrl+C = exit monitor" -ForegroundColor Cyan
    Write-Host $sep -ForegroundColor DarkCyan

    $allDone = $true

    foreach ($run in $RUNS) {
        $dsName    = $run.Name
        $proc      = $run.Proc
        $isRunning = -not $proc.HasExited
        if ($isRunning) { $allDone = $false }

        $procBadge = if ($isRunning) { "RUNNING  pid=$($proc.Id)" } else { "FINISHED  exit=$($proc.ExitCode)" }
        $hdrColor  = if ($isRunning) { "Yellow" } else { "Green" }

        Write-Host ""
        Write-Host "  [$dsName]  $procBadge" -ForegroundColor $hdrColor
        Write-Host ("  " + "-" * 78) -ForegroundColor DarkGray

        $logDir = Join-Path $LOGS_BASE $dsName
        if (-not (Test-Path $logDir)) {
            Write-Host "  (log dir not created yet - still initialising)" -ForegroundColor DarkGray
            continue
        }

        $logFiles = @(Get-ChildItem -Path $logDir -Filter "*.log" -ErrorAction SilentlyContinue | Sort-Object Name)
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

            $completed = 0
            $total     = 0
            if ($doneMatches.Count -gt 0) {
                $m         = $doneMatches[-1].Matches[0]
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
        Write-Host "  Both Data_set_3 and Data_set_4 are complete. Monitor exiting." -ForegroundColor Green
        Write-Host ("=" * 82) -ForegroundColor Green
        break
    }

    Write-Host ""
    Write-Host "  Next refresh in ${RefreshSeconds}s ..." -ForegroundColor DarkGray
    Start-Sleep -Seconds $RefreshSeconds
}
