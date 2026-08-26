# Step 07 - train every branch on every LOGO fold.
#
# 5 branches x 7 folds = 35 jobs. train.py hardcodes devices=1, so one job owns
# one GPU; this schedules 35 jobs across however many GPUs the box has.
# Batch sizes come from config.BRANCHES - do not override unless a job OOMs.
# AMP is already on (Lightning precision="16-mixed").
#
# Resumable: a fold with an existing checkpoint is skipped.
#   $env:BRANCHES="ab"; $env:EPOCHS="20"; .\scripts\win\07_train_all.ps1
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot

$branches = if ($env:BRANCHES) { $env:BRANCHES } else { "abcde" }
$folds    = @("suno","udio","mureka","minimax","yue","ace-step","none")
$epochs   = if ($env:EPOCHS)  { $env:EPOCHS }  else { "10" }
# Windows dataloader workers use spawn, not fork - keep this modest.
$workers  = if ($env:WORKERS) { $env:WORKERS } else { "2" }
$nGpu     = if ($env:NGPU) { [int]$env:NGPU } else { @(nvidia-smi --query-gpu=index --format=csv,noheader).Count }
if ($nGpu -lt 1) { Die "no GPUs visible" }

Say "smoke test first (CPU, seconds) - validates the pipeline before GPU-weeks"
& $PY src\train.py --branch a --holdout suno --smoke
if ($LASTEXITCODE -ne 0) { Die "smoke test failed" }
Ok "smoke passed"

# --- build the job list, skipping folds already trained ---
$queue = [System.Collections.Queue]::new()
foreach ($b in $branches.ToCharArray()) {
  foreach ($f in $folds) {
    $foldDir = if ($f -eq "none") { "full" } else { "logo_$f" }
    $dir = Join-Path $env:MIREX_CHECKPOINT_DIR "$b\$foldDir"
    if ((Test-Path $dir) -and (Get-ChildItem $dir -Filter *.ckpt -ErrorAction SilentlyContinue)) {
      Ok "skip $b/$f (checkpoint exists)"
    } else {
      $queue.Enqueue(@{ Branch = "$b"; Fold = $f })
    }
  }
}
$total = $queue.Count
if ($total -eq 0) { Ok "every fold already trained"; exit 0 }
Say "$total job(s) to run across $nGpu GPU(s), $epochs epochs each"

# --- GPU slot scheduler: poll for a free slot, launch, repeat ---
$running = @{}                       # gpu index -> @{Proc;Branch;Fold;Log}
$start = Get-Date; $n = 0
while ($queue.Count -gt 0 -or $running.Count -gt 0) {
  while ($running.Count -lt $nGpu -and $queue.Count -gt 0) {
    $free = (0..($nGpu-1) | Where-Object { -not $running.ContainsKey($_) })[0]
    $job  = $queue.Dequeue(); $n++
    $log  = Join-Path $LOGS "train_$($job.Branch)_$($job.Fold).log"
    Write-Host ("  [{0,2}/{1,2}] gpu{2}  branch {3}  holdout {4,-9} -> {5}" -f $n,$total,$free,$job.Branch,$job.Fold,$log)
    $env:CUDA_VISIBLE_DEVICES = "$free"     # inherited by the child at launch
    $p = Start-Process -PassThru -NoNewWindow -FilePath $PY -ArgumentList @(
          "src\train.py","--branch",$job.Branch,"--holdout",$job.Fold,
          "--epochs",$epochs,"--workers",$workers) `
          -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $running[$free] = @{ Proc = $p; Branch = $job.Branch; Fold = $job.Fold; Log = $log }
  }
  Start-Sleep -Seconds 10
  foreach ($gpu in @($running.Keys)) {
    if ($running[$gpu].Proc.HasExited) {
      $r = $running[$gpu]
      if ($r.Proc.ExitCode -eq 0) { Write-Host "  done  $($r.Branch)/$($r.Fold)" -ForegroundColor Green }
      else { Write-Host "  FAIL  $($r.Branch)/$($r.Fold) - see $($r.Log)" -ForegroundColor Red }
      $running.Remove($gpu)
    }
  }
}
Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue

Say "training finished in $([int]((Get-Date) - $start).TotalMinutes) min"
Say "checkpoints"
Get-ChildItem $env:MIREX_CHECKPOINT_DIR -Recurse -Filter *.ckpt |
  ForEach-Object { "  " + $_.FullName.Replace("$env:MIREX_CHECKPOINT_DIR\","") }
$failed = @(Get-ChildItem $LOGS -Filter "train_*.log*" | Where-Object { Select-String -Path $_ -Pattern "Traceback" -Quiet })
if ($failed.Count -gt 0) { Warn "$($failed.Count) job log(s) contain tracebacks - check $LOGS" }
Ok "next: scripts\win\08_fusion.ps1"
