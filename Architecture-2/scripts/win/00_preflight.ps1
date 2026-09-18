# Step 00 - check the machine before committing to anything.
. "$PSScriptRoot\env.ps1"

Say "GPUs"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Die "nvidia-smi not found" }
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
$caps  = nvidia-smi --query-gpu=compute_cap --format=csv,noheader
$nGpu  = @($caps).Count
$capNum = [int](@($caps)[0] -replace '[^0-9]','')
Ok "$nGpu GPU(s), compute capability $(@($caps)[0].Trim())"
if ($capNum -lt 80) {
  Warn "no native bf16 (pre-Ampere). ACE-Step will auto-select float32."
  Warn "fp16 training is fine - Lightning precision=16-mixed works on Volta."
}

Say "Storage"
$drive = (Split-Path -Qualifier $env:MIREX_DATA_DIR)
$free  = [math]::Floor((Get-PSDrive $drive.TrimEnd(':')).Free / 1GB)
if ($free -lt 600) { Warn "only $free GB free - the 501 GB sample plus harness cache needs ~700 GB" }
else               { Ok "$free GB free on $drive" }

Say "Tools"
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
  Die "ffmpeg missing - install it and put it on PATH (winget install Gyan.FFmpeg)"
}
Ok "ffmpeg present"
Ok "$env:NUMBER_OF_PROCESSORS logical cores"

Say "Paths"
"  MIREX_DATA_DIR       $env:MIREX_DATA_DIR"
"  MIREX_CHECKPOINT_DIR $env:MIREX_CHECKPOINT_DIR"
"  HF_HOME              $env:HF_HOME"
if ($env:MIREX_DATA_DIR.StartsWith($MirexRoot)) {
  Warn "MIREX_DATA_DIR is inside the repo - point it at your data drive"
}
Warn "Native Windows runs ONE GPU per job with no scheduler parallelism gains"
Warn "beyond your GPU count. For a multi-GPU box, use WSL2 and the bash scripts."
Ok "preflight done"
