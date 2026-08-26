# Step 02 - the 501 GB sample. Resumable: re-run safely after any interruption.
. "$PSScriptRoot\env.ps1"
Need-Venv
Set-Location $MirexRoot

function Fetch { Say ("data_fetch " + ($args -join " ")); & $PY src\data_fetch.py @args }

Say "AI pool - small, high family diversity, in full (109 GB)"
Fetch --dataset echoes
Fetch --dataset fakemusiccaps
Fetch --dataset sonics
Fetch --dataset aime

Say "AI pool - giant single-family dumps, cut hard (90 GB)"
Fetch --dataset suno_audio --subset-gb 40
Fetch --dataset muse       --subset-gb 25
Fetch --dataset udio       --subset-gb 25

Say "Real pool (33 GB + MTG below)"
Fetch --dataset musicnet
Fetch --dataset fma        --subset-gb 22
Fetch --dataset mtg_jamendo

Say "SDD - metadata ONLY, audio is quarantined"
Fetch --dataset sdd

# --- MTG-Jamendo audio: capped full-quality subset ---
$mtgAudio = "$env:MIREX_DATA_DIR\raw\mtg_jamendo\audio"
$capGB    = [int]$env:MTG_CAP_GB
$cur      = Dir-SizeGB $mtgAudio
if ($cur -ge $capGB) {
  Ok "MTG audio already at $cur GB (cap $capGB GB)"
} else {
  Say "MTG-Jamendo audio - full quality, stopping at $capGB GB"
  Warn "NOT using the audio-low tier: transcoded real class fails the confound gate"
  if (-not (Test-Path "$env:TEMP\mtgj")) {
    git clone --depth 1 https://github.com/MTG/mtg-jamendo-dataset.git "$env:TEMP\mtgj"
  }
  New-Item -ItemType Directory -Force -Path $mtgAudio | Out-Null
  $p = Start-Process -PassThru -NoNewWindow -FilePath "python" -ArgumentList @(
        "$env:TEMP\mtgj\scripts\download\download.py",
        "--dataset","raw_30s","--type","audio",$mtgAudio)
  while (-not $p.HasExited) {
    Start-Sleep -Seconds 60
    $sz = Dir-SizeGB $mtgAudio
    Write-Host "`r  MTG audio: $sz / $capGB GB" -NoNewline
    if ($sz -ge $capGB) { Write-Host ""; Say "cap reached - stopping"; Stop-Process -Id $p.Id -Force; break }
  }
  Write-Host ""
  Ok "MTG audio at $(Dir-SizeGB $mtgAudio) GB"
}

Say "registering everything into the metadata DB"
& $PY src\data_fetch.py --dataset all --register-only

Say "census"
& $PY -c @"
import sys; sys.path.insert(0,'src')
from collections import Counter
from metadata_db import MetadataDatabase
rows = MetadataDatabase().fetch()
print(f'  total tracks: {len(rows)}')
print(f'  AI / real   : {sum(r[\"is_ai\"] for r in rows)} / {sum(1-r[\"is_ai\"] for r in rows)}')
for fam, n in Counter(r['generator_family'] for r in rows).most_common():
    print(f'    {str(fam):16s} {n}')
"@
Ok "data step complete - next: scripts\win\03_generate.ps1"
