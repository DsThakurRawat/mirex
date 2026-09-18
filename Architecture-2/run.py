#!/usr/bin/env python3
"""MIREX 2026 pipeline driver — one cross-platform entry point.

Runs on Linux, macOS and Windows with the SYSTEM python and the stdlib only:
it has to work before the venv it creates exists. Every heavyweight import is
deferred into the venv interpreter it spawns.

    python3 run.py preflight          # check the machine
    python3 run.py setup              # venv, deps, tests
    python3 run.py fetch              # the 501 GB sample
    python3 run.py generate           # ACE-Step / YuE / Mureka / MiniMax
    python3 run.py quarantine         # HARD GATE
    python3 run.py confound           # HARD GATE
    python3 run.py harness            # freeze dev set, materialize strata
    python3 run.py train              # 35 jobs across the GPUs
    python3 run.py fusion             # stacked fusion + calibration
    python3 run.py container          # offline Docker + rehearsal
    python3 run.py all                # every step, stopping at a failed gate

Point the data at your array first (the default puts 500 GB in the repo):

    MIREX_DATA_DIR=/raid/mirex/data python3 run.py fetch          # POSIX
    $env:MIREX_DATA_DIR="D:\\mirex\\data"; python run.py fetch     # Windows
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WINDOWS = os.name == "nt"

# --- environment ----------------------------------------------------------
DATA_DIR = Path(os.environ.setdefault("MIREX_DATA_DIR", str(ROOT / "data")))
CKPT_DIR = Path(os.environ.setdefault("MIREX_CHECKPOINT_DIR", str(ROOT / "checkpoints")))
os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
LOGS = ROOT / "logs"

VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if WINDOWS else "bin/python")
PIP = VENV / ("Scripts/pip.exe" if WINDOWS else "bin/pip")

FOLDS = ["suno", "udio", "mureka", "minimax", "yue", "ace-step", "none"]
BRANCHES = "abcde"


# --- output ---------------------------------------------------------------
def _color_ok() -> bool:
    if not sys.stdout.isatty():
        return False
    if WINDOWS:                       # enable ANSI on Windows 10+ consoles
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


_C = _color_ok()
def _p(code: str, tag: str, msg: str) -> None:
    print(f"\033[{code}m{tag}\033[0m {msg}" if _C else f"{tag} {msg}", flush=True)

def say(m):  _p("1", "==>", m)
def ok(m):   _p("32", "  ok", m)
def warn(m): _p("33", " warn", m)
def die(m):  _p("31", " fail", m); sys.exit(1)


class GateFailed(SystemExit):
    """A hard gate rejected the data. Nothing downstream is valid."""


# --- helpers --------------------------------------------------------------
def need_venv() -> None:
    if not PY.exists():
        die("no venv — run `python3 run.py setup` first")


def run(args, *, log: Path | None = None, check: bool = True,
        env: dict | None = None, cwd: Path | None = None) -> int:
    """Run a subprocess, optionally teeing stdout+stderr to a log file."""
    args = [str(a) for a in args]
    full_env = {**os.environ, **(env or {})}
    if log is None:
        rc = subprocess.run(args, env=full_env, cwd=cwd).returncode
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as fh:
            rc = subprocess.run(args, env=full_env, cwd=cwd,
                                stdout=fh, stderr=subprocess.STDOUT).returncode
    if check and rc != 0:
        die(f"command failed (rc={rc}): {' '.join(args[:4])}… see {log or 'output above'}")
    return rc


def dir_size_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    total = 0
    for dirpath, _, names in os.walk(p):
        for n in names:
            try:
                total += (Path(dirpath) / n).stat().st_size
            except OSError:
                pass
    return total / 2**30


def gpu_info() -> list[dict]:
    """[{index, name, mem_mb, cap}] via nvidia-smi, or [] if unavailable."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [f.strip() for f in line.split(",")]
        if len(parts) >= 4:
            gpus.append({"index": int(parts[0]), "name": parts[1],
                         "mem_mb": int(float(parts[2])), "cap": parts[3]})
    return gpus


# --- steps ----------------------------------------------------------------
def step_preflight(_a) -> None:
    say("GPUs")
    gpus = gpu_info()
    if not gpus:
        warn("no nvidia-smi / no GPU visible — training and generation need one")
    for g in gpus:
        print(f"  gpu{g['index']}  {g['name']}  {g['mem_mb']} MiB  cap {g['cap']}")
    if gpus:
        cap = float(gpus[0]["cap"])
        ok(f"{len(gpus)} GPU(s), compute capability {cap}")
        if cap < 8.0:
            warn("no native bf16 (pre-Ampere). ACE-Step auto-selects float32.")
            warn("fp16 training is fine — Lightning precision=16-mixed works on Volta.")

    say("Storage")
    target = DATA_DIR if DATA_DIR.exists() else DATA_DIR.parent
    target.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(target).free / 2**30
    (warn if free_gb < 600 else ok)(
        f"{free_gb:.0f} GB free on {target} "
        f"({'need ~700 GB for the 501 GB sample plus harness cache' if free_gb < 600 else 'sufficient'})")

    say("Tools")
    if not shutil.which("ffmpeg"):
        die("ffmpeg missing — required by the simulator and the tests")
    ok("ffmpeg present")
    ok(f"{os.cpu_count()} logical cores, python {sys.version.split()[0]}")

    say("Paths")
    for k in ("MIREX_DATA_DIR", "MIREX_CHECKPOINT_DIR", "HF_HOME"):
        print(f"  {k:22s} {os.environ[k]}")
    if str(DATA_DIR).startswith(str(ROOT)):
        warn("MIREX_DATA_DIR is inside the repo — point it at your array")
    if WINDOWS:
        warn("Windows: dataloader workers use spawn; train defaults to --workers 2")
        warn("For a multi-GPU box, WSL2 is faster (keep data off /mnt/c)")
    ok("preflight done")


def step_setup(_a) -> None:
    if not PY.exists():
        say("creating venv")
        run([sys.executable, "-m", "venv", str(VENV)])
    say("installing dependencies")
    run([PIP, "install", "--upgrade", "pip", "-q"])
    run([PIP, "install", "-r", str(ROOT / "requirements.txt"), "-q"])
    ok("dependencies installed")

    say("creating directory tree")
    run([PY, "-c", "import sys; sys.path.insert(0,'src'); import config; "
                   "config.ensure_dirs(); print(config.DATA_DIR)"], cwd=ROOT)

    say("running tests (expect 107 passed)")
    run([VENV / ("Scripts/pytest.exe" if WINDOWS else "bin/pytest"),
         "tests/", "-q"], cwd=ROOT)
    ok("setup complete")


# 501 GB allocation: buy generator-family diversity, not volume.
FETCH_PLAN = [
    ("echoes",       None, "8.0 GB · ~10 systems"),
    ("fakemusiccaps", None, "12.9 GB · 5 TTM models"),
    ("sonics",       None, "30.4 GB · suno + udio"),
    ("aime",         None, "58.0 GB · 12 models"),
    ("suno_audio",     40, "subset · keeps version labels for the drift proxy"),
    ("muse",           25, "subset · 576 GB on the hub, one family"),
    ("udio",           25, "subset · 583 GB on the hub, one family"),
    ("musicnet",     None, "11.1 GB · full-length classical"),
    ("fma",            22, "medium tier · hard negatives"),
    ("mtg_jamendo",  None, "metadata TSVs only; audio handled below"),
    ("sdd",          None, "METADATA ONLY — audio is quarantined"),
]


def step_fetch(a) -> None:
    need_venv()
    for name, subset, note in FETCH_PLAN:
        say(f"{name} — {note}")
        cmd = [PY, "src/data_fetch.py", "--dataset", name]
        if subset is not None:
            cmd += ["--subset-gb", str(subset)]
        run(cmd, log=LOGS / "fetch.log", cwd=ROOT)

    _fetch_mtg_audio(cap_gb=a.mtg_cap)

    say("registering everything into the metadata DB")
    run([PY, "src/data_fetch.py", "--dataset", "all", "--register-only"], cwd=ROOT)
    _census()
    ok("data step complete — next: `run.py generate`")


def _fetch_mtg_audio(cap_gb: float) -> None:
    audio = DATA_DIR / "raw" / "mtg_jamendo" / "audio"
    cur = dir_size_gb(audio)
    if cur >= cap_gb:
        ok(f"MTG audio already at {cur:.0f} GB (cap {cap_gb:.0f} GB)")
        return
    say(f"MTG-Jamendo audio — full quality, stopping at {cap_gb:.0f} GB")
    warn("NOT using the audio-low tier: a transcoded real class fails the confound gate")
    repo = Path(os.environ.get("TEMP", "/tmp")) / "mtgj"
    if not repo.exists():
        run(["git", "clone", "--depth", "1",
             "https://github.com/MTG/mtg-jamendo-dataset.git", str(repo)])
    audio.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, str(repo / "scripts" / "download" / "download.py"),
         "--dataset", "raw_30s", "--type", "audio", str(audio)],
        stdout=open(LOGS / "mtg_download.log", "ab"), stderr=subprocess.STDOUT)
    try:
        while proc.poll() is None:
            time.sleep(60)
            sz = dir_size_gb(audio)
            print(f"\r  MTG audio: {sz:.0f} / {cap_gb:.0f} GB", end="", flush=True)
            if sz >= cap_gb:
                print()
                say("cap reached — stopping the MTG downloader")
                proc.terminate()
                break
    finally:
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    print()
    ok(f"MTG audio at {dir_size_gb(audio):.0f} GB")


def _census() -> None:
    say("census")
    run([PY, "-c", (
        "import sys; sys.path.insert(0,'src');"
        "from collections import Counter;"
        "from metadata_db import MetadataDatabase;"
        "rows=MetadataDatabase().fetch();"
        "print(f'  total tracks: {len(rows)}');"
        "print(f\"  AI / real   : {sum(r['is_ai'] for r in rows)} / "
        "{sum(1-r['is_ai'] for r in rows)}\");"
        "[print(f'    {str(f):16s} {n}') for f,n in "
        "Counter(r['generator_family'] for r in rows).most_common()]"
    )], cwd=ROOT, check=False)


GEN_PLAN = [("ace_step", 2000, 1), ("yue", 800, 1),
            ("mureka", 400, 4), ("minimax", 400, 2)]


def step_generate(_a) -> None:
    need_venv()
    for key in ("MUREKA_API_KEY", "MINIMAX_API_KEY"):
        if not os.environ.get(key):
            die(f"{key} is not set")
    src = ROOT / "src"

    say("dry run — checks credentials, spends nothing")
    run([PY, "-m", "generation.campaign", "--backend", "mureka",
         "--count", "10", "--dry-run"], cwd=src)

    for backend, count, workers in GEN_PLAN:
        say(f"{backend} × {count}")
        run([PY, "-m", "generation.campaign", "--backend", backend,
             "--count", str(count), "--workers", str(workers)],
            log=LOGS / f"gen_{backend}.log", cwd=src)

    _wav_to_flac()
    say("re-registering generated tracks")
    run([PY, "src/data_fetch.py", "--dataset", "all", "--register-only"], cwd=ROOT)
    ok("generation complete — next: `run.py quarantine`")


def _wav_to_flac() -> None:
    """ACE-Step writes .wav. FLAC saves ~40% and is LOSSLESS — never use MP3
    here: a lossy codec on the AI class only manufactures the exact confound
    the delivery-chain simulator exists to destroy."""
    gen = DATA_DIR / "generated"
    wavs = list(gen.rglob("*.wav"))
    if not wavs:
        ok("no WAV files to transcode")
        return
    say(f"transcoding {len(wavs)} WAV → FLAC (lossless)")
    before = dir_size_gb(gen)
    from concurrent.futures import ThreadPoolExecutor

    def conv(w: Path) -> None:
        flac = w.with_suffix(".flac")
        rc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(w), str(flac)],
                            capture_output=True).returncode
        if rc == 0 and flac.exists():
            w.unlink()

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        list(ex.map(conv, wavs))
    ok(f"generated pool: {before:.0f} GB → {dir_size_gb(gen):.0f} GB")


def step_quarantine(_a) -> None:
    need_venv()
    say("building the quarantine blocklist")
    run([PY, "src/quarantine.py", "build"], cwd=ROOT)
    say("verifying zero overlap")
    if run([PY, "src/quarantine.py", "verify"], cwd=ROOT, check=False) != 0:
        raise GateFailed("quarantine gate FAILED — do not train")
    ok("quarantine gate PASSED")


def step_confound(a) -> None:
    need_venv()
    say(f"confound audit — decoding up to {a.audit_n} tracks")
    rc = run([PY, "scripts/lib/confound_gate.py"], cwd=ROOT, check=False,
             env={"AUDIT_N": str(a.audit_n)})
    if rc != 0:
        warn("confound gate FAILED")
        warn("suspect the SUBSET COMPOSITION before the model:")
        warn("  · MTG audio tier — audio-low is transcoded → bitrate floor on the real class")
        warn("  · FMA clip length — fma_large/medium are 30 s → duration signature")
        raise GateFailed("not proceeding to training")
    ok("confound gate PASSED")


def step_harness(_a) -> None:
    need_venv()
    say("freezing dev set and materializing strata")
    run([PY, "scripts/lib/harness_build.py"], cwd=ROOT)
    ok(f"harness ready — regenerable, delete {DATA_DIR / 'harness_cache'} to reclaim space")


def _fold_dir(branch: str, fold: str) -> Path:
    return CKPT_DIR / branch / ("full" if fold == "none" else f"logo_{fold}")


def step_train(a) -> None:
    """5 branches x 7 LOGO folds = 35 jobs. train.py hardcodes devices=1, so
    one job owns one GPU; this keeps exactly n_gpu of them in flight."""
    need_venv()
    gpus = [g["index"] for g in gpu_info()] or [0]
    if a.gpus:
        gpus = gpus[:a.gpus]
    workers = a.workers if a.workers is not None else (2 if WINDOWS else 6)

    say("smoke test first (CPU, seconds) — validates the pipeline before GPU-weeks")
    run([PY, "src/train.py", "--branch", "a", "--holdout", "suno", "--smoke"], cwd=ROOT)
    ok("smoke passed")

    queue = []
    for b in a.branches:
        for f in FOLDS:
            d = _fold_dir(b, f)
            if d.exists() and any(d.glob("*.ckpt")):
                ok(f"skip {b}/{f} (checkpoint exists)")
            else:
                queue.append((b, f))
    if not queue:
        ok("every fold already trained")
        return

    total = len(queue)
    say(f"{total} job(s) across {len(gpus)} GPU(s), {a.epochs} epochs each")
    LOGS.mkdir(parents=True, exist_ok=True)

    running: dict[int, tuple] = {}          # gpu -> (proc, branch, fold, log)
    started = failed = 0
    t0 = time.time()
    while queue or running:
        while queue and len(running) < len(gpus):
            gpu = next(g for g in gpus if g not in running)
            b, f = queue.pop(0)
            started += 1
            log = LOGS / f"train_{b}_{f}.log"
            print(f"  [{started:2d}/{total:2d}] gpu{gpu}  branch {b}  "
                  f"holdout {f:<9s} -> {log}")
            proc = subprocess.Popen(
                [str(PY), "src/train.py", "--branch", b, "--holdout", f,
                 "--epochs", str(a.epochs), "--workers", str(workers)],
                cwd=ROOT, stdout=open(log, "wb"), stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            running[gpu] = (proc, b, f, log)
        time.sleep(5)
        for gpu in list(running):
            proc, b, f, log = running[gpu]
            if proc.poll() is not None:
                if proc.returncode == 0:
                    ok(f"done  {b}/{f}")
                else:
                    failed += 1
                    _p("31", "  FAIL", f"{b}/{f} — see {log}")
                del running[gpu]

    say(f"training finished in {(time.time() - t0) / 60:.0f} min")
    ckpts = sorted(CKPT_DIR.rglob("*.ckpt"))
    say(f"{len(ckpts)} checkpoint(s)")
    for c in ckpts:
        print(f"  {c.relative_to(CKPT_DIR)}")
    if failed:
        warn(f"{failed} job(s) failed — check {LOGS}")
    ok("next: `run.py fusion`")


def step_fusion(a) -> None:
    need_venv()
    oof = Path(a.oof)
    if not oof.exists():
        warn(f"MISSING PIECE: {oof} does not exist.")
        warn("StackedFusion.fit_from_logo() needs rows shaped")
        warn('  {"scores": {"a":0.9,...,"e":0.3}, "label": 1, "fold": "suno"}')
        warn("but nothing in the repo generates them yet. You need a script")
        warn("that, for each fold, loads that fold's held-out tracks and scores")
        warn("them with that fold's five checkpoints. The one remaining gap.")
        die("see RUNBOOK.md step 09")
    say(f"fitting fusion on {oof}")
    run([PY, "scripts/lib/fit_fusion.py", str(oof)], cwd=ROOT)
    ok("fusion fitted — select on worst-stratum and macro AUROC, never pooled accuracy")


def step_container(a) -> None:
    need_venv()
    say("pre-populating hf_cache (the Dockerfile COPYs it; build fails if absent)")
    cache = ROOT / "hf_cache"
    cache.mkdir(exist_ok=True)
    run([PY, "-c",
         "from transformers import AutoModel;"
         "AutoModel.from_pretrained('facebook/wav2vec2-xls-r-300m');"
         "AutoModel.from_pretrained('m-a-p/MERT-v1-330M', trust_remote_code=True);"
         "print('  hf cache populated')"],
        cwd=ROOT, env={"HF_HOME": str(cache)})

    say("building image")
    run(["docker", "build", "-t", "mirex2026-detector", "."], cwd=ROOT)
    ok("built mirex2026-detector")

    rehearsal = Path(a.rehearsal) if a.rehearsal else DATA_DIR / "rehearsal"
    if not rehearsal.exists():
        warn(f"no rehearsal dir at {rehearsal} — pass --rehearsal <dir of WAVs>")
        warn("the 10k-track rehearsal on one GPU is a submission requirement")
        return
    n = len(list(rehearsal.rglob("*.wav")))
    out = Path(os.environ.get("TEMP", "/tmp")) / "mirex_out"
    out.mkdir(parents=True, exist_ok=True)
    say(f"runtime rehearsal: {n} tracks, ONE gpu, --network none (must beat 24 h)")
    t0 = time.time()
    run(["docker", "run", "--gpus", "device=0", "--network", "none",
         "-v", f"{rehearsal}:/data/input", "-v", f"{out}:/data/output",
         "mirex2026-detector"], cwd=ROOT)
    el = time.time() - t0
    if n:
        ok(f"scored {n} tracks in {el/60:.0f} min — "
           f"extrapolated 10k: {el * 10000 / n / 3600:.1f} h")


# --- orchestration --------------------------------------------------------
STEPS = [
    ("preflight", step_preflight, "check GPUs, disk, ffmpeg"),
    ("setup",     step_setup,     "venv, dependencies, 107 tests"),
    ("fetch",     step_fetch,     "the 501 GB sample + registration"),
    ("generate",  step_generate,  "ACE-Step / YuE / Mureka / MiniMax -> FLAC"),
    ("quarantine", step_quarantine, "HARD GATE: zero SDD overlap"),
    ("confound",  step_confound,  "HARD GATE: probe AUROC < 0.60"),
    ("harness",   step_harness,   "freeze dev set, materialize strata"),
    ("train",     step_train,     "35 jobs across the GPUs"),
    ("fusion",    step_fusion,    "stacked fusion + calibration"),
    ("container", step_container, "offline Docker + runtime rehearsal"),
]


def step_all(a) -> None:
    for name, fn, _ in STEPS:
        if name in ("preflight", "container"):
            continue
        say(f"───── {name} ─────")
        fn(a)
    ok("pipeline complete")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="run.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn, help_ in STEPS:
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        if name == "fetch":
            p.add_argument("--mtg-cap", type=float,
                           default=float(os.environ.get("MTG_CAP_GB", 190)),
                           help="GB of full-quality MTG-Jamendo audio (default 190)")
        if name == "confound":
            p.add_argument("--audit-n", type=int,
                           default=int(os.environ.get("AUDIT_N", 4000)))
        if name == "train":
            p.add_argument("--branches", default=os.environ.get("BRANCHES", BRANCHES))
            p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 10)))
            p.add_argument("--workers", type=int, default=None,
                           help="dataloader workers (default 6, or 2 on Windows)")
            p.add_argument("--gpus", type=int, default=None, help="cap GPUs used")
        if name == "fusion":
            p.add_argument("--oof", default="oof_scores.jsonl")
        if name == "container":
            p.add_argument("--rehearsal", default=None)

    p = sub.add_parser("all", help="every step, stopping at a failed gate")
    p.set_defaults(fn=step_all)
    p.add_argument("--mtg-cap", type=float, default=190)
    p.add_argument("--audit-n", type=int, default=4000)
    p.add_argument("--branches", default=BRANCHES)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--gpus", type=int, default=None)
    p.add_argument("--oof", default="oof_scores.jsonl")
    p.add_argument("--rehearsal", default=None)

    a = ap.parse_args(argv)
    LOGS.mkdir(parents=True, exist_ok=True)
    try:
        a.fn(a)
    except GateFailed as e:
        die(str(e))
    except KeyboardInterrupt:
        warn("interrupted — every step is resumable, just run it again")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
