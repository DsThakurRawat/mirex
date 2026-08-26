# Step 01 - venv, dependencies, and the test suite.
. "$PSScriptRoot\env.ps1"
Set-Location $MirexRoot

if (-not (Test-Path $PY)) { Say "creating venv"; python -m venv .venv }
Say "installing dependencies"
& $PIP install --upgrade pip -q
& $PIP install -r requirements.txt -q
Ok "dependencies installed"

Say "creating directory tree"
& $PY -c "import sys; sys.path.insert(0,'src'); import config; config.ensure_dirs(); print(config.DATA_DIR)"

Say "running tests (expect 107 passed)"
& "$MirexRoot\.venv\Scripts\pytest.exe" tests/ -q
if ($LASTEXITCODE -ne 0) { Die "tests failed" }
Ok "setup complete"
