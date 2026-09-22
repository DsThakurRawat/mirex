"""
MusicScope-CL — Official MIREX 2026 Submission Entrypoint
Powered by the ultra-fast FastMusicScopeScorer engine (new_inference.py).

Legacy single-threaded implementation is preserved in: old_inference.py
"""
import sys
from pathlib import Path

_CUR_DIR = Path(__file__).resolve().parent
if str(_CUR_DIR) not in sys.path:
    sys.path.insert(0, str(_CUR_DIR))

from new_inference import (
    FastMusicScopeScorer,
    FastMusicScopeScorer as MusicScopeCLScorer,
    extract_features_single,
    main,
)

if __name__ == "__main__":
    main()
