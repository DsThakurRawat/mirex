"""Fit stacked fusion + isotonic calibration on LOGO out-of-fold scores.

Expects oof_scores.jsonl, one row per held-out track:
    {"scores": {"a": 0.9, "b": 0.7, ...}, "label": 1, "fold": "suno"}

NOTE: nothing in the repo generates that file yet — see RUNBOOK.md step 09.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import config
from fusion import StackedFusion

path = sys.argv[1] if len(sys.argv) > 1 else "oof_scores.jsonl"
if not os.path.exists(path):
    sys.exit(f"{path} not found — the OOF scoring script still needs writing "
             "(RUNBOOK.md step 09).")
oof = [json.loads(l) for l in open(path)]
print(f"  {len(oof)} out-of-fold rows, folds: "
      f"{sorted(set(r['fold'] for r in oof))}")
f = StackedFusion()
f.fit_from_logo(oof)          # logs the a..e weights and intercept
f.save()
print(f"  saved to {config.FUSION_DIR}")
