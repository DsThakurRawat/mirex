"""Confound audit gate (plan §4.4). Probe AUROC on non-content features must
stay below config.CONFOUND_GATE_AUROC or training is not permitted."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import config
from metadata_db import MetadataDatabase
from datasets import load_audio
from simulator import DeliveryChainSimulator
from confound_audit import run_audit

N = int(os.environ.get("AUDIT_N", "4000"))
db, sim = MetadataDatabase(), DeliveryChainSimulator()
rows = db.fetch("split IS NULL OR split != 'dev_frozen'")[:N]
if not rows:
    sys.exit("no rows in the metadata DB — run scripts/02_fetch_data.sh first")

waves, labels, groups = [], [], []
for i, r in enumerate(rows):
    try:
        w, sr = load_audio(r["file_path"], max_s=60)
    except Exception:
        continue
    waves.append(sim.random_chain(w, sr, item_key=r["track_id"], excerpt=False))
    labels.append(int(r["is_ai"]))
    groups.append(r["source_dataset"])
    if i % 500 == 0:
        print(f"  ...{i}/{len(rows)}", flush=True)

rep = run_audit(waves, labels, groups, sr=sim.sr,
                report_path=config.PROCESSED_DATA_DIR / "confound_report.json")
print(f"\n  worst probe AUROC : {rep['worst_auroc']:.4f}")
print(f"  gate threshold    : {config.CONFOUND_GATE_AUROC}")
print("  top feature leaks:")
for k, v in sorted(rep["per_feature_auroc"].items(), key=lambda kv: -kv[1])[:6]:
    print(f"    {k:26s} {v:.3f}")
sys.exit(0 if rep["gate_passed"] else 1)
