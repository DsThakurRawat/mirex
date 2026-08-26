"""Freeze the dev set and materialize the condition x excerpt strata grid."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from harness import freeze_dev_set, materialize_strata

n = freeze_dev_set(per_family=int(os.environ.get("DEV_PER_FAMILY", "150")),
                   real_n=int(os.environ.get("DEV_REAL_N", "1500")))
print(f"  frozen dev tracks: {n}")
manifest = materialize_strata(max_per_cell=int(os.environ.get("MAX_PER_CELL", "40")))
print(f"  manifest: {manifest}")
with open(manifest) as f:
    items = f.readlines()
print(f"  materialized cells: {len(items)} items across "
      f"{len(set(__import__('json').loads(l)['stratum'] for l in items))} strata")
