"""Download the Stage 1a pilot datasets into data/pilot with a hash manifest."""
import json
import sys
from pathlib import Path

from eboost import data

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pilot")
chosen = data.select_pilot()
manifest = []
for kind, task_id, did, name in chosen:
    d = data.fetch(task_id)
    meta = data.save(d, root)
    manifest.append(meta)
    print(f"{task_id:>7} {d.kind:<10} n={meta['n']:>6} p={meta['p']:>4} {name}", flush=True)
(root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
print("datasets", len(manifest))
