"""Extra pilot comparisons: row-sample arms against the constant-step and tuned
arms, plus how often the round caps bind (search cap 4000, A3 cap 40000)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "scripts")
import read_stage1a as R

d = Path(sys.argv[1])
loss, meta, skipped = R.load(d)
rng = np.random.default_rng(0)
for arm in ("A3-s0.7", "A3-s0.4", "A3-s0.2", "A1e"):
    for ref in ("C1", "A0-T"):
        s = R.summarize(R.per_dataset(loss, arm, ref), rng)
        print(f"{arm} vs {ref}: n={s['n']} median {s['median']:+.2%} [{s['ci'][0]:+.2%}, {s['ci'][1]:+.2%}] p={s['wilcoxon_p']:.3g} better on {s['share_better']:.0%}")
tcap, a3cap, subs = [], {}, []
for f in d.glob("task*_seed*.jsonl"):
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if (r["task"], r["seed"]) not in loss:
            continue
        if r["record"] == "search_done":
            tcap.append(r["T_star"] >= 3990)
        if r["record"] == "arm" and r["arm"].startswith("A3"):
            a3cap.setdefault(r["arm"], []).append(r["rounds_built"] >= 40000)
        if r["record"] == "arm" and r["arm"] == "A0-T":
            subs.append(r["config"]["subsample"])
print(f"tuned rounds at the 4000 cap: {sum(tcap)} of {len(tcap)} runs")
for a, v in sorted(a3cap.items()):
    print(f"{a}: rounds at the 40000 cap in {sum(v)} of {len(v)} runs")
print(f"tuned row share (A0-T subsample): median {np.median(subs):.2f}, range {min(subs):.2f}-{max(subs):.2f}")
