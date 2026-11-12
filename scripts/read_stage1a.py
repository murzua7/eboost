"""Read Stage 1a pilot results: per-dataset medians over seeds, then the
F-stage comparisons with the dataset as the unit of inference.

Relative difference of arm X against reference R on one dataset:
    rel = median_over_seeds(L_X) / median_over_seeds(L_R) - 1   (negative = X better)
Across datasets: median rel, a bootstrap 95% CI over datasets (10,000
resamples, seed 0) and a two-sided Wilcoxon signed-rank p-value.
Knee (H3): the largest row share s whose test loss is within +0.2% of the
smallest loss over the s grid (s = 1 is A1r), per dataset and seed.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

COMPARISONS = [
    ("H1 matched rounds: A1r vs C1", "A1r", "C1"),
    ("H1 matched step sum: A1e vs C1", "A1e", "C1"),
    ("A0-T vs C1 (tuned row sampling)", "A0-T", "C1"),
    ("A0 defaults vs A0-T", "A0", "A0-T"),
] + [(f"H2 {a} vs A1r", a, "A1r") for a in
     ("A2-full-l0.1", "A2-full-l1", "A2-full-l10", "A2-full-l100", "A2-damped-l1",
      "A2-damped-l10")] + [(f"H3 {a} vs A1r (matched row-passes)", a, "A1r") for a in
                           ("A3-s0.7", "A3-s0.4", "A3-s0.2", "A3-s0.1", "A3-s0.05")]
S_GRID = {"A3-s0.05": 0.05, "A3-s0.1": 0.1, "A3-s0.2": 0.2, "A3-s0.4": 0.4, "A3-s0.7": 0.7,
          "A1r": 1.0}


def load(run_dir: Path):
    loss = defaultdict(dict)   # (task, seed) -> arm -> test loss
    meta, done, skipped = {}, set(), set()
    for f in sorted(run_dir.glob("task*_seed*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            key = (r["task"], r["seed"])
            meta[r["task"]] = (r["dataset"], r["kind"], r["n_train"])
            if r["record"] == "arm":
                loss[key][r["arm"]] = r["test_loss"]
            elif r["record"] == "done":
                done.add(key)
            elif r["record"] == "refit_skipped":
                skipped.add(key)
    return {k: v for k, v in loss.items() if k in done}, meta, skipped


def per_dataset(loss, arm, ref):
    by_task = defaultdict(lambda: ([], []))
    for (task, _), arms in loss.items():
        if arm in arms and ref in arms:
            by_task[task][0].append(arms[arm])
            by_task[task][1].append(arms[ref])
    return {t: np.median(a) / np.median(r) - 1.0 for t, (a, r) in by_task.items()
            if np.median(r) > 0}


def summarize(rel: dict, rng) -> dict:
    v = np.array(list(rel.values()))
    if len(v) < 2:
        return {"n": len(v), "median": float(v[0]) if len(v) else float("nan")}
    boots = np.median(v[rng.integers(0, len(v), size=(10_000, len(v)))], axis=1)
    p = stats.wilcoxon(v).pvalue if np.any(v != 0) else 1.0
    return {"n": len(v), "median": float(np.median(v)),
            "ci": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "wilcoxon_p": float(p), "share_better": float(np.mean(v < 0))}


def knees(loss):
    out = defaultdict(list)
    for (task, _), arms in loss.items():
        pts = [(s, arms[a]) for a, s in S_GRID.items() if a in arms]
        if len(pts) < len(S_GRID):
            continue
        best = min(l for _, l in pts)
        out[task].append(max(s for s, l in pts if l <= best * 1.002))
    return {t: float(np.median(v)) for t, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    loss, meta, skipped = load(a.run_dir)
    rng = np.random.default_rng(0)
    print(f"finished (task, seed) pairs: {len(loss)}; datasets: {len({t for t, _ in loss})}; "
          f"refit skipped for size: {len(skipped)}")
    rows = {}
    for label, arm, ref in COMPARISONS:
        s = summarize(per_dataset(loss, arm, ref), rng)
        rows[label] = s
        ci = f"[{s['ci'][0]:+.2%}, {s['ci'][1]:+.2%}]" if "ci" in s else ""
        extra = (f"p={s['wilcoxon_p']:.3g} better on {s['share_better']:.0%}"
                 if "wilcoxon_p" in s else "")
        print(f"{label:<44} n={s['n']:>2} median {s['median']:+.2%} {ci} {extra}")
    k = knees(loss)
    if k:
        kv = np.array(list(k.values()))
        print(f"H3 knee share s (per dataset, median over seeds): median {np.median(kv):.2f}, "
              f"counts {dict(zip(*np.unique(kv, return_counts=True), strict=True))}")
    if a.json:
        a.json.write_text(json.dumps({"comparisons": rows, "knees": k,
                                      "datasets": {str(t): m for t, m in meta.items()}},
                                     indent=1))


if __name__ == "__main__":
    main()
