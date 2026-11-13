# Stage 1a pilot results and stop decision (2026-11-12)

Owner decision, 2026-11-12: "lets kill the idea here and write the substack as a curiosity that led to a failed experiment". e-boost is stopped after this pilot; Stages 1 to 3 do not run.

Run: NScale array 989, processors only, commit 97f34be8, 90 runs (30 OpenML tasks x 3 seeds), 20:50Z to 23:46Z. Raw logs: /mnt/shared-00/matzua/eboost/runs/stage1a-cpu-97f34be8 (copy under runs/nscale/, gitignored). An earlier attempt on cards (jobs 685-688, commit 10d28769) was cancelled after 6 of 90 runs in 4 h 17 min.

Reader output (`uv run python scripts/read_stage1a.py runs/nscale/stage1a-cpu-97f34be8`):

```
finished (task, seed) pairs: 90; datasets: 30; refit skipped for size: 15
H1 matched rounds: A1r vs C1                 n=30 median +2.09% [+1.39%, +4.13%] p=5.14e-06 better on 17%
H1 matched step sum: A1e vs C1               n=30 median +0.04% [-0.10%, +0.55%] p=0.309 better on 47%
A0-T vs C1 (tuned row sampling)              n=30 median -1.17% [-2.56%, -0.07%] p=0.00871 better on 73%
A0 defaults vs A0-T                          n=30 median +17.39% [+10.38%, +21.67%] p=1.02e-07 better on 7%
H2 A2-full-l0.1 vs A1r                       n=26 median +73.39% [+26.76%, +113.69%] p=1.6e-05 better on 12%
H2 A2-full-l1 vs A1r                         n=26 median +36.70% [+11.56%, +55.98%] p=0.000465 better on 19%
H2 A2-full-l10 vs A1r                        n=26 median +7.72% [-0.04%, +23.71%] p=0.0272 better on 35%
H2 A2-full-l100 vs A1r                       n=26 median +0.50% [-1.60%, +5.12%] p=0.548 better on 46%
H2 A2-damped-l1 vs A1r                       n=26 median -0.03% [-0.09%, -0.01%] p=7.02e-05 better on 92%
H2 A2-damped-l10 vs A1r                      n=26 median -0.03% [-0.07%, -0.01%] p=6.03e-05 better on 92%
H3 A3-s0.7 vs A1r (matched row-passes)       n=30 median -3.71% [-5.66%, -0.86%] p=0.00046 better on 80%
H3 A3-s0.4 vs A1r (matched row-passes)       n=30 median -3.83% [-7.82%, -0.49%] p=0.0093 better on 73%
H3 A3-s0.2 vs A1r (matched row-passes)       n=30 median -1.86% [-4.23%, +1.29%] p=0.229 better on 57%
H3 A3-s0.1 vs A1r (matched row-passes)       n=30 median +4.57% [-0.13%, +7.33%] p=0.0473 better on 33%
H3 A3-s0.05 vs A1r (matched row-passes)      n=30 median +15.80% [+9.68%, +23.34%] p=5.14e-06 better on 13%
H3 knee share s (per dataset, median over seeds): median 0.70, counts {np.float64(0.1): np.int64(1), np.float64(0.2): np.int64(2), np.float64(0.4): np.int64(9), np.float64(0.7): np.int64(16), np.float64(1.0): np.int64(2)}
```

Extra comparisons (scripts/extra_read_stage1a.py):

```
A3-s0.7 vs C1: n=30 median -0.10% [-2.41%, +0.94%] p=0.503 better on 50%
A3-s0.7 vs A0-T: n=30 median +0.94% [+0.36%, +2.36%] p=0.000798 better on 27%
A3-s0.4 vs C1: n=30 median -0.89% [-2.83%, +1.29%] p=0.36 better on 57%
A3-s0.4 vs A0-T: n=30 median +0.82% [-0.84%, +1.47%] p=0.229 better on 40%
A3-s0.2 vs C1: n=30 median +1.45% [-0.84%, +2.60%] p=0.339 better on 40%
A3-s0.2 vs A0-T: n=30 median +3.70% [+1.20%, +5.58%] p=0.000256 better on 17%
A1e vs C1: n=30 median +0.04% [-0.10%, +0.55%] p=0.309 better on 47%
A1e vs A0-T: n=30 median +0.76% [-0.07%, +2.86%] p=0.0113 better on 37%
tuned rounds at the 4000 cap: 14 of 90 runs
A3-s0.05: rounds at the 40000 cap in 30 of 90 runs
A3-s0.1: rounds at the 40000 cap in 11 of 90 runs
A3-s0.2: rounds at the 40000 cap in 0 of 90 runs
A3-s0.4: rounds at the 40000 cap in 0 of 90 runs
A3-s0.7: rounds at the 40000 cap in 0 of 90 runs
tuned row share (A0-T subsample): median 0.74, range 0.60-1.00
```

Findings on the way: XGBoost's LearningRateScheduler applies a rate list one round late (tests/test_schedule.py); GPU pred_leaf in XGBoost 3.4.1 returns wrong leaves for some rows with missing values (tests/test_refit.py; refit now predicts on the processor); damped refit's CG tolerance (1e-10) made some runs take up to 50 minutes.
