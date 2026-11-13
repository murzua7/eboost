# eboost

A failed experiment. The question was whether the recipe of
[e-means](https://arxiv.org/abs/2611.09501) (an annealed step, a finishing
stage, row-sampled updates and fitted scaling laws in place of tuning) carries
over to XGBoost. A pilot on 30 public OpenML datasets said no, and the project
stopped there on 2026-11-12.

- Results: [docs/progress/2026-11-12-stage1a-pilot.md](docs/progress/2026-11-12-stage1a-pilot.md)
- Write-up: link to follow

## What is here

- `src/eboost/schedule.py`: cosine step size, set before each round
- `src/eboost/refit.py`: joint refit of all leaf values with the tree shapes held fixed
- `src/eboost/arms.py`: the variants compared in the pilot
- `src/eboost/search.py`: random search over TabArena's XGBoost space, and the data splits
- `src/eboost/data.py`, `metrics.py`, `runlog.py`: datasets, scores and an append-only run log
- `scripts/`: fetch the pilot data, run one dataset and seed, read the results
- `jobs/`: the Slurm jobs used on the cluster
- `tests/`: unit and property tests (`uv run pytest`)

## Two things worth knowing about XGBoost

- `xgboost.callback.LearningRateScheduler` applies a list of rates one round
  late, because it sets the rate in `after_iteration`
  (`tests/test_schedule.py`).
- In XGBoost 3.4.1, `pred_leaf` on a GPU can return the wrong leaf for rows
  with missing values, while the GPU margin is right (`tests/test_refit.py`).

## Reproduce

```
uv sync --extra data
uv run python scripts/fetch_pilot.py data/pilot
uv run python scripts/stage1a.py --data data/pilot --task 6 --seed 0 --out runs/local
uv run python scripts/read_stage1a.py runs/local
```

The pilot ran 30 tasks with seeds 0 to 2 on processors (`jobs/nscale_stage1a_cpu.job`).
