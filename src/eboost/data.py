"""Dataset loading for the fit set F, with a hash manifest.

Pilot rule (Stage 1a): OpenML-CC18 (suite 99) and OpenML-CTR23 (suite 353),
minus every dataset in TabArena-v0.1 (suite 457) by id or name, keeping tasks
with 1,000 <= n <= 100,000 rows and at most 500 features, the first 20
classification and first 10 regression tasks by task id.

Categorical columns are ordinal-coded (missing -> NaN) and the share of
categorical columns is recorded for the A4 probe.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SUITES = {"cc18": 99, "ctr23": 353, "tabarena": 457}


@dataclass
class Dataset:
    name: str
    task_id: int
    did: int
    kind: str          # "binary", "multiclass" or "regression"
    X: np.ndarray      # float32, n x p
    y: np.ndarray
    num_class: int | None
    cat_share: float

    @property
    def objective(self) -> str:
        return {"binary": "binary:logistic", "multiclass": "multi:softprob",
                "regression": "reg:squarederror"}[self.kind]


def select_pilot(max_clf: int = 20, max_reg: int = 10, n_min: int = 1_000,
                 n_max: int = 100_000, p_max: int = 500) -> list[tuple[str, int, int, str]]:
    import openml

    listing = openml.datasets.list_datasets(output_format="dataframe").set_index("did")
    tabarena = openml.study.get_suite(SUITES["tabarena"])
    ta_names = {listing.loc[d, "name"].lower() for d in tabarena.data}
    out = []
    for suite_key, kind, cap in [("cc18", "clf", max_clf), ("ctr23", "reg", max_reg)]:
        suite = openml.study.get_suite(SUITES[suite_key])
        kept = 0
        for task_id, did in sorted(zip(suite.tasks, suite.data, strict=True)):
            row = listing.loc[did]
            if did in tabarena.data or row["name"].lower() in ta_names:
                continue
            n, p = int(row["NumberOfInstances"]), int(row["NumberOfFeatures"])
            if n_min <= n <= n_max and p <= p_max and kept < cap:
                out.append((kind, int(task_id), int(did), str(row["name"])))
                kept += 1
    return out


def fetch(task_id: int) -> Dataset:
    import openml
    import pandas as pd

    task = openml.tasks.get_task(task_id, download_splits=False)
    ds = task.get_dataset()
    X, y, cat_mask, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")
    cols = []
    for c, is_cat in zip(X.columns, cat_mask, strict=True):
        s = X[c]
        if is_cat or not pd.api.types.is_numeric_dtype(s):
            codes = pd.Categorical(s).codes.astype(np.float32)
            codes[codes < 0] = np.nan
            cols.append(codes)
        else:
            cols.append(s.to_numpy(dtype=np.float32, na_value=np.nan))
    Xa = np.stack(cols, axis=1)
    cat_share = float(np.mean(cat_mask)) if len(cat_mask) else 0.0
    if task.task_type_id.name == "SUPERVISED_REGRESSION":
        return Dataset(ds.name, task_id, ds.dataset_id, "regression", Xa,
                       y.to_numpy(dtype=np.float64), None, cat_share)
    codes = pd.Categorical(y).codes.astype(np.int64)
    if np.any(codes < 0):
        raise ValueError(f"task {task_id}: missing labels")
    k = int(codes.max()) + 1
    kind = "binary" if k == 2 else "multiclass"
    return Dataset(ds.name, task_id, ds.dataset_id, kind, Xa, codes.astype(np.float64),
                   None if k == 2 else k, cat_share)


def save(d: Dataset, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"task{d.task_id}.npz"
    np.savez_compressed(path, X=d.X, y=d.y)
    meta = {"name": d.name, "task_id": d.task_id, "did": d.did, "kind": d.kind,
            "num_class": d.num_class, "cat_share": d.cat_share, "n": int(len(d.y)),
            "p": int(d.X.shape[1]), "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (root / f"task{d.task_id}.json").write_text(json.dumps(meta, indent=1))
    return meta


def load(root: Path, task_id: int, check_hash: bool = True) -> Dataset:
    meta = json.loads((root / f"task{task_id}.json").read_text())
    path = root / meta["file"]
    if check_hash and hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError(f"{path}: sha256 does not match its manifest")
    z = np.load(path)
    return Dataset(meta["name"], task_id, meta["did"], meta["kind"], z["X"], z["y"],
                   meta["num_class"], meta["cat_share"])
