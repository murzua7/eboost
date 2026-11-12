import numpy as np
import pytest

from eboost import data, runlog


def _toy():
    rng = np.random.default_rng(0)
    return data.Dataset("toy", 1, 2, "multiclass", rng.normal(size=(50, 4)).astype(np.float32),
                        rng.integers(0, 3, size=50).astype(np.float64), 3, 0.25)


def test_save_load_roundtrip_is_exact(tmp_path):
    d = _toy()
    data.save(d, tmp_path)
    back = data.load(tmp_path, 1)
    np.testing.assert_array_equal(back.X, d.X)
    np.testing.assert_array_equal(back.y, d.y)
    assert (back.kind, back.num_class, back.cat_share, back.objective) == (
        "multiclass", 3, 0.25, "multi:softprob")


def test_load_refuses_a_file_that_changed_after_its_manifest(tmp_path):
    data.save(_toy(), tmp_path)
    path = tmp_path / "task1.npz"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="sha256"):
        data.load(tmp_path, 1)


def test_git_state_fails_loud_outside_a_checkout(tmp_path, monkeypatch):
    monkeypatch.delenv("EBOOST_COMMIT", raising=False)
    with pytest.raises(RuntimeError, match="EBOOST_COMMIT"):
        runlog.git_state(tmp_path)
    monkeypatch.setenv("EBOOST_COMMIT", "abc123")
    assert runlog.git_state(tmp_path)["commit"] == "abc123"
