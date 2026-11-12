import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from eboost import search


@given(n=st.integers(50, 3000), seed=st.integers(0, 10**6), stratified=st.booleans())
@settings(max_examples=60, deadline=None)
def test_split_partitions_rows_without_leakage(n, seed, stratified):
    strata = np.random.default_rng(seed).integers(0, 3, size=n) if stratified else None
    sp = search.split(n, seed, strata=strata)
    # inner rows come from train only, and the three final pieces tile 0..n-1
    assert set(sp.inner_train) | set(sp.inner_val) == set(sp.train)
    assert not set(sp.inner_train) & set(sp.inner_val)
    assert not set(sp.train) & set(sp.test)
    assert sorted(np.concatenate([sp.inner_train, sp.inner_val, sp.test])) == list(range(n))


def test_split_is_reproducible():
    a, b = search.split(1000, 7), search.split(1000, 7)
    np.testing.assert_array_equal(a.test, b.test)
    np.testing.assert_array_equal(a.inner_val, b.inner_val)


def test_sampled_configs_stay_inside_tabarena_ranges():
    for c in search.sample_configs(500, seed=3):
        assert 5e-3 <= c["eta"] <= 1e-1
        assert 4 <= c["max_depth"] <= 10
        assert 1e-3 <= c["min_child_weight"] <= 5.0
        for k in ("subsample", "colsample_bylevel", "colsample_bynode"):
            assert 0.6 <= c[k] <= 1.0
        for k in ("reg_alpha", "reg_lambda"):
            assert 1e-4 <= c[k] <= 5.0
        assert c["grow_policy"] in ("depthwise", "lossguide")
        assert 8 <= c["max_leaves"] <= 1024
