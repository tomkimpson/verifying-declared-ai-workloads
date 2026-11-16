"""Spec §3 checks 1, 2, 3, 5 on the real corpus, each with a perturbation
that must make it fail (spec task 1.3)."""

import numpy as np
import pytest

import checks
import detectors
import windows

#: ≥ 20 replication traces for check 2: every 15th of the 300, which spans all
#: five groups and three classes.
CHECK2_STRIDE = 15


def check2_ids():
    _, _, _, _, ids = detectors.replication_traces()
    return ids[::CHECK2_STRIDE]


def test_check1_replication_reproduces():
    r = checks.check1_replication()
    assert r["ok"], r


def test_check1_fails_on_shuffled_labels():
    _, _, y, _, _ = detectors.replication_traces()
    r = checks.check1_replication(np.random.default_rng(0).permutation(y))
    assert not r["ok"], r


def test_check2_features_match():
    ids = check2_ids()
    assert len(ids) >= 20
    r = checks.check2_features(ids)
    assert r["ok"], r


def test_check2_fails_on_shifted_samples():
    r = checks.check2_features(check2_ids()[:3], offset_samples=1000)
    assert not r["ok"], r


def test_check5_episode_labels_match():
    r = checks.check5_episodes()
    assert r["ok"], r
    # 192 episodes is the count spec task 5.3 is written against.
    assert r["n_episodes"] == 192


def test_check5_fails_on_2ms_shift():
    assert not checks.check5_episodes(shift_s=2e-3)["ok"]


@pytest.fixture(scope="module")
def sprint_rows():
    ids = [e["capture_id"] for e in
           windows.usable_captures(checks.SPRINT_FAMILIES, split="development")]
    assert len(ids) == 35
    return list(windows.window_many(ids))


@pytest.mark.slow
def test_check3_sprint_reproduces(sprint_rows):
    r = checks.check3_sprint(checks.replica_label_counts(sprint_rows))
    assert r["ok"], r


@pytest.mark.slow
def test_check3_fails_on_one_window_with_training_features(sprint_rows):
    """Give one serving-decode window a training window's features; the
    forest and tabulation must then disagree with the sprint."""
    rows = list(sprint_rows)
    i = next(k for k, r in enumerate(rows) if r["condition"] == "serving-decode")
    donor = next(r for r in rows if r["condition"] == "training" and r["label"] == "training")
    rows[i] = {**rows[i], "features": donor["features"]}
    assert not checks.check3_sprint(checks.replica_label_counts(rows))["ok"]

def test_pin_rejects_a_path_absent_at_the_commit():
    import corpus
    with pytest.raises(corpus.CorpusNotPinned):
        corpus.check_pinned(("code/no_such_module.py",))


def _grouped_synthetic(seed: int = 0):
    """Three separable classes in 20 groups of 10 rows (task 7.4's shape)."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.array(["training", "inference", "non-ai"] * 7)[:20], 10)
    centre = {"training": 2.0, "inference": 0.0, "non-ai": -2.0}
    x = np.array([centre[c] for c in y])[:, None] + rng.normal(0, 0.5, (len(y), 4))
    return x, y, np.repeat(np.arange(20), 10)


def test_own_check_scores_separable_classes():
    x, y, g = _grouped_synthetic()
    f1, acc = detectors.grouped_kfold_scores(x, y, g, n_estimators=50)
    assert f1 >= 0.90 and acc >= 0.90


def test_own_check_fails_on_shuffled_labels():
    x, y, g = _grouped_synthetic()
    y = np.random.default_rng(1).permutation(y)
    f1, _ = detectors.grouped_kfold_scores(x, y, g, n_estimators=50)
    assert f1 < 0.90
