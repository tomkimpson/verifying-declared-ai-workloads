"""Unit tests of the windowing and labelling rules on synthetic inputs."""

import numpy as np
import pytest

import windows


@pytest.mark.parametrize("hop", [1.0, 5.0])
def test_window_starts_match_count_formula(hop):
    rng = np.random.default_rng(0)
    for _ in range(500):
        s0 = rng.uniform(0, 100)
        s1 = s0 + rng.uniform(0, 400)
        starts = windows.window_starts(s0, s1, 5.0, hop)
        assert len(starts) == windows.expected_count(s0, s1, 5.0, hop)
        if len(starts):
            assert starts[-1] + 5.0 <= s1 and starts[0] == s0


def test_window_count_detects_an_off_by_one():
    s0, s1 = 0.3, 175.9
    assert len(windows.window_starts(s0, s1, 5.0, 1.0)[:-1]) != \
        windows.expected_count(s0, s1, 5.0, 1.0)


def test_step_labels():
    steps = np.array([[1.0, 1.3], [4.8, 5.2]])
    full = windows.step_labels(steps, 0.0, 5.0)
    assert full["label"] == "training" and full["n_complete_steps"] == 1
    part = windows.step_labels(steps, 5.1, 10.1)
    assert part["label"] == "partial" and part["n_complete_steps"] == 0
    assert windows.step_labels(steps, 6.0, 11.0)["label"] == "none"
    assert windows.step_labels(np.empty((0, 2)), 0, 5)["label"] == "none"
    assert full["train_frac"] == pytest.approx((0.3 + 0.2) / 5.0)
