"""The H3 monitor rules (spec D4; pre-registration §5)."""

import numpy as np
import pytest

import monitor as m


def test_three_of_five_onset_and_rearm():
    f = [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1]
    onset, active = m.alarm(f)
    # Third positive in five at index 3; cleared by the 5th negative (index 8).
    assert np.flatnonzero(onset).tolist() == [3, 11]
    assert active.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1]


def test_stuck_positive_one_onset_and_full_alarm():
    onset, active = m.alarm(np.ones(833, bool))
    assert onset.sum() == 1 and onset[2]
    assert active.mean() == pytest.approx(831 / 833)


def test_persistence_variants():
    f = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1], bool)
    assert np.flatnonzero(m.alarm(f, need=2, of=3)[0]).tolist() == [1, 8]
    assert m.alarm(f, need=5, of=5)[0].sum() == 0


def test_alarm_intervals_open_at_end():
    t1 = np.arange(10.0)
    onset, active = m.alarm([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert m.alarm_intervals(t1, onset, active) == [(3.0, np.inf)]


def test_garwood_zero_count():
    assert m.garwood_upper(0) == pytest.approx(2.9957, abs=1e-4)


def windows_at(times):
    return [{"t0_s": t - 5.0, "t1_s": t} for t in times]


def episode(first, last):
    return {"first": first, "last": last, "steps": np.array([[first, first + 0.3]])}


def test_episode_classes_in_order():
    times = np.arange(5.0, 100.0)
    flags = (times >= 30) & (times <= 40)
    onset, active = m.alarm(flags)  # onset at t=32, clears at t=45
    eps = [episode(10, 10.5),   # truncated: nominal 15
           episode(33, 48),     # pre-alarmed: alarm active at 33
           episode(28, 43),     # detected: onset 32 inside [28, 48]
           episode(60, 75)]     # missed
    out = m.classify_episodes(eps, windows_at(times), onset, active, nominal_s=15.0)
    assert [e["class"] for e in out] == ["truncated", "pre-alarmed", "detected", "missed"]
    assert out[2]["delay_s"] == pytest.approx(4.0)


def test_unscorable_episode_in_break():
    times = np.concatenate([np.arange(5.0, 20.0), np.arange(40.0, 60.0)])
    onset, active = m.alarm(np.zeros(len(times), bool))
    out = m.classify_episodes([episode(21, 34)], windows_at(times), onset, active, 13.0)
    assert out[0]["class"] == "unscorable"


def test_gap_guard_switches_headline():
    eps = [{"class": "pre-alarmed", "complete_step_window": True, "delay_s": None}] * 4
    assert m.cell_summary(eps, 0.1)["headline"]["rate"] == 1.0
    s = m.cell_summary(eps, 0.5)
    assert s["headline_metric"] == "onset" and s["headline"]["rate"] == 0.0


def test_lower_bound_is_min_of_two():
    s = np.array([1] * 9 + [0], float)
    lb = m.within_run_lower_bound(s)
    assert lb["lower"] == min(lb["clopper_pearson"], lb["block_bootstrap_p5"]) < 0.9


def test_lmin():
    assert m.lmin({5.1: 0.2, 15.1: 0.6, 60.1: 0.7}) == 15.1
    assert m.lmin({5.1: 0.6, 15.1: 0.4, 60.1: 0.7}) == 60.1
    assert m.lmin({5.1: 0.2, 60.1: 0.3}) is None


def test_cusum():
    assert m.cusum([1, 1, -5, 2], k=0.5).tolist() == [0.5, 1.0, 0.0, 1.5]
