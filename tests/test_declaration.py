"""R1 (spec D3): §3 check 6, the Phase 2a firewall and the bootstrap helpers."""

import numpy as np
import pytest

import checks
import declaration as d
import groups

#: The alpha the check-6 tests run at; every grid value gives the same verdicts
#: (results/declaration/heldout.json).
ALPHA = 0.05


@pytest.mark.parametrize("family", d.FAMILIES)
def test_check6_positive_control(family):
    r = checks.check6_positive_control(family, ALPHA)
    assert r["positive_sessions"] == ["2026-09-22"]
    assert r["n_positive"] >= 100
    assert r["ok_positive"], r


@pytest.mark.parametrize("family", d.FAMILIES)
def test_check6_self_scored_fails_conservative(family):
    """Recorded failure (amendment 2026-10-02b): at the calibration threshold
    R1's own fit windows are flagged below alpha, not at it."""
    r = checks.check6_positive_control(family, ALPHA)
    assert not r["ok_self"] and r["self_ci95"][1] < ALPHA, r


def test_check6_positive_control_fails_on_declared_windows():
    """Declared windows as the 'positive' set must fail the >= 99% bar."""
    _, cal = d.load_windows("calibration")
    r = checks.check6_positive_control("mahalanobis", ALPHA,
                                       positives=d.select(cal, ("serving-decode",)))
    assert not r["ok_positive"], r


def test_firewall_rejects_hidden_training():
    _, dev = d.load_windows("development")
    with pytest.raises(RuntimeError):
        d.assert_not_hidden(d.select(dev, ("mixture-concurrent-0p5",)))
    d.assert_not_hidden(d.select(dev, groups.DECLARED + ("training",)))


def test_groups_cover_every_condition():
    for split in ("development", "calibration"):  # test is unread before Phase 3
        _, rows = d.load_windows(split)
        for c in {r["condition"] for r in rows}:
            groups.group(c)


def test_block_bootstrap_brackets_rate():
    flags = np.zeros(170)
    flags[::10] = 1
    lo, hi = d.block_bootstrap_ci(flags)
    assert lo <= flags.mean() <= hi and hi - lo > 0
    assert d.block_bootstrap_ci(np.zeros(170)) == (0.0, 0.0)


def test_run_bootstrap_pools_over_windows():
    lo, hi = d.run_bootstrap_ci({"a": np.ones(10), "b": np.zeros(30)})
    assert lo == 0.0 and hi == 1.0


@pytest.mark.parametrize("pos", [0, 85, 169])
def test_block_bootstrap_is_position_invariant(pos):
    """One flag anywhere in a run gives the same interval, containing the
    rate (the non-circular version gave [0, 0.0056] around 0.0059 at pos 0)."""
    flags = np.zeros(170)
    flags[pos] = 1
    lo, hi = d.block_bootstrap_ci(flags)
    assert lo <= flags.mean() <= hi
    ref = np.zeros(170)
    ref[85] = 1
    assert (lo, hi) == d.block_bootstrap_ci(ref)
