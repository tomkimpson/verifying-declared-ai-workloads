"""S2's TOST (code/ceiling.py) and its classifier fit (code/heldout.py)."""

import numpy as np
import pytest

import ceiling as c
import declaration as d
import groups
import heldout as h


def test_identical_saturated_arms_are_equivalent_with_zero_width():
    out = c.tost(np.ones(170), np.ones(170), 0, 1)
    assert out["difference"] == 0.0 and out["ci90"] == [0.0, 0.0]
    assert out["equivalent"]


def test_large_difference_rejects_equivalence():
    """R1's positive-control shape: 1.0 against about two thirds."""
    b = np.ones(170)
    b[np.arange(170) % 3 == 0] = 0
    out = c.tost(np.ones(170), b, 0, 3)
    assert out["difference"] == pytest.approx(1 - b.mean())
    assert not out["equivalent"]


def test_interval_touching_the_margin_is_not_equivalent():
    """A difference whose interval reaches delta fails, the boundary included."""
    a = np.ones(200)
    b = np.ones(200)
    b[:10] = 0  # difference 0.05 exactly; a block of zeros, so the interval is wide
    out = c.tost(a, b, 0, 1)
    assert out["difference"] == pytest.approx(0.05)
    assert out["ci90"][1] >= 0.05 and not out["equivalent"]


def test_small_noisy_difference_is_equivalent():
    rng = np.random.default_rng(5)
    a = rng.random(2000) < 0.5
    b = rng.random(2000) < 0.5
    out = c.tost(a, b, 0, 1)
    assert abs(out["difference"]) < 0.03 and out["equivalent"]


def test_runs_get_independent_streams():
    """Same seed for both arms would correlate the resamples and shrink the
    interval of the difference of two identical runs to zero."""
    x = np.zeros(170)
    x[::7] = 1
    assert c.tost(x, x, 0, 0)["ci90"] == [0.0, 0.0]
    lo, hi = c.tost(x, x, 0, 1)["ci90"]
    assert lo < 0 < hi


def test_block_rates_match_block_ci():
    x = np.zeros(170)
    x[::10] = 1
    r = d.block_bootstrap_rates(x, d.BLOCK_WINDOWS, d.N_BOOT, np.random.default_rng(d.SEED))
    assert (float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))) == d.block_bootstrap_ci(x)


def test_s2_fit_holds_out_time_sharing_and_the_arms():
    group, runs = h.CLASSIFIER_FITS[h.S2_FIT]
    assert group == "time-sharing" and runs == h.S2_ARMS
    assert set(c.SEEDS) == set(h.S2_ARMS) | {c.CONTROL}
    assert h.S2_FIT not in h.FITS  # Phase 5 scripts iterate FITS unchanged


def test_s2_fit_rows_exclude_arms_and_control():
    def row(cond, run):
        return {"condition": cond, "label": "training", "split": "development",
                "session": "s", "run_id": run, "capture_id": run, "t0_s": 0.0}
    dev = [row("training", "panel-training-arm1"), row("training-discarded", "panel-training-arm2"),
           row("training-copying", "panel-training-arm3"),
           row("mixture-timeshared-0p5", c.CONTROL), row("training", "other-training")]
    kept = {r["run_id"] for r in h.classifier_fit_rows(dev, h.S2_FIT)}
    assert kept == {"other-training"}
    assert groups.group("mixture-timeshared-0p5") == "time-sharing"
