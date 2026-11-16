"""Synthetic tests of the logic that decides Phase 2's choice and kill verdict:
held-out workload labelling, like-for-like pooling, threshold stability, the
selection rule and R1a's amplitude feature (PR #2 review)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import declaration as d

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


heldout_mod = _script("declaration_heldout")
choose_mod = _script("declaration_choose")


def _rows(run, condition, n, rng, shift=0.0, capture=None, rms=1.0):
    return [{"run_id": run, "capture_id": capture or run, "condition": condition,
             "session": "s", "t0_s": float(i), "ac_rms_a": rms * (1 + 0.1 * i),
             "features": {"a": rng.normal(shift), "b": rng.normal(shift)}}
            for i in range(n)]


def test_heldout_marks_only_sole_runs_of_a_workload_unseen():
    rng = np.random.default_rng(0)
    fit = (_rows("dec1", "serving-decode", 40, rng) + _rows("dec2", "serving-decode", 40, rng)
           + _rows("idle1", "idle", 40, rng, shift=3.0))
    cal = _rows("calA", "serving-decode", 40, rng)
    ho = heldout_mod.heldout(fit, cal, ["a", "b"], "mahalanobis", False)
    assert {k: v["unseen_workload"] for k, v in ho.items()} == \
        {"dec1": False, "dec2": False, "idle1": True}


def test_check4_pools_like_for_like_and_reports_literal():
    flags = {"dec1": np.r_[np.ones(1), np.zeros(169)],
             "dec2": np.r_[np.ones(1), np.zeros(169)],
             "prefill": np.ones(170)}
    unseen = {"dec1": False, "dec2": False, "prefill": True}
    c = heldout_mod.check4(flags, unseen, 0.05)
    assert c["like_for_like_runs"] == ["dec1", "dec2"]
    assert c["fp_rate"] == pytest.approx(1 / 170)
    assert c["direction"] == "conservative" and not c["ok"]
    assert not c["kill_3alpha_fires"]
    assert c["literal"]["fp_rate"] == pytest.approx(172 / 510)
    assert c["literal"]["kill_3alpha_fires"]


def test_check4_mislabelled_unseen_changes_the_verdict():
    """Marking the unseen run as seen must flip the like-for-like clause."""
    flags = {"dec1": np.zeros(170), "prefill": np.ones(170)}
    assert not heldout_mod.check4(flags, {"dec1": False, "prefill": True}, 0.05)["kill_3alpha_fires"]
    assert heldout_mod.check4(flags, {"dec1": False, "prefill": False}, 0.05)["kill_3alpha_fires"]


def test_stability_flags_scale_moves_and_passes_identical_runs():
    rng = np.random.default_rng(0)
    same = {"r1": rng.normal(10, 1, 200), "r2": rng.normal(10, 1, 200)}
    s = heldout_mod.stability(same, 0.05)
    assert s["stable"] and s["max_factor"] < 1.2
    apart = {"r1": rng.normal(10, 1, 200), "r2": rng.normal(100, 1, 200)}
    s = heldout_mod.stability(apart, 0.05)
    assert not s["stable"] and s["max_factor"] > 2


def _cand(family, variant, alpha, ok=True, stable=True):
    return {"family": family, "variant": variant, "alpha": alpha,
            "check6": {"ok_positive": ok},
            "check4": {"kill_3alpha_fires": False},
            "stability": {"stable": stable}}


def test_choose_takes_largest_alpha_then_r1_then_stable_family():
    res = [_cand(f, v, a, stable=(f == "iforest"))
           for f in ("mahalanobis", "iforest") for v in ("R1", "R1a")
           for a in (0.01, 0.05, 0.10)]
    c = choose_mod.choose(res)
    assert (c["family"], c["variant"], c["alpha"]) == ("iforest", "R1", 0.05)


def test_choose_fixes_variant_before_family():
    """Docstring order: an R1 that passes step 2 wins even if unstable, over
    a stable R1a."""
    res = [_cand("mahalanobis", "R1", 0.05, stable=False),
           _cand("iforest", "R1", 0.05, ok=False),
           _cand("iforest", "R1a", 0.05, stable=True)]
    c = choose_mod.choose(res)
    assert (c["family"], c["variant"]) == ("mahalanobis", "R1")


def test_choose_skips_alpha_where_nothing_passes():
    res = [_cand("iforest", "R1", 0.05, ok=False), _cand("iforest", "R1", 0.02)]
    assert choose_mod.choose(res)["alpha"] == 0.02


def test_r1a_amplitude_is_log_ratio_to_capture_median():
    rng = np.random.default_rng(0)
    rows = (_rows("x", "idle", 5, rng, capture="cx", rms=1.0)
            + _rows("y", "idle", 5, rng, capture="cy", rms=7.0))
    m = d.matrix(rows, ["a", "b"], amplitude=True)
    expect = np.log([1 + 0.1 * i for i in range(5)]) - np.log(1.2)
    np.testing.assert_allclose(m[:5, 2], expect)
    np.testing.assert_allclose(m[5:, 2], expect)  # gain-free across captures
    assert m.shape == (10, 3)
