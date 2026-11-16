import numpy as np
import pytest

import coverage as cv
from detectors import NON_AI


def row(cond, label="none", split="development", run=None, t0=0.0):
    return {"condition": cond, "label": label, "split": split,
            "run_id": run or f"run-{cond}-{split}", "t0_s": t0, "session": "s"}


@pytest.mark.parametrize("cond,label,expected", [
    ("training", "training", "training"),
    ("mixture-concurrent-0p5", "training", "training"),
    ("cadence-jitter-0p4", "partial", None),
    ("serving-decode", "none", "inference"),
    ("serving-prefill", "none", "inference"),
    ("mixture-timeshared-0p25", "none", "inference"),
    ("idle", "none", NON_AI),
    ("periodic-gemm-5s", "none", NON_AI),
    ("cadence-jitter-0p4", "none", None),
    ("training", "none", None),
])
def test_window_class_follows_preregistration(cond, label, expected):
    assert cv.window_class(row(cond, label)) == expected


def test_fit_rows_hold_out_group_and_test_negatives():
    rows = [row("training", "training", "test"),
            row("mixture-concurrent-0p5", "training"),
            row("mixture-timeshared-0p5", "training", "test"),
            row("idle", split="development"), row("idle", split="calibration"),
            row("idle", split="test")]
    fit = cv.fit_rows(rows, "co-location")
    assert not any(r["condition"].startswith("mixture-concurrent") for r in fit)
    assert not any(r["condition"] == "idle" and r["split"] == "test" for r in fit)
    # Strategies and honest training are kept from every split, test included.
    assert any(r["condition"] == "mixture-timeshared-0p5" for r in fit)
    assert any(r["condition"] == "training" and r["split"] == "test" for r in fit)


def test_fit_rows_hold_out_single_run():
    rows = [row("cadence-jitter-0p1", "training", run="a"),
            row("cadence-jitter-0p2", "training", run="b")]
    assert [r["run_id"] for r in cv.fit_rows(rows, None, frozenset({"a"}))] == ["b"]


def test_interval_method_by_run_count():
    rng = np.random.default_rng(1)
    runs = {f"r{i}": rng.random(50) < 0.5 for i in range(3)}
    assert cv.pooled_interval({"r0": runs["r0"]})[0] == "within-run block"
    assert cv.pooled_interval({k: runs[k] for k in ("r0", "r1")})[0].startswith("stratified")
    assert cv.pooled_interval(runs)[0] == "run-level"


def test_stratified_interval_covers_pooled_rate_and_is_degenerate_when_constant():
    a, b = np.ones(60, bool), np.zeros(60, bool)
    lo, hi = cv.stratified_block_ci({"a": a, "b": b})
    assert lo == hi == 0.5
    rng = np.random.default_rng(2)
    x, y = rng.random(200) < 0.3, rng.random(100) < 0.8
    lo, hi = cv.stratified_block_ci({"x": x, "y": y})
    pooled = (x.sum() + y.sum()) / 300
    assert lo < pooled < hi


def test_summarise_excludes_partial_windows():
    rows = [row("mixture-timeshared-0p5", lab, run="r", t0=float(i))
            for i, lab in enumerate(["training", "partial", "training", "none"])]
    pred = np.array(["training", "training", "inference", "training"], dtype=object)
    s = cv.summarise(rows, pred)
    run = s["per_run"]["r"]
    assert run["n_training"] == 2 and run["n_partial"] == 1
    assert run["detection"] == 0.5
    assert run["flag_rate_all"] == 0.75
    assert s["pooled"]["interval_method"] == "within-run block"
