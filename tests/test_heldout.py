"""H2/H3 helpers (code/heldout.py) and the episode aggregation in code/monitor.py."""

import numpy as np
import pytest

import heldout as h
import monitor as m


def row(cond, label="none", run=None, t0=0.0, split="development"):
    return {"condition": cond, "label": label, "split": split, "session": "s",
            "run_id": run or f"run-{cond}", "capture_id": run or f"run-{cond}", "t0_s": t0}


def test_roc_endpoints_and_strict_rule():
    r = h.roc(np.array([0.1, 0.2, 0.3]), np.array([0.25, 0.4]))
    assert (r["fpr"][0], r["tpr"][0]) == (1.0, 1.0)
    assert (r["fpr"][-1], r["tpr"][-1]) == (0.0, 0.0)
    # Thresholds are -inf, 0.1, 0.2, 0.25, 0.3, 0.4. At t = 0.25 (flag iff
    # score > t) one negative of three and one positive of two lie above.
    assert r["fpr"][3] == pytest.approx(1 / 3) and r["tpr"][3] == 0.5


def test_false_positives_per_workload():
    rows = ([row("serving-decode", run="d1", t0=i) for i in range(10)]
            + [row("serving-decode", run="d2", t0=i) for i in range(10)]
            + [row("idle", run=f"i{k}", t0=i) for k in range(3) for i in range(10)])
    flags = np.zeros(len(rows), bool)
    flags[:2] = True
    out = h.false_positives(rows, flags)
    assert out["serving-decode"]["fpr"] == pytest.approx(0.1)
    assert out["serving-decode"]["per_run"] == {"d1": 0.2, "d2": 0.0}
    assert out["serving-decode"]["interval_method"].startswith("stratified")
    assert out["idle"]["fpr"] == 0.0 and out["idle"]["interval_method"] == "run-level"


def test_detection_excludes_partial_and_splits():
    rows = [row("mixture-timeshared-0p5", "training", "a", 0), row("mixture-timeshared-0p5", "partial", "a", 1),
            row("mixture-timeshared-0p5", "training", "b", 0, "test")]
    out = h.detection(rows, np.array([True, False, False]))
    assert out["all"]["pooled"]["detection"] == 0.5
    assert out["development"]["pooled"]["detection"] == 1.0
    assert out["test"]["pooled"]["detection"] == 0.0


def test_fit_classifier_rejects_run_spanning_groups(monkeypatch):
    monkeypatch.setattr(h.cv, "fit_level2", lambda rows, names: None)
    # One run with a co-location window and a training window: the row filter
    # would keep the training window; the run check must refuse it.
    dev = [row("mixture-concurrent-0p5", "training", "mixed"), row("training", "training", "mixed", 1),
           row("training", "training", "honest"), row("idle")]
    with pytest.raises(AssertionError):
        h.fit_classifier(dev, "co-location", [])
    _, manifest = h.fit_classifier(dev[2:], "co-location", [])
    assert manifest == ["honest", "run-idle"]


def test_no_evasion_rows_keep_only_honest_and_negatives():
    dev = [row("training", "training", "honest"), row("lora", "training", "lora"),
           row("idle"), row("serving-decode"), row("steady-gemm"),
           row("mixture-concurrent-0p5", "training", "co"), row("mixture-timeshared-0p5", "training", "ts"),
           row("cadence-jitter-0p4", "training", "jit"), row("powercap-200", "training", "cap"),
           row("cadence-pad-0p5", "training", "pad"),
           # A run that spans groups is dropped whole.
           row("mixture-concurrent-0p25", "training", "mixed"), row("training", "training", "mixed", 1)]
    runs = {r["run_id"] for r in h.no_evasion_fit_rows(dev)}
    assert runs == {"honest", "lora", "run-idle", "run-serving-decode", "run-steady-gemm"}
    # Every honest and negative run of the co-location-held-out fit is kept.
    clf_runs = {r["run_id"] for r in h.classifier_fit_rows(dev, "co-location")}
    assert {q for q in clf_runs if not q.startswith(("ts", "jit", "cap", "pad", "mixed"))} <= runs


def test_duty_parse():
    assert h.duty("mixture-timeshared-0p25-slot60") == 0.25
    assert h.duty("mixture-timeshared-0p75") == 0.75


def run_dict(nominal, none_mask, eps):
    rows = [{"t0_s": t - 5.0, "t1_s": float(t)} for t in range(5, 5 + len(none_mask))]
    return {"rows": rows, "eps": eps, "nominal": nominal, "none": np.array(none_mask),
            "condition": "mixture-timeshared-0p5", "duty": 0.5}


def ep(first, last):
    return {"first": first, "last": last, "steps": np.array([[first, first + 0.3]])}


def test_gap_rates_pooled_fallback():
    runs = {"a": run_dict(10, [True, True, False, False], []),
            "b": run_dict(10, [True, True, True, True], []),
            "c": run_dict(10, [False] * 4, [])}
    flags = {"a": np.array([1, 0, 1, 1], bool), "b": np.array([1, 1, 1, 0], bool),
             "c": np.ones(4, bool)}
    g = m.gap_rates(runs, flags)
    assert g["a"] == 0.5 and g["b"] == 0.75
    assert g["c"] == pytest.approx(4 / 6)  # window-weighted pool of a and b


def test_evaluate_episodes_takes_smallest_bound_per_duration():
    n = 200
    # Run x: every episode detected; run y (same nominal): every episode missed.
    flags_x = np.zeros(n, bool)
    eps_x, eps_y = [], []
    for k in range(10):
        s = 10 + 15 * k
        flags_x[s:s + 4] = True
        eps_x.append(ep(s + 2.5, s + 7.5))
        eps_y.append(ep(s + 2.5, s + 7.5))
    runs = {"x": run_dict(5.0, [False] * n, eps_x), "y": run_dict(5.0, [True] * n, eps_y)}
    out = m.evaluate_episodes(runs, {"x": flags_x, "y": np.zeros(n, bool)})
    assert out["cells"]["x"]["headline"]["rate"] == 1.0
    assert out["cells"]["y"]["headline"]["rate"] == 0.0
    assert out["lmin_bounds_by_duration"] == {"5.0": 0.0}
    assert out["Lmin_s"] is None
    assert sum(out["counts"].values()) == 20
