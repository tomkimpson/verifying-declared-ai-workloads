"""H2 and H3 detectors on the held-out day (spec Phase 5; pre-registration §2-§4).

Both detectors are fitted on development windows only (spec D1). The level-2
classifier is the H1 recipe (`coverage.fit_rows`, the §2.1 class map) refitted
on development rows, with the scored strategy group held out, and scores
`p_training`. R1 is the frozen one-class model of `results/declaration/choice.json`.

Thresholds follow pre-registration §3: the (1 - alpha) quantile of the
detector's score on the 5 s-hop calibration declared windows
(`panel-core-idle-r4`, `panel-core-serving-decode-r4`, pooled); a window is
flagged iff its score is strictly greater. Calibration windows are never
fitted on.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

import corpus
import coverage as cv
import declaration as d
import groups
import windows
from detectors import TwoStageForest

REPO = Path(__file__).resolve().parents[1]
CHOICE = REPO / "results" / "declaration" / "choice.json"
THRESHOLDS = REPO / "results" / "heldout" / "thresholds.json"
RECOVERY = corpus.CORPUS_REPO / "results" / "panel" / "episodes-recovery.json"

#: Shared target alpha (pre-registration §3).
ALPHA = 0.05
#: Calibration declared runs the thresholds are read from (pre-registration §3).
CALIBRATION_RUNS = ("panel-core-idle-r4", "panel-core-serving-decode-r4")
#: The classifier fits, one per held-out group (pre-registration §2.1):
#: name -> (held-out group, extra held-out runs).
FITS = {g: (g, ()) for g in cv.STRATEGY_GROUPS}
#: S2's three arms, all development (pre-registration §7).
S2_ARMS = ("panel-training-arm1", "panel-training-arm2", "panel-training-arm3")
#: S2's classifier: time-sharing and the three arms held out (pre-registration
#: §7). Kept out of FITS, which the Phase 5 scripts iterate per strategy group.
S2_FIT = "s2"
CLASSIFIER_FITS = {**FITS, S2_FIT: ("time-sharing", S2_ARMS)}
#: Declared workloads with test windows (pre-registration §4).
TEST_DECLARED = ("serving-decode", "idle")
#: The fit behind X' (pre-registration §4) and the fit H3 uses (§2.1).
H2_HEADLINE_FIT = "co-location"
H3_FIT = "time-sharing"


def frozen_r1() -> dict:
    return json.loads(CHOICE.read_text())["frozen"]


def fit_r1(dev: list[dict]) -> tuple[d.OneClass, dict]:
    """R1 refitted exactly as frozen in Phase 2 (pre-registration §2.2)."""
    f = frozen_r1()
    rows = d.select(dev, groups.DECLARED)
    assert sorted({r["run_id"] for r in rows}) == f["fit_runs"]
    return d.OneClass(f["family"]).fit(d.matrix(rows, f["features"], f["amplitude"])), f


def classifier_fit_rows(dev: list[dict], fit: str) -> list[dict]:
    """Development rows of one classifier fit: the §2.1 class map with the
    fit's group and runs held out."""
    group, runs = CLASSIFIER_FITS[fit]
    assert all(r["split"] == "development" for r in dev)
    return cv.fit_rows(dev, group, frozenset(runs))


def no_evasion_fit_rows(dev: list[dict]) -> list[dict]:
    """Development rows of the E2 ablation: the §2.1 class map with every
    evasion group held out, so only honest training and negatives remain
    (exploratory; amendment 2026-10-14)."""
    assert all(r["split"] == "development" for r in dev)
    hidden_runs = {r["run_id"] for r in dev if groups.is_hidden(r["condition"])}
    return [r for r in cv.fit_rows(dev, None) if r["run_id"] not in hidden_runs]


def fit_no_evasion(dev: list[dict], names: list[str]) -> tuple[TwoStageForest, list[str]]:
    """(forest, sorted fit-run manifest) of the E2 ablation; recipe and settings
    as `fit_classifier`."""
    rows = no_evasion_fit_rows(dev)
    return cv.fit_level2(rows, names), sorted({r["run_id"] for r in rows})


def fit_classifier(dev: list[dict], fit: str, names: list[str]) -> tuple[TwoStageForest, list[str]]:
    """(forest, sorted fit-run manifest). Asserts, by run ID, that the manifest
    holds no held-out run and no run with any window in the held-out group.
    `cv.fit_rows` filters row by row; this checks whole runs. Callers check the
    manifest against the rows they score (h2_compare)."""
    rows = classifier_fit_rows(dev, fit)
    group, held = CLASSIFIER_FITS[fit]
    manifest = sorted({r["run_id"] for r in rows})
    group_runs = {r["run_id"] for r in dev if groups.group(r["condition"]) == group}
    bad = sorted(set(manifest) & (group_runs | set(held)))
    assert not bad, (fit, bad)
    return cv.fit_level2(rows, names), manifest


def calibration_rows(cal: list[dict]) -> list[dict]:
    rows = [r for r in cal if r["run_id"] in CALIBRATION_RUNS]
    assert sorted({r["run_id"] for r in rows}) == sorted(CALIBRATION_RUNS)
    return rows


def threshold_stability(scores_by_run: dict[str, np.ndarray], alpha: float = ALPHA,
                        n_boot: int = d.N_BOOT, seed: int = d.SEED) -> dict:
    """The threshold under a run-level bootstrap of the calibration runs
    (reported, not acted on; pre-registration §3). With two runs a resample is
    one run twice or both, so this is the spread between the runs' quantiles."""
    runs = sorted(scores_by_run)
    rng = np.random.default_rng(seed)
    t = [d.threshold(np.concatenate([scores_by_run[runs[i]]
                                     for i in rng.integers(0, len(runs), len(runs))]), alpha)
         for _ in range(n_boot)]
    lo, hi = float(np.quantile(t, 0.025)), float(np.quantile(t, 0.975))
    per_run = {r: d.threshold(scores_by_run[r], alpha) for r in runs}
    return {"ci95": [lo, hi], "per_run_quantile": per_run,
            "ratio_hi_lo": (hi / lo) if lo > 0 else None}


def threshold_entry(scores_by_run: dict[str, np.ndarray]) -> dict:
    """A fit's frozen threshold (§3) with the T4 alpha grid, calibration
    quantiles and run-level stability, from its calibration scores by run."""
    pooled = np.concatenate(list(scores_by_run.values()))
    return {"threshold": d.threshold(pooled, ALPHA),
            "alpha_grid": {str(a): d.threshold(pooled, a) for a in d.ALPHAS},
            "calibration_quantiles": {q: float(np.quantile(pooled, q))
                                      for q in (0.5, 0.9, 0.95, 0.99)},
            "frac_calibration_zero": float((pooled == 0).mean()),
            "stability": threshold_stability(scores_by_run)}


def load_thresholds() -> dict:
    return json.loads(THRESHOLDS.read_text())


def flags(scores: np.ndarray, thr: float) -> np.ndarray:
    """Strictly greater (pre-registration §3)."""
    return np.asarray(scores) > thr


def roc(neg: np.ndarray, pos: np.ndarray) -> dict:
    """Descriptive ROC: detection of `pos` against false-positive rate on
    `neg` at every distinct score threshold (flag iff score > t)."""
    neg, pos = np.sort(neg), np.sort(pos)
    ts = np.concatenate([[-np.inf], np.unique(np.concatenate([neg, pos]))])
    fpr = 1.0 - np.searchsorted(neg, ts, side="right") / len(neg)
    tpr = 1.0 - np.searchsorted(pos, ts, side="right") / len(pos)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


# --- H3 detectors (pre-registration §5) ---------------------------------------

def fit_h3(dev: list[dict], cal: list[dict], thr: dict) -> dict:
    """R1 and the time-sharing-held-out classifier, refitted on development
    windows, with the frozen thresholds (re-derived and checked)."""
    r1, f = fit_r1(dev)
    names = f["features"]
    clf, manifest = fit_classifier(dev, H3_FIT, names)
    assert manifest == thr["fit_manifests"][H3_FIT]
    cal_rows = calibration_rows(cal)
    t_r1 = thr["r1"]["threshold"]
    t_clf = thr["classifier"]["fits"][H3_FIT]["threshold"]
    assert d.threshold(r1.score(d.matrix(cal_rows, names)), ALPHA) == t_r1
    assert d.threshold(clf.p_training(d.matrix(cal_rows, names)), ALPHA) == t_clf
    return {"r1": r1, "classifier": clf, "names": names, "manifest": manifest,
            "thresholds": {"r1": t_r1, "classifier": t_clf}}


def h3_scores(det: dict, rows: list[dict]) -> dict[str, np.ndarray]:
    x = d.matrix(rows, det["names"])
    return {"r1": det["r1"].score(x), "classifier": det["classifier"].p_training(x)}


def h3_flags(scores: dict[str, np.ndarray], thresholds: dict[str, float]) -> dict[str, np.ndarray]:
    """Window flags of R1, the classifier and their OR (pre-registration §5)."""
    out = {k: flags(scores[k], thresholds[k]) for k in ("r1", "classifier")}
    out["or"] = out["r1"] | out["classifier"]
    return out


# --- H2 summaries (pre-registration §4) -------------------------------------

def as_pred(flags: np.ndarray) -> np.ndarray:
    return np.where(flags, "training", "not flagged").astype(object)


def detection(rows: list[dict], flags: np.ndarray) -> dict:
    """Per-run and pooled detection on `training` windows, by split and pooled."""
    out = {"all": cv.summarise(rows, as_pred(flags))}
    splits = sorted({r["split"] for r in rows})
    if len(splits) > 1:
        for s in splits:
            idx = [i for i, r in enumerate(rows) if r["split"] == s]
            out[s] = cv.summarise([rows[i] for i in idx], as_pred(flags[idx]))
    return out


def false_positives(rows: list[dict], flags: np.ndarray) -> dict:
    """Achieved false-positive rate per declared workload, all windows."""
    out = {}
    for w in TEST_DECLARED:
        idx = [i for i, r in enumerate(rows) if r["condition"] == w]
        fr = d.by_run([rows[i] for i in idx], flags[idx])
        method, ci = cv.pooled_interval(fr)
        out[w] = {"n_runs": len(fr), "n_windows": len(idx),
                  "fpr": float(flags[idx].mean()), "interval_method": method,
                  "fpr_ci95": list(ci),
                  "per_run": {k: float(v.mean()) for k, v in fr.items()}}
    return out


# --- H3 episodes (pre-registration §5) ----------------------------------------

def duty(condition: str) -> float:
    return float(re.match(r"mixture-timeshared-(\d)p(\d+)", condition).expand(r"\1.\2"))


def _run_episodes(arc, run_id: str) -> list[dict]:
    """Each episode of a run: first and last step edge, and its steps."""
    cap = arc.open(run_id)
    steps = cap.events("step")
    slots = cap.events("train-slot")
    out = []
    for slot, edges in zip(slots, windows.episode_edges(steps, slots)):
        if edges is None:
            continue
        inside = np.array([(e["scope_start_s"], e["scope_end_s"]) for e in steps
                           if slot["scope_start_s"] <= e["scope_start_s"] <= slot["scope_end_s"]])
        out.append({"first": edges[0], "last": edges[1], "steps": inside})
    return out


def episode_runs(test: list[dict]) -> dict[str, dict]:
    """The test episode runs of episodes-recovery.json: each run's 1 s-hop
    rows (time order), episodes from its `step` events (labels, not current;
    code/prereg.py), nominal duration (median episode duration), duty and
    gap-window mask. Asserts each run's episode count matches the file."""
    rec = json.loads(RECOVERY.read_text())
    arc = corpus.archive()
    runs = {}
    for rr in rec["runs"]:
        rid = rr["run_id"]
        rows = sorted((r for r in test if r["run_id"] == rid), key=lambda r: r["t0_s"])
        assert rows and len({r["capture_id"] for r in rows}) == 1, rid
        eps = _run_episodes(arc, rid)
        assert len(eps) == rr["n_episodes"], (rid, len(eps), rr["n_episodes"])
        runs[rid] = {"rows": rows, "eps": eps,
                     "nominal": float(np.median([e["last"] - e["first"] for e in eps])),
                     "none": np.array([r["label"] == "none" for r in rows]),
                     "condition": rows[0]["condition"], "duty": duty(rows[0]["condition"]),
                     "n_recovery": rr["n_episodes"],
                     "recovery_median_s": rr["episode_s_median"]}
    return runs
