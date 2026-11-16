#!/usr/bin/env python3
"""Task 8.4: E1, the operating point (exploratory; not pre-registered).

Scores the co-location runs, development declared windows and test declared
windows with two detectors, each at the default decision tau_1/2 = 0.5 and at
its calibration-day threshold tau_alpha (the H2 rule, pre-registration §3):

- the H2 classifier: the level-2 forest with co-location held out, fitted on
  development windows exactly as scripts/h2_compare.py fits it, with its
  frozen tau_alpha from results/heldout/thresholds.json;
- the level-1 replica (detectors.fit_replica), with tau_alpha set by the same
  rule on the same calibration declared windows;
- the E2 no-evasion fit (task 9.2, amendment 2026-10-14): the H2 classifier's
  recipe on development honest training and negatives only
  (heldout.fit_no_evasion), with tau_alpha by the same rule.

Reports detection per co-location run, achieved false-positive rates per
declared workload (development and test), the descriptive ROC AUC of
co-location training windows against test declared windows (pooled, and
against test decode alone), and the per-window scores behind F6. Writes results/heldout/operating_point.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import corpus  # noqa: E402
import declaration as d  # noqa: E402
import groups  # noqa: E402
import heldout as h  # noqa: E402
import prereg  # noqa: E402
from detectors import fit_replica  # noqa: E402

OUT = REPO / "results" / "heldout" / "operating_point.json"
SPLITS = ("development", "calibration", "test")
HOP_S = 5
#: The default decision of a two-stage forest: stage two's binary class.
TAU_HALF = 0.5
#: Decimals kept for the per-window scores F6 plots.
SCORE_DECIMALS = 4


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def auc(neg: np.ndarray, pos: np.ndarray) -> float:
    """Descriptive ROC AUC of `pos` against `neg`."""
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def at(rows_co: list[dict], s_co: np.ndarray, decl: dict, tau: float) -> dict:
    """Detection per co-location run and achieved FPR per declared set at tau."""
    det = h.detection(rows_co, h.flags(s_co, tau))["all"]
    return {"tau": tau,
            "detection_per_run": {k: v["detection"] for k, v in det["per_run"].items()},
            "detection_pooled": det["pooled"],
            "fpr": {name: h.false_positives(rows, h.flags(s, tau))
                    if name == "test" else
                    {w: float(h.flags(s[[r["condition"] == w for r in rows]], tau).mean())
                     for w in sorted({r["condition"] for r in rows})}
                    for name, (rows, s) in decl.items()}}


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    thr = h.load_thresholds()
    assert thr["preregistration_sha256"] == prereg.PREREG_SHA256
    rows, headers = [], {}
    for split in SPLITS:
        headers[split], r = d.load_windows(split, HOP_S)
        assert headers[split]["corpus_commit"] == corpus.CORPUS_COMMIT
        rows += r
    dev = [r for r in rows if r["split"] == "development"]
    cal = h.calibration_rows([r for r in rows if r["split"] == "calibration"])
    co = [r for r in rows if groups.group(r["condition"]) == h.H2_HEADLINE_FIT]
    assert {r["split"] for r in co} == {"development"}
    decl = {"development": d.select(dev, groups.DECLARED),
            "test": [r for r in rows if r["split"] == "test"
                     and r["condition"] in h.TEST_DECLARED]}
    names = h.frozen_r1()["features"]

    clf, manifest = h.fit_classifier(dev, h.H2_HEADLINE_FIT, names)
    assert manifest == thr["fit_manifests"][h.H2_HEADLINE_FIT]
    rep_names, replica = fit_replica()
    assert sorted(rep_names) == sorted(names), "replica and corpus features differ"
    no_ev, no_ev_manifest = h.fit_no_evasion(dev, names)
    assert not set(no_ev_manifest) - set(manifest), "no-evasion fit has a run the H2 fit lacks"
    scorers = {"classifier": lambda rr: clf.p_training(d.matrix(rr, names)),
               "replica": lambda rr: replica.p_training(d.matrix(rr, rep_names)),
               "no_evasion": lambda rr: no_ev.p_training(d.matrix(rr, names))}

    co_tr = np.array([r["label"] == "training" for r in co])
    is_test_decode = np.array([r["condition"] == "serving-decode" for r in decl["test"]])
    result, scores = {}, {}
    for k, score in scorers.items():
        s_cal = score(cal)
        tau_a = d.threshold(s_cal, h.ALPHA)
        if k == "classifier":
            assert tau_a == thr["classifier"]["fits"][h.H2_HEADLINE_FIT]["threshold"]
        s_co = score(co)
        s_decl = {name: (rr, score(rr)) for name, rr in decl.items()}
        result[k] = {
            "tau_half": at(co, s_co, s_decl, TAU_HALF),
            "tau_alpha": at(co, s_co, s_decl, tau_a),
            "frac_calibration_zero": float((s_cal == 0).mean()),
            "frac_stage1_non_ai": {
                run: float((s_co[[r["run_id"] == run for r in co]] == 0).mean())
                for run in sorted({r["run_id"] for r in co})},
            "median_training_score": {
                run: float(np.median(s_co[co_tr & np.array([r["run_id"] == run for r in co])]))
                for run in sorted({r["run_id"] for r in co})},
            "roc_auc_vs_test_declared": auc(s_decl["test"][1], s_co[co_tr]),
            "roc_auc_per_run_vs_test_declared": {
                run: auc(s_decl["test"][1],
                         s_co[co_tr & np.array([r["run_id"] == run for r in co])])
                for run in sorted({r["run_id"] for r in co})},
            "roc_auc_per_run_vs_test_decode": {
                run: auc(s_decl["test"][1][is_test_decode],
                         s_co[co_tr & np.array([r["run_id"] == run for r in co])])
                for run in sorted({r["run_id"] for r in co})},
        }
        scores[k] = {
            "co-location": {run: np.round(s_co[co_tr & np.array([r["run_id"] == run for r in co])],
                                          SCORE_DECIMALS).tolist()
                            for run in sorted({r["run_id"] for r in co})},
            **{f"{name} declared": {w: np.round(s[[r["condition"] == w for r in rr]],
                                                SCORE_DECIMALS).tolist()
                                    for w in sorted({r["condition"] for r in rr})}
               for name, (rr, s) in s_decl.items()},
        }
        print(f"{k}: tau_alpha {tau_a:.4f}; per run at tau_1/2 "
              f"{result[k]['tau_half']['detection_per_run']}; at tau_alpha "
              f"{result[k]['tau_alpha']['detection_per_run']}; "
              f"AUC {result[k]['roc_auc_vs_test_declared']:.4f}", flush=True)

    # The verification clause: the classifier at tau_alpha reproduces h2.json's X'.
    h2 = json.loads((REPO / "results" / "heldout" / "h2.json").read_text())
    assert result["classifier"]["tau_alpha"]["detection_per_run"] == \
        h2["headline"]["X_prime_per_run"], "classifier at tau_alpha does not reproduce X'"

    OUT.write_text(json.dumps({
        "generated_by": "scripts/h2_operating_point.py", "spec_task": "8.4, 9.2",
        "preregistered": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "thresholds_repo_commit": thr["repo_commit"],
        "corpus_commit": corpus.CORPUS_COMMIT,
        "windows": {s: {k: headers[s][k] for k in ("window_s", "hop_s", "repo_commit")}
                    for s in SPLITS},
        "alpha": h.ALPHA,
        "flag": "score > tau; detection over co-location windows labelled training, "
                "partial excluded; FPR over all declared windows",
        "notes": ["exploratory (spec E1, amendment 2026-10-11); not pre-registered",
                  "development declared windows are in the fit data of the H2 "
                  "classifier and of no_evasion (in-sample scores); the replica "
                  "never saw them",
                  "co-location is development-only attack data",
                  "ROC AUC is descriptive: co-location training windows (all runs "
                  "pooled, and per run) against test declared windows, and per run "
                  "against test decode windows alone",
                  "no_evasion is E2 (task 9.2, amendment 2026-10-14): the H2 recipe "
                  "fitted on development honest training and negatives only; "
                  "exploratory"],
        "classifier_fit": h.H2_HEADLINE_FIT, "classifier_fit_manifest": manifest,
        "no_evasion_fit_manifest": no_ev_manifest,
        **result, "scores": scores,
    }) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
