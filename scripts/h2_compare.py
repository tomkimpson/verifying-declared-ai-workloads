#!/usr/bin/env python3
"""Task 5.1: H2, R1 against the level-2 classifier under the shared threshold rule.

Both detectors are fitted on development windows only and thresholded by the
calibration-day rule frozen in results/heldout/thresholds.json (refits must
reproduce every threshold exactly). For each strategy group, R1 and the
classifier fit holding that group out score every run of the group, on every
split where it exists; time-sharing is reported dev and test apart, then
pooled. Each detector's achieved false-positive rate on test declared windows
is reported per workload. The ROC is descriptive.

Co-location is development-only attack data, never fitted or selected on by
R1: it is never "day-held-out". Reads test windows through the
pre-registration guard. Writes results/heldout/h2.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import corpus  # noqa: E402
import coverage as cv  # noqa: E402
import declaration as d  # noqa: E402
import groups  # noqa: E402
import heldout as h  # noqa: E402
import prereg  # noqa: E402

OUT = REPO / "results" / "heldout" / "h2.json"
SPLITS = ("development", "calibration", "test")
HOP_S = 5
TEST_DECLARED = h.TEST_DECLARED


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def require_frozen_thresholds() -> dict:
    """The thresholds file must be committed, unchanged, before test is scored."""
    rel = str(h.THRESHOLDS.relative_to(REPO))
    if not git("ls-files", rel):
        raise SystemExit(f"{rel} is not committed: run scripts/h2_thresholds.py and commit it")
    t = h.load_thresholds()
    assert t["preregistration_sha256"] == prereg.PREREG_SHA256
    return t


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    thr = require_frozen_thresholds()
    rows, headers = [], {}
    for split in SPLITS:
        headers[split], r = d.load_windows(split, HOP_S)
        assert headers[split]["corpus_commit"] == corpus.CORPUS_COMMIT
        rows += r
    dev = [r for r in rows if r["split"] == "development"]
    cal_rows = h.calibration_rows([r for r in rows if r["split"] == "calibration"])
    test_decl = [r for r in rows if r["split"] == "test" and r["condition"] in TEST_DECLARED]
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(groups.group(r["condition"]), []).append(r)

    r1, f = h.fit_r1(dev)
    names = f["features"]
    t_r1 = thr["r1"]["threshold"]
    assert d.threshold(r1.score(d.matrix(cal_rows, names)), h.ALPHA) == t_r1 == f["threshold"]

    result = {"r1": {"threshold": t_r1, "detection": {}, "test_fpr": None},
              "classifier": {}}
    result["r1"]["test_fpr"] = h.false_positives(
        test_decl, h.flags(r1.score(d.matrix(test_decl, names)), t_r1))
    r1_scores = {}
    for g in cv.STRATEGY_GROUPS:
        r1_scores[g] = r1.score(d.matrix(by_group[g], names))
        result["r1"]["detection"][g] = h.detection(by_group[g], h.flags(r1_scores[g], t_r1))

    manifests, clf_scores = {}, {}
    for fit in h.FITS:
        model, manifests[fit] = h.fit_classifier(dev, fit, names)
        assert manifests[fit] == thr["fit_manifests"][fit], fit
        t = thr["classifier"]["fits"][fit]["threshold"]
        assert d.threshold(model.p_training(d.matrix(cal_rows, names)), h.ALPHA) == t, fit
        group = h.FITS[fit][0]
        held = set(manifests[fit]) & {r["run_id"] for r in by_group[group]}
        assert not held, (fit, held)
        s = model.p_training(d.matrix(by_group[group], names))
        clf_scores[fit] = s
        result["classifier"][fit] = {
            "held_out_group": group, "threshold": t,
            "detection": h.detection(by_group[group], h.flags(s, t)),
            "test_fpr": h.false_positives(
                test_decl, h.flags(model.p_training(d.matrix(test_decl, names)), t)),
        }
        print(f"{fit}: R1 {result['r1']['detection'][group]['all']['pooled']['detection']:.3f}"
              f"  classifier {result['classifier'][fit]['detection']['all']['pooled']['detection']:.3f}",
              flush=True)

    # Headline: Z (R1) against X' (classifier, co-location held out).
    co = h.H2_HEADLINE_FIT
    hl = {"Z_r1": result["r1"]["detection"][co]["all"]["pooled"],
          "X_prime_classifier": result["classifier"][co]["detection"]["all"]["pooled"],
          "Z_per_run": {k: v["detection"] for k, v in
                        result["r1"]["detection"][co]["all"]["per_run"].items()},
          "X_prime_per_run": {k: v["detection"] for k, v in
                              result["classifier"][co]["detection"]["all"]["per_run"].items()},
          "r1_test_fpr": {w: v["fpr"] for w, v in result["r1"]["test_fpr"].items()},
          "classifier_test_fpr": {w: v["fpr"] for w, v in
                                  result["classifier"][co]["test_fpr"].items()},
          "colocation_split": "development only (attack data; never fitted or selected on by R1)"}

    # Descriptive ROC: co-location training windows against test declared windows.
    co_tr = np.array([r["label"] == "training" for r in by_group[co]])
    neg_r1 = r1.score(d.matrix(test_decl, names))
    co_model, _ = h.fit_classifier(dev, co, names)
    neg_clf = co_model.p_training(d.matrix(test_decl, names))
    roc = {"description": "descriptive only; no operating point is read off it",
           "negatives": "test declared windows (serving-decode, idle)",
           "positives": "co-location windows labelled training (development)",
           "r1": h.roc(neg_r1, r1_scores[co][co_tr]),
           "classifier": h.roc(neg_clf, clf_scores[co][co_tr])}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_by": "scripts/h2_compare.py", "spec_task": "5.1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "thresholds_repo_commit": thr["repo_commit"],
        "corpus_commit": corpus.CORPUS_COMMIT,
        "preregistration_sha256": prereg.PREREG_SHA256,
        "windows": {s: {k: headers[s][k] for k in ("window_s", "hop_s", "repo_commit")}
                    for s in SPLITS},
        "alpha": h.ALPHA,
        "flag": "score > threshold; detection over windows labelled training, partial excluded",
        "headline": hl, **result, "roc": roc, "fit_manifests": manifests,
    }, indent=1) + "\n")
    print(json.dumps({k: v for k, v in hl.items() if not isinstance(v, dict) or "detection" not in v},
                     indent=1, default=str))
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
