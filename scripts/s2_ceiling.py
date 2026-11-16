#!/usr/bin/env python3
"""Task 6.1: S2, the TOST of discarded-update against genuine training.

Per pre-registration §7: R1 (frozen in Phase 2) and the S2 classifier
(development only, time-sharing and the three arms held out; threshold frozen in
results/ceiling/threshold.json) score every 5 s-hop window of `training` arm 1,
`training-discarded` and `training-copying`, `partial` windows included. The TOST
runs on the per-window flag rate: genuine - discarded (primary) and genuine -
copying (secondary), at delta = 0.05 for every detector. The positive control,
genuine against `panel-mixture-timeshared-0p5`, must reject equivalence;
a detector whose control does not is reported as uninformative.

n = 1 run per arm: every interval covers within-run variability only. R1's
outcome was visible in results/declaration/killcheck.json before the
pre-registration (flag rate 1.0 on all three arms); the classifier's half is blind.

Reads development and calibration windows only; never the test split.
Writes results/ceiling/s2.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import ceiling as c  # noqa: E402
import corpus  # noqa: E402
import declaration as d  # noqa: E402
import heldout as h  # noqa: E402
import prereg  # noqa: E402

THRESHOLD = REPO / "results" / "ceiling" / "threshold.json"
OUT = REPO / "results" / "ceiling" / "s2.json"
HOP_S = 5
ARMS = {"genuine": c.GENUINE, "discarded": "panel-training-arm2",
        "copying": "panel-training-arm3", "control": c.CONTROL}
COMPARISONS = {"primary": "discarded", "secondary": "copying", "positive_control": "control"}
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    frozen = json.loads(THRESHOLD.read_text())
    assert git("log", "-1", "--format=%H", "--", str(THRESHOLD.relative_to(REPO))), \
        "threshold.json must be committed before scoring"
    dev_header, dev = d.load_windows("development", HOP_S)
    cal_header, cal = d.load_windows("calibration", HOP_S)
    for hd in (dev_header, cal_header):
        assert hd["corpus_commit"] == corpus.CORPUS_COMMIT
    cal_rows = h.calibration_rows(cal)

    r1, f = h.fit_r1(dev)
    names = f["features"]
    clf, manifest = h.fit_classifier(dev, h.S2_FIT, names)
    assert manifest == frozen["fit_manifest"]
    assert not set(ARMS.values()) & set(manifest)
    t = {"r1": f["threshold"], "classifier": frozen["classifier"]["threshold"]}
    assert d.threshold(r1.score(d.matrix(cal_rows, names, f["amplitude"])), h.ALPHA) == t["r1"]
    assert d.threshold(clf.p_training(d.matrix(cal_rows, names)), h.ALPHA) == t["classifier"]

    runs, flags = {}, {"r1": {}, "classifier": {}}
    for arm, run in ARMS.items():
        rows = sorted((r for r in dev if r["run_id"] == run), key=lambda r: r["t0_s"])
        assert rows, run
        x = d.matrix(rows, names, f["amplitude"])
        scores = {"r1": r1.score(x), "classifier": clf.p_training(x)}
        runs[arm] = {"run_id": run, "condition": rows[0]["condition"],
                     "session": rows[0]["session"], "n_windows": len(rows),
                     "labels": dict(Counter(r["label"] for r in rows)), "detectors": {}}
        for det, s in scores.items():
            fl = h.flags(s, t[det])
            flags[det][arm] = fl
            runs[arm]["detectors"][det] = {
                "flag_rate": float(fl.mean()),
                "flag_rate_ci95_within_run": list(d.block_bootstrap_ci(fl)),
                "score_quantiles": {str(q): float(np.quantile(s, q)) for q in QUANTILES},
                "scores": s.tolist()}

    tests = {}
    for det in flags:
        g = flags[det]["genuine"]
        res = {name: {"comparison": f"genuine - {arm}",
                      **c.tost(g, flags[det][arm], c.SEEDS[c.GENUINE], c.SEEDS[ARMS[arm]])}
               for name, arm in COMPARISONS.items()}
        control_rejects = not res["positive_control"]["equivalent"]
        res["control_rejects"] = control_rejects
        res["verdict"] = (("equivalent" if res["primary"]["equivalent"] else "not equivalent")
                          if control_rejects else "uninformative")
        tests[det] = res
        p = res["primary"]
        print(f"{det}: genuine {p['rate_a']:.3f} discarded {p['rate_b']:.3f} "
              f"diff {p['difference']:+.3f} 90% [{p['ci90'][0]:+.3f}, {p['ci90'][1]:+.3f}]; "
              f"control diff {res['positive_control']['difference']:+.3f} "
              f"{res['positive_control']['ci90']} -> {res['verdict']}")

    OUT.write_text(json.dumps({
        "generated_by": "scripts/s2_ceiling.py", "spec_task": "6.1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "corpus_commit": corpus.CORPUS_COMMIT,
        "preregistration_sha256": prereg.PREREG_SHA256,
        "windows_repo_commit": {"development": dev_header["repo_commit"],
                                "calibration": cal_header["repo_commit"]},
        "splits_read": ["development", "calibration"],
        "hop_s": HOP_S,
        "window_set": "all windows of each run, partial included (pre-registration §7)",
        "thresholds": t,
        "threshold_source": {"r1": "results/declaration/choice.json (frozen, Phase 2)",
                             "classifier": "results/ceiling/threshold.json"},
        "classifier_fit_manifest": manifest,
        "tost": {"delta": c.DELTA, "interval": "90% percentile (two one-sided tests at 5%)",
                 "bootstrap": f"independent circular block bootstrap per run, "
                              f"{d.BLOCK_WINDOWS}-window blocks, {c.N_BOOT} resamples",
                 "seeds": c.SEEDS},
        "notes": ["n = 1 run per arm: the test covers within-run variability only.",
                  "R1's S2 outcome was visible in results/declaration/killcheck.json "
                  "before the pre-registration; the classifier's half is blind.",
                  "Score distributions are descriptive."],
        "runs": runs,
        "tests": tests,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
