#!/usr/bin/env python3
"""Task 5.1, step 1: freeze the H2/H3 thresholds before any test window is scored.

For R1 (frozen in Phase 2) and each development-only level-2 classifier fit
(one per held-out strategy group), the threshold is the (1 - alpha) quantile
of the detector's score on the 5 s-hop calibration declared windows
(pre-registration §3), at the shared alpha and at the T4 alpha grid. Each
threshold's stability under a run-level bootstrap of the two calibration runs
is reported, not acted on.

Reads development and calibration windows only; never the test split.
Writes results/heldout/thresholds.json, which must be committed before
scripts/h2_compare.py, scripts/h3_false_alarms.py, scripts/h3_episodes.py or
scripts/t4_robustness.py run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import corpus  # noqa: E402
import declaration as d  # noqa: E402
import heldout as h  # noqa: E402
import prereg  # noqa: E402

OUT = h.THRESHOLDS
HOP_S = 5


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    dev_header, dev = d.load_windows("development", HOP_S)
    cal_header, cal = d.load_windows("calibration", HOP_S)
    for hd in (dev_header, cal_header):
        assert hd["corpus_commit"] == corpus.CORPUS_COMMIT
    cal_rows = h.calibration_rows(cal)

    r1, f = h.fit_r1(dev)
    r1_scores = r1.score(d.matrix(cal_rows, f["features"], f["amplitude"]))
    r1_entry = h.threshold_entry(d.by_run(cal_rows, r1_scores))
    assert r1_entry["threshold"] == f["threshold"], (r1_entry["threshold"], f["threshold"])

    names = f["features"]
    classifiers, manifests = {}, {}
    for fit in h.FITS:
        model, manifests[fit] = h.fit_classifier(dev, fit, names)
        s = model.p_training(d.matrix(cal_rows, names))
        classifiers[fit] = {"held_out_group": h.FITS[fit][0],
                            "held_out_runs": list(h.FITS[fit][1]),
                            **h.threshold_entry(d.by_run(cal_rows, s))}
        print(f"classifier, {fit} held out: threshold {classifiers[fit]['threshold']:.4f}",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_by": "scripts/h2_thresholds.py", "spec_task": "5.1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "corpus_commit": corpus.CORPUS_COMMIT,
        "preregistration_sha256": prereg.PREREG_SHA256,
        "windows_repo_commit": {"development": dev_header["repo_commit"],
                                "calibration": cal_header["repo_commit"]},
        "splits_read": ["development", "calibration"],
        "alpha": h.ALPHA,
        "rule": "(1 - alpha) quantile (numpy default) of the 5 s-hop calibration "
                "declared windows; flag iff score > threshold",
        "calibration_runs": list(h.CALIBRATION_RUNS),
        "features": names,
        "r1": {"family": f["family"], "score": "-score_samples", **r1_entry},
        "classifier": {"score": "p_training", "fits": classifiers},
        "fit_manifests": manifests,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
