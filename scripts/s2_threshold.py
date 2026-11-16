#!/usr/bin/env python3
"""Task 6.1, step 1: freeze the S2 classifier's threshold before it scores any arm.

The S2 classifier is the development-only level-2 fit with time-sharing and the
three S2 arms held out (pre-registration §7). Its threshold follows §3: the
(1 - alpha) quantile of its P(training) on the 5 s-hop calibration declared
windows; a window is flagged iff its score is strictly greater. Stability under
a run-level bootstrap of the two calibration runs is reported, not acted on.
R1's threshold is the one frozen in Phase 2 and is not recomputed here.

Reads development and calibration windows only; never the test split.
Writes results/ceiling/threshold.json, which must be committed before
scripts/s2_ceiling.py runs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import ceiling as c  # noqa: E402
import corpus  # noqa: E402
import declaration as d  # noqa: E402
import heldout as h  # noqa: E402
import prereg  # noqa: E402

OUT = REPO / "results" / "ceiling" / "threshold.json"
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

    names = h.frozen_r1()["features"]
    model, manifest = h.fit_classifier(dev, h.S2_FIT, names)
    scored = set(h.S2_ARMS) | {c.CONTROL}
    assert not scored & set(manifest), sorted(scored & set(manifest))
    entry = h.threshold_entry(d.by_run(cal_rows, model.p_training(d.matrix(cal_rows, names))))
    group, runs = h.CLASSIFIER_FITS[h.S2_FIT]
    print(f"S2 classifier threshold {entry['threshold']:.6g} "
          f"({entry['frac_calibration_zero']:.4f} of calibration windows at 0)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_by": "scripts/s2_threshold.py", "spec_task": "6.1",
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
        "classifier": {"score": "p_training", "held_out_group": group,
                       "held_out_runs": list(runs), **entry},
        "fit_manifest": manifest,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
