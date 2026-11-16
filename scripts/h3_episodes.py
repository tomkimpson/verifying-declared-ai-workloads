#!/usr/bin/env python3
"""Task 5.3: H3 episode detection and delay on the held-out day (F4).

Episodes come from the test episode runs' `step` and `train-slot` events
(labels, not current; the sanctioned exception in code/prereg.py), by the
definition §3 check 5 verifies against amodo episodes-recovery.json. The
3-of-5 monitor runs over each run's 1 s-hop windows for R1 (headline), the
classifier (dev-only fit, time-sharing held out) and their OR, at the frozen
thresholds. Every episode gets one class (pre-registration §5): truncated,
unscorable, pre-alarmed, detected by onset, missed. Per cell (one run each):
P with pre-alarmed credit under the per-run gap guard, the onset-only metric,
the complete-step-restricted metric, the within-run lower bound, and median
delay to the window-end onset. Lmin from each cell's headline bound.

Verification (spec task 5.3): the episode count per run matches
episodes-recovery.json (192 in all), and the five classes sum to it.
Writes results/monitor/episodes.json.
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
import monitor as m  # noqa: E402
import prereg  # noqa: E402

OUT = REPO / "results" / "monitor" / "episodes.json"
EXPECTED_TOTAL = 192
DETECTORS = ("r1", "classifier", "or")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    if not git("ls-files", str(h.THRESHOLDS.relative_to(REPO))):
        raise SystemExit("results/heldout/thresholds.json is not committed")
    thr = h.load_thresholds()
    assert thr["preregistration_sha256"] == prereg.PREREG_SHA256
    dev_header, dev = d.load_windows("development", 5)
    cal_header, cal = d.load_windows("calibration", 5)
    test_header, test = d.load_windows("test", 1)
    for hd in (dev_header, cal_header, test_header):
        assert hd["corpus_commit"] == corpus.CORPUS_COMMIT
    det = h.fit_h3(dev, cal, thr)
    runs = h.episode_runs(test)
    total = sum(len(v["eps"]) for v in runs.values())
    assert total == EXPECTED_TOTAL, total

    out = {}
    for k in DETECTORS:
        fl = {rid: h.h3_flags(h.h3_scores(det, v["rows"]), det["thresholds"])[k]
              for rid, v in runs.items()}
        out[k] = m.evaluate_episodes(runs, fl)
        assert sum(out[k]["counts"].values()) == EXPECTED_TOTAL
        print(f"{k}: counts {out[k]['counts']}  Lmin {out[k]['Lmin_s']}", flush=True)
        for rid, c in out[k]["cells"].items():
            print(f"   {rid:42s} {c['nominal_s']:6.1f}s duty {c['duty']:.2f} G={c['gap_flag_rate']:.2f} "
                  f"{c['headline_metric']:5s} {c['headline']['rate']:.3f} lb {c['headline']['lower']:.3f} "
                  f"D {c['delay']['median_s']}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_by": "scripts/h3_episodes.py", "spec_task": "5.3",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "thresholds_repo_commit": thr["repo_commit"],
        "corpus_commit": corpus.CORPUS_COMMIT,
        "preregistration_sha256": prereg.PREREG_SHA256,
        "windows_repo_commit": {"development": dev_header["repo_commit"],
                                "calibration": cal_header["repo_commit"],
                                "test": test_header["repo_commit"]},
        "rule": {"need": m.NEED, "of": m.OF, "rearm": m.REARM, "grace_s": m.GRACE_S,
                 "truncated_frac": m.TRUNCATED_FRAC, "gap_guard": m.GAP_GUARD,
                 "lmin_level": m.LMIN_LEVEL},
        "thresholds": det["thresholds"], "headline_detector": "r1",
        "classifier_fit": h.H3_FIT, "classifier_fit_manifest": det["manifest"],
        "bound_note": "within-run: every cell is one run, so no between-run variance enters it",
        "verification": {"episodes_per_run": {rid: len(v["eps"]) for rid, v in runs.items()},
                         "recovery_per_run": {rid: v["n_recovery"] for rid, v in runs.items()},
                         "nominal_s": {rid: v["nominal"] for rid, v in runs.items()},
                         "recovery_median_s": {rid: v["recovery_median_s"] for rid, v in runs.items()},
                         "total": total, "expected_total": EXPECTED_TOTAL},
        "detectors": out,
    }, indent=1, default=float) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
