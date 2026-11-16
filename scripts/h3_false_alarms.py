#!/usr/bin/env python3
"""Task 5.2: H3 false alarms on the held-out day (T2).

The 3-of-5 monitor (code/monitor.py) runs over the 1 s-hop windows of each
test background capture, for R1, the classifier (dev-only fit, time-sharing
held out) and their OR, at the thresholds frozen in
results/heldout/thresholds.json (5 s-hop calibration rule, reused at the 1 s
hop). Backgrounds are never pooled: declared inference (serving-decode),
idle, non-AI busy (steady-gemm + memory-bound).

F is onsets per monitored hour with an exact Poisson (Garwood) upper bound,
labelled a Poisson model; E is the summed manifest `scored_s`. A is the
fraction of windows after which the alarm is active. Spec §8's finding
threshold reads the declared-inference point estimates.

Verification (spec task 5.2): E equals the windowed scored duration of the
background's captures, and a stuck-positive synthetic detector gives one
onset per capture and A close to 1. Writes results/monitor/false_alarms.json.
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
import declaration as d  # noqa: E402
import heldout as h  # noqa: E402
import monitor as m  # noqa: E402
import prereg  # noqa: E402

OUT = REPO / "results" / "monitor" / "false_alarms.json"
MANIFEST = REPO / "results" / "windows" / "manifest.json"
BACKGROUNDS = {"declared inference": ("serving-decode",),
               "idle": ("idle",),
               "non-AI busy": ("steady-gemm", "memory-bound")}
#: Scored hours stated in the pre-registration (§0, §5), checked against E.
PREREG_HOURS = {"declared inference": 0.475, "idle": 8.35, "non-AI busy": 0.950}
#: Spec §8: H3 reports unusable false alerts if every detector exceeds either.
FINDING_F_PER_H, FINDING_A = 10.0, 0.10
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
    manifest = {c["capture_id"]: c for c in json.loads(MANIFEST.read_text())["captures"]}

    out, checks = {}, {}
    for bg, conds in BACKGROUNDS.items():
        rows = [r for r in test if r["condition"] in conds]
        caps = sorted({r["capture_id"] for r in rows})
        expected_caps = sorted(c for c, e in manifest.items()
                               if e["split"] == "test" and e["condition"] in conds)
        assert caps == expected_caps, (bg, caps, expected_caps)
        exposure = float(sum(manifest[c]["scored_s"] for c in caps))
        n_win = {c: sum(r["capture_id"] == c for r in rows) for c in caps}
        assert all(n_win[c] == manifest[c]["windows"]["1"] for c in caps), bg
        assert abs(exposure / 3600 - PREREG_HOURS[bg]) < 0.005, (bg, exposure)

        fl = h.h3_flags(h.h3_scores(det, rows), det["thresholds"])
        out[bg] = {"conditions": list(conds), "captures": caps,
                   "window_flag_rate": {k: float(v.mean()) for k, v in fl.items()},
                   **{k: m.false_alarms(m.sequences(rows, fl[k]), exposure) for k in DETECTORS}}
        stuck = m.false_alarms(m.sequences(rows, np.ones(len(rows), bool)), exposure)
        assert all(v == 1 for v in stuck["onsets_per_capture"].values()) and stuck["A"] > 0.99, bg
        checks[bg] = {"exposure_s": exposure, "exposure_h": exposure / 3600,
                      "prereg_hours": PREREG_HOURS[bg],
                      "windows_match_manifest": True,
                      "stuck_positive": {"onsets_per_capture": stuck["onsets_per_capture"],
                                         "A": stuck["A"],
                                         "ok": all(v == 1 for v in stuck["onsets_per_capture"].values())
                                         and stuck["A"] > 0.99}}
        print(f"{bg}: " + "  ".join(f"{k} F={out[bg][k]['F_per_h']:.2f}/h A={out[bg][k]['A']:.3f}"
                                    for k in DETECTORS), flush=True)

    decl = out["declared inference"]
    finding = {k: {"F_per_h": decl[k]["F_per_h"], "A": decl[k]["A"],
                   "exceeds": decl[k]["F_per_h"] > FINDING_F_PER_H or decl[k]["A"] > FINDING_A}
               for k in DETECTORS}
    verdict = {"thresholds": {"F_per_h": FINDING_F_PER_H, "A": FINDING_A},
               "per_detector": finding,
               "unusable_finding": all(v["exceeds"] for v in finding.values())}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_by": "scripts/h3_false_alarms.py", "spec_task": "5.2",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "thresholds_repo_commit": thr["repo_commit"],
        "corpus_commit": corpus.CORPUS_COMMIT,
        "preregistration_sha256": prereg.PREREG_SHA256,
        "windows_repo_commit": {"development": dev_header["repo_commit"],
                                "calibration": cal_header["repo_commit"],
                                "test": test_header["repo_commit"]},
        "rule": {"need": m.NEED, "of": m.OF, "rearm": m.REARM,
                 "state": "carries across interval breaks; resets per capture"},
        "thresholds": det["thresholds"], "headline_detector": "r1",
        "classifier_fit": h.H3_FIT, "classifier_fit_manifest": det["manifest"],
        "bound_label": "Poisson model (exact one-sided 95%, Garwood)",
        "A_interval_note": "approximate; onset blocks of 600 windows under-cover on ~830-window runs",
        "backgrounds": out, "section8_finding": verdict, "verification": checks,
    }, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
