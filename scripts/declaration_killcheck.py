#!/usr/bin/env python3
"""Phase 2b, task 2.5: the R1 kill check, run once on the frozen choice.

Refits R1 exactly as results/declaration/choice.json froze it (and asserts the
threshold reproduces), then scores every dev window outside the declared
workload: every hidden-training configuration (co-location included), honest
training and the non-declared negatives. Per-window detection counts windows
labelled `training` (>= 1 complete step); `partial` windows are excluded and
counted (spec §3). Rates over all windows are given beside, since time-shared
runs are mostly non-training windows.

§8 co-location clause: at alpha <= 0.05, R1 detecting co-location in < 50%
of its training windows kills R1 as a headline. The 3-alpha clause is read
from the 2.2 result (like-for-like, with the literal outcome beside it;
amendment 2026-10-02b), read from the evidence frozen in choice.json so the
verdict cannot mix a later heldout.json with the earlier choice. No parameter
may change after this script runs.
Writes results/declaration/killcheck.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import declaration as d  # noqa: E402
import groups  # noqa: E402

CHOICE = REPO / "results" / "declaration" / "choice.json"
OUT = REPO / "results" / "declaration" / "killcheck.json"
#: §8: co-location detection below this kills R1 as a headline.
KILL_DETECTION_MIN = 0.50


def main() -> None:
    if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD"]).returncode:
        raise SystemExit("working tree is dirty: commit first")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    choice = json.loads(CHOICE.read_text())
    f = choice["frozen"]
    dev_header, dev = d.load_windows("development", int(f["hop_s"]))
    cal_header, cal = d.load_windows("calibration", int(f["hop_s"]))
    fit_rows = d.select(dev, groups.DECLARED)
    cal_rows = d.select(cal, groups.DECLARED)
    assert sorted({r["run_id"] for r in fit_rows}) == f["fit_runs"]
    assert sorted({r["run_id"] for r in cal_rows}) == f["calibration_runs"]

    model = d.OneClass(f["family"]).fit(d.matrix(fit_rows, f["features"], f["amplitude"]))
    thr = d.threshold(model.score(d.matrix(cal_rows, f["features"], f["amplitude"])), f["alpha"])
    assert thr == f["threshold"], (thr, f["threshold"])

    scored = [r for r in dev if r["condition"] not in groups.DECLARED]
    # R1a's amplitude is normalised within capture, so score whole captures together.
    flags = model.score(d.matrix(scored, f["features"], f["amplitude"])) > thr

    per_run, by_group = {}, defaultdict(dict)
    for rid in sorted({r["run_id"] for r in scored}):
        idx = [i for i, r in enumerate(scored) if r["run_id"] == rid]
        rows = [scored[i] for i in idx]
        fl = flags[idx][np.argsort([r["t0_s"] for r in rows])]
        lab = np.array([r["label"] for r in sorted(rows, key=lambda r: r["t0_s"])])
        tr = fl[lab == "training"]
        entry = {"condition": rows[0]["condition"], "group": groups.group(rows[0]["condition"]),
                 "session": rows[0]["session"], "n_windows": len(rows),
                 "n_training": int(len(tr)), "n_partial": int((lab == "partial").sum()),
                 "detection_training": float(tr.mean()) if len(tr) else None,
                 "detection_training_ci95": list(d.block_bootstrap_ci(tr)) if len(tr) else None,
                 "flag_rate_all": float(fl.mean())}
        per_run[rid] = entry
        by_group[entry["group"]][rid] = tr

    group_rates = {}
    for g, runs in by_group.items():
        runs = {k: v for k, v in runs.items() if len(v)}
        if not runs:
            continue
        lo, hi = d.run_bootstrap_ci(runs)
        group_rates[g] = {"runs": sorted(runs), "detection_training":
                          float(np.concatenate(list(runs.values())).mean()), "ci95": [lo, hi]}

    coloc = group_rates["co-location"]["detection_training"]
    # The 3-alpha clause as frozen with the choice, not a later heldout.json.
    c4 = choice["evidence"]["check4"]
    verdict = {
        "colocation_detection": coloc, "colocation_min": KILL_DETECTION_MIN,
        "colocation_clause_fires": coloc < KILL_DETECTION_MIN,
        "fp_3alpha_like_for_like_fires": c4["kill_3alpha_fires"],
        "fp_3alpha_literal_fires": c4["literal"]["kill_3alpha_fires"],
    }
    verdict["kill_fires"] = (verdict["colocation_clause_fires"]
                             or verdict["fp_3alpha_like_for_like_fires"])
    OUT.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/declaration_killcheck.py", "spec_task": "2.5",
        "repo_commit": commit, "choice_repo_commit": choice["repo_commit"],
        "corpus_commit": choice["corpus_commit"],
        "windows_repo_commit": {"development": dev_header["repo_commit"],
                                "calibration": cal_header["repo_commit"]},
        "frozen": f,
        "verdict": verdict, "by_group": group_rates, "per_run": per_run,
    }, indent=1) + "\n")
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
