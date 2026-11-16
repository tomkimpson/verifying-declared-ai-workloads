#!/usr/bin/env python3
"""Task 13.6: envelope CV of declared inference, split by workload (amendment 2026-11-13d).

The mechanism appendix compares the unthrottled run's median `env_cv` with that of
"declared inference" (results/coverage/mechanism.json), which pools two decode runs
with one prefill run. This splits that pool: the median `env_cv` of development
decode, development prefill, each declared run, and the unthrottled co-location
run. The windows and quantile rule are those of scripts/h1_mechanism.py; the pooled
median must reproduce mechanism.json's. Descriptive.

Reads development windows only. Writes results/coverage/cv_by_workload.json.
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

OUT = REPO / "results" / "coverage" / "cv_by_workload.json"
MECH = REPO / "results" / "coverage" / "mechanism.json"
FEATURE = "env_cv"
UNPACED = "panel-mixture-concurrent-0p5"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def median(rows: list[dict]) -> float:
    return float(np.quantile([r["features"][FEATURE] for r in rows], 0.5))


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    header, dev = d.load_windows("development", 5)
    assert header["corpus_commit"] == corpus.CORPUS_COMMIT
    mech = json.loads(MECH.read_text())["distributions"]

    pooled = [r for r in dev if r["condition"] in cv.INFERENCE_CONDITIONS]
    assert sorted({r["run_id"] for r in pooled}) == mech["declared inference"]["runs"]
    ref = mech["declared inference"]["features"][FEATURE]["quantiles"]["0.5"]
    assert median(pooled) == ref, (median(pooled), ref)
    unp = [r for r in dev if r["run_id"] == UNPACED]
    assert median(unp) == mech[UNPACED]["features"][FEATURE]["quantiles"]["0.5"]

    by_workload = {w: {"runs": sorted({r["run_id"] for r in rr}), "n_windows": len(rr),
                       "median": median(rr)}
                   for w in cv.INFERENCE_CONDITIONS
                   for rr in [[r for r in pooled if r["condition"] == w]]}
    by_run = {run: median([r for r in pooled if r["run_id"] == run])
              for run in sorted({r["run_id"] for r in pooled})}
    for k, v in by_workload.items():
        print(f"{k:16s} median {FEATURE} {v['median']:.3f} ({v['n_windows']} windows, {v['runs']})")
    print(f"pooled {median(pooled):.3f} (= mechanism.json); unthrottled run {median(unp):.3f}")

    OUT.write_text(json.dumps({
        "generated_by": "scripts/mechanism_cv_split.py", "spec_task": "13.6",
        "preregistered": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "corpus_commit": corpus.CORPUS_COMMIT,
        "windows": {k: header[k] for k in ("window_s", "hop_s", "repo_commit")},
        "splits_read": ["development"],
        "feature": FEATURE, "quantile": 0.5,
        "notes": ["descriptive; splits the 'declared inference' pool of mechanism.json",
                  "pooled and unthrottled medians asserted equal to mechanism.json"],
        "mechanism_repo_commit": json.loads(MECH.read_text())["repo_commit"],
        "pooled_declared_inference": {"runs": sorted({r["run_id"] for r in pooled}),
                                      "n_windows": len(pooled), "median": median(pooled)},
        "by_workload": by_workload, "by_run": by_run,
        "unthrottled": {"run": UNPACED, "n_windows": len(unp), "median": median(unp)},
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
