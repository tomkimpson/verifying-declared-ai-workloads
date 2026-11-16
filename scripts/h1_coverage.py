#!/usr/bin/env python3
"""Task 4.1: T1, detection per strategy group at levels 1 and 2.

Level 1: the replica (fitted on the 300 replication traces) scores honest
training and every strategy group. Level 2: for each strategy group, the
two-stage forest fitted with that group held out (pre-registration §2.1)
scores it. "Seen" reference: for each run of a group with >= 2 runs, the
forest fitted with the group included but that run held out scores the run.
Padding has one run and no seen reference.

Reads hop-5 windows from all three splits (test through the pre-registration
guard: the held-out time-sharing group has test-day runs). Writes
results/coverage/t1.json.
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
import prereg  # noqa: E402
from detectors import fit_replica  # noqa: E402

OUT = REPO / "results" / "coverage" / "t1.json"
SPLITS = ("development", "calibration", "test")
HOP_S = 5

#: Gargiulo & Kulp (arXiv 2609.00309) Table 1, worst row for the hardened
#: unseen-strategy detector: diluted LoRA, 48-88% across families (spec §2).
GK_WORST_HELD_OUT = {"strategy": "diluted LoRA", "detection_range": [0.48, 0.88],
                     "detector": "hardened, unseen strategy",
                     "source": "arXiv 2609.00309 Table 1"}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def runs_of(rows: list[dict]) -> list[str]:
    return sorted({r["run_id"] for r in rows})


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    rows, headers = [], {}
    for split in SPLITS:
        headers[split], r = d.load_windows(split, HOP_S)
        assert headers[split]["corpus_commit"] == corpus.CORPUS_COMMIT
        rows += r
    names = json.loads((REPO / "results/declaration/choice.json").read_text())["frozen"]["features"]
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(groups.group(r["condition"]), []).append(r)

    rep_names, replica = fit_replica()
    assert sorted(rep_names) == sorted(names), "replica and corpus features differ"
    level1 = {}
    for g in ("honest",) + cv.STRATEGY_GROUPS:
        level1[g] = cv.summarise(by_group[g], replica.predict(d.matrix(by_group[g], rep_names)))

    level2, seen, manifests = {}, {}, {}
    for g in cv.STRATEGY_GROUPS:
        fit = cv.fit_rows(rows, g)
        fit_runs = runs_of(fit)
        assert not any(groups.group(r["condition"]) == g for r in fit), g
        manifests[f"level2/{g}"] = fit_runs
        model = cv.fit_level2(fit, names)
        level2[g] = cv.summarise(by_group[g], model.predict(d.matrix(by_group[g], names)))
        print(f"level 2, {g} held out: {level2[g]['pooled']['detection']}", flush=True)

        g_runs = runs_of(by_group[g])
        if len(g_runs) < 2:
            seen[g] = None
            continue
        seen_rows, seen_pred = [], []
        for run in g_runs:
            fit = cv.fit_rows(rows, None, frozenset({run}))
            assert run not in runs_of(fit)
            manifests[f"seen/{g}/{run}"] = runs_of(fit)
            scored = [r for r in by_group[g] if r["run_id"] == run]
            seen_rows += scored
            seen_pred += list(cv.fit_level2(fit, names).predict(d.matrix(scored, names)))
        seen[g] = cv.summarise(seen_rows, np.array(seen_pred, dtype=object))
        print(f"seen, {g}: {seen[g]['pooled']['detection']}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_by": "scripts/h1_coverage.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "corpus_commit": corpus.CORPUS_COMMIT,
        "preregistration_sha256": prereg.PREREG_SHA256,
        "windows": {s: {k: headers[s][k] for k in ("window_s", "hop_s", "repo_commit")}
                    for s in SPLITS},
        "features": names,
        "flag": "predict == 'training'; rate over windows labelled training, partial excluded",
        "level1_replica": level1,
        "level2_held_out": level2,
        "level2_seen_leave_one_run_out": seen,
        "gk_reference": GK_WORST_HELD_OUT,
        "fit_manifests": manifests,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
