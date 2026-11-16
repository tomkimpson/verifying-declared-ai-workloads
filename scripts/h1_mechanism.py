#!/usr/bin/env python3
"""Task 4.2: what separates the missed unpaced co-location run from the caught
paced ones (descriptive; amendment 2026-10-02g).

1. Impurity importances of both stages of the level-2 forest with co-location
   held out (the T1 fit, refitted identically).
2. The top stage-2 features' per-window values for declared inference, honest
   training, time-shared training windows (development) and each co-location
   run, with each co-location run's median placed as a percentile of the
   inference and honest-training distributions. Stage-2 P(training) per run.
3. The step-locked envelope profile and its CV (code/mechanism.py) per run:
   honest training, time-shared and each co-location run, each against its
   random-alignment null (amendment 2026-10-02g).

The 5 Hz item of 4.2 is in scripts/h1_5hz_stage_{a,b}.py. The scored windows
and profiles are development data only; the forest is the pre-registered
all-days T1 fit, read through the pre-registration guard.
Writes results/coverage/mechanism.json.
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
import mechanism as mech  # noqa: E402
import prereg  # noqa: E402

OUT = REPO / "results" / "coverage" / "mechanism.json"
SPLITS = ("development", "calibration", "test")
TOP_K = 4
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

COLOCATION = ("panel-mixture-concurrent-0p25", "panel-mixture-concurrent-0p5",
              "panel-mixture-concurrent-0p75")
#: Step-profile runs. Honest training: the four plain dev `training` runs
#: outside the jitter/padding baselines. Time-shared: the three dev runs.
PROFILE_RUNS = {
    "honest": ("panel-training-arm1", "panel-core-training-r1", "panel-training-15min",
               "panel-training-15min-nosampler"),
    "time-sharing": ("panel-mixture-timeshared-0p25", "panel-mixture-timeshared-0p5",
                     "panel-mixture-timeshared-0p75"),
    "co-location": COLOCATION,
}


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    rows = []
    for split in SPLITS:
        rows += d.load_windows(split, 5)[1]
    names = json.loads((REPO / "results/declaration/choice.json").read_text())["frozen"]["features"]
    model = cv.fit_level2(cv.fit_rows(rows, "co-location"), names)
    imp = {f"stage{k}": dict(sorted(zip(names, map(float, m.feature_importances_)),
                                    key=lambda p: -p[1]))
           for k, m in ((1, model.stage1), (2, model.stage2))}
    top = list(imp["stage2"])[:TOP_K]

    dev = [r for r in rows if r["split"] == "development"]
    sets = {
        "declared inference": [r for r in dev if r["condition"] in cv.INFERENCE_CONDITIONS],
        "honest training": [r for r in dev if groups.group(r["condition"]) == "honest"
                            and r["label"] == "training"],
        "time-shared training": [r for r in dev if groups.group(r["condition"]) == "time-sharing"
                                 and r["label"] == "training"],
    }
    sets.update({run: [r for r in dev if r["run_id"] == run] for run in COLOCATION})

    dist = {}
    for name, rs in sets.items():
        x = d.matrix(rs, names)
        entry = {"n_windows": len(rs), "runs": sorted({r["run_id"] for r in rs}),
                 "p_training_median": float(np.median(model.p_training(x))),
                 "features": {}}
        for f in top:
            v = x[:, names.index(f)]
            entry["features"][f] = {"quantiles": dict(zip(map(str, QUANTILES),
                                                          map(float, np.quantile(v, QUANTILES)))),
                                    "values": [float(f"{a:.6g}") for a in v]}
        dist[name] = entry
    ref = {k: {f: np.array(dist[k]["features"][f]["values"]) for f in top}
           for k in ("declared inference", "honest training")}
    for run in COLOCATION:
        dist[run]["median_percentile_in"] = {
            k: {f: float((ref[k][f] < dist[run]["features"][f]["quantiles"]["0.5"]).mean())
                for f in top} for k in ref}

    arc = corpus.archive()
    profiles = {}
    for grp, runs in PROFILE_RUNS.items():
        for run in runs:
            assert arc.open(run).entry["split"] == "development", run
            p = mech.step_locked(run, arc)
            profiles[run] = {"group": grp, **p}
            print(f"{run:34s} CV {p['cv']:.3f} {np.round(p['ci95'], 3)}  "
                  f"null p95 {p['null_cv_p95']:.3f}  within-step {p['within_step']['cv']:.3f} "
                  f"(null p95 {p['within_step']['null_cv_p95']:.3f})", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated_by": "scripts/h1_mechanism.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "corpus_commit": corpus.CORPUS_COMMIT,
        "preregistration_sha256": prereg.PREREG_SHA256,
        "forest": "level 2, co-location held out (as results/coverage/t1.json)",
        "importances": imp,
        "top_stage2_features": top,
        "distributions": dist,
        "step_profiles": {
            "statistic": "CV of the mean envelope profile over one median step duration "
                         "centred on each step end (amendment 2026-10-02g)",
            "interval": "95% basic circular block bootstrap over consecutive segments",
            "null": "profile CV at uniformly random times, same counts, N_NULL draws",
            "block_segments": mech.BLOCK_SEGMENTS, "n_boot": mech.N_BOOT,
            "n_null": mech.N_NULL, "seed": mech.SEED,
            "note": ("chosen after the PR #4 review of the fixed-window dip depth, on "
                     "development data; not pre-registered (pre-registration §11)"),
            "runs": profiles,
        },
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
