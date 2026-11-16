#!/usr/bin/env python3
"""Task 13.3: the session check (exploratory; amendments 2026-11-08g, 2026-11-13).

G&K's design on our corpus. The reference is 10 windows from one ~4-min block of
one decode run, as G&K's per-label recordings are. Is the false-positive rate on
other days' decode runs much higher than on the rest of the same run? If so,
a reference recorded in one session partly recognises the session, not the work.

Per decode run, four reference blocks (window offsets 0, 40, 80, 120; 10 windows
taken every 5th, 25 s apart). The rule is leave-one-out and same-model: z is
flagged only if it outscores every held-out reference window under that window's
own model (fitted on the other 9). Scores from models fitted on different sets
are never compared. The nominal false-positive rate is about 1/11.

Two scorers: an isolation forest (primary, as notes/.../ours_session.py) and a
nearest-neighbour diagnostic (standardised minimum distance, as
notes/.../gk_specificity.py).

The verdict rule was fixed before this ran (amendment 2026-11-13). S and O are
the means over the 3 decode runs of the same-run and other-day FP; b =
max(S, 1/11). Low iff O <= 2b and O <= 0.20, else high.

Successor of notes/2026-11-07-framing-checks/ours_session.py, which execs
stress2.py and so loads the test split. This reads development and calibration
only. Writes results/heldout/session_check.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import corpus  # noqa: E402
import declaration as d  # noqa: E402

OUT = REPO / "results" / "heldout" / "session_check.json"
THRESHOLDS = REPO / "results" / "heldout" / "thresholds.json"
SPLITS = ("development", "calibration")
HOP_S = 5
DECODE = "serving-decode"
#: Expected decode runs (amendment 2026-11-08g): three sessions.
DECODE_RUNS = ("panel-core-serving-decode-r1", "panel-core-serving-decode-r4",
               "panel-serving-decode")
#: Reference blocks, as ours_session.py: start offsets, span, stride.
BLOCK_STARTS = (0, 40, 80, 120)
BLOCK_SPAN, BLOCK_STRIDE = 50, 5
N_REF = BLOCK_SPAN // BLOCK_STRIDE
IFOREST_KWARGS = {"n_estimators": 200, "random_state": 0, "n_jobs": -1}
#: Verdict rule (amendment 2026-11-13).
NOMINAL = 1 / (N_REF + 1)
RATIO, CEILING = 2.0, 0.20
PRIMARY = "iforest"


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def iforest(a: np.ndarray):
    m = IsolationForest(**IFOREST_KWARGS).fit(a)
    return lambda z: -m.score_samples(z)


def nearest(a: np.ndarray):
    sd = np.maximum(a.std(0, ddof=1), 1e-9)
    return lambda z: np.min(np.linalg.norm((z[:, None, :] - a[None]) / sd, axis=2), 1)


SCORERS = {"iforest": iforest, "nearest": nearest}


def loo_flagger(make, A: np.ndarray):
    """Flag z iff it outscores every held-out reference row under that row's own model."""
    models = [make(np.delete(A, i, 0)) for i in range(len(A))]
    held = [m(A[i:i + 1])[0] for i, m in enumerate(models)]
    return lambda z: np.min([m(z) - c for m, c in zip(models, held)], axis=0) > 0


def verdict(same: float, other: float) -> dict:
    b = max(same, NOMINAL)
    low = other <= RATIO * b and other <= CEILING
    return {"S": same, "O": other, "baseline_b": b, "limit": min(RATIO * b, CEILING),
            "verdict": "low" if low else "high"}


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    names = json.loads(THRESHOLDS.read_text())["features"]
    rows, headers = [], {}
    for split in SPLITS:
        headers[split], r = d.load_windows(split, HOP_S)
        assert headers[split]["corpus_commit"] == corpus.CORPUS_COMMIT
        rows += r
    assert all(r["split"] in SPLITS for r in rows), "a test window was read"
    X = lambda rr: d.matrix(rr, names)  # noqa: E731

    dec = [r for r in rows if r["condition"] == DECODE]
    runs = tuple(sorted({r["run_id"] for r in dec}))
    assert runs == DECODE_RUNS, runs
    by_run = {rid: sorted([r for r in dec if r["run_id"] == rid],
                          key=lambda r: (r["interval"], r["t0_s"])) for rid in runs}
    sessions = {rid: by_run[rid][0]["session"] for rid in runs}
    assert len(set(sessions.values())) == len(runs), sessions

    dev = [r for r in rows if r["split"] == "development"]
    tr = lambda sub: [r for r in dev if sub in r["run_id"] and r["label"] == "training"]  # noqa: E731
    targets = {"rho.5 (unpaced)": tr("concurrent-0p5"), "rho.25": tr("concurrent-0p25"),
               "rho.75": tr("concurrent-0p75"),
               "lora (dev+cal)": [r for r in rows if r["condition"] == "lora"
                                  and r["label"] == "training"],
               "training (dev)": [r for r in dev if r["condition"] == "training"
                                  and r["label"] == "training"]}
    x_runs = {rid: X(w) for rid, w in by_run.items()}
    x_tgt = {k: X(v) for k, v in targets.items()}

    out = {}
    for sname, make in SCORERS.items():
        per_run = {}
        for rid in runs:
            w = by_run[rid]
            blocks = []
            for st in BLOCK_STARTS:
                idx = list(range(st, st + BLOCK_SPAN, BLOCK_STRIDE))
                assert idx[-1] < len(w), (rid, st, len(w))  # never skip a block silently
                flag = loo_flagger(make, x_runs[rid][idx])
                rest = np.setdiff1d(np.arange(len(w)), idx)
                other = {o: float(flag(x_runs[o]).mean()) for o in runs if o != rid}
                blocks.append({
                    "start": st, "n_ref": len(idx),
                    "same_run_fp": float(flag(x_runs[rid][rest]).mean()),
                    "other_run_fp": other, "other_run_fp_mean": float(np.mean(list(other.values()))),
                    "detection": {k: float(flag(v).mean()) for k, v in x_tgt.items()}})
            same_r = float(np.mean([b["same_run_fp"] for b in blocks]))
            other_r = float(np.mean([b["other_run_fp_mean"] for b in blocks]))
            per_run[rid] = {"session": sessions[rid], "n_blocks": len(blocks),
                            "same_run_fp": same_r, "other_run_fp": other_r,
                            "other_run_fp_by_run": {o: float(np.mean([b["other_run_fp"][o]
                                                                      for b in blocks]))
                                                    for o in runs if o != rid},
                            "detection": {k: float(np.mean([b["detection"][k] for b in blocks]))
                                          for k in targets},
                            "verdict": verdict(same_r, other_r), "blocks": blocks}
            print(f"{sname:8s} {rid:30s} {sessions[rid]} same {same_r:.3f} other {other_r:.3f} "
                  f"det {per_run[rid]['detection']}", flush=True)
        S = float(np.mean([p["same_run_fp"] for p in per_run.values()]))
        O = float(np.mean([p["other_run_fp"] for p in per_run.values()]))
        out[sname] = {"pooled": verdict(S, O), "per_run": per_run}
        print(f"{sname}: pooled {out[sname]['pooled']}", flush=True)

    OUT.write_text(json.dumps({
        "generated_by": "scripts/session_check.py", "spec_task": "13.3",
        "preregistered": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "corpus_commit": corpus.CORPUS_COMMIT,
        "windows": {s: {k: headers[s][k] for k in ("window_s", "hop_s", "repo_commit")}
                    for s in SPLITS},
        "splits_read": list(SPLITS),
        "features": names,
        "design": {"block_starts": BLOCK_STARTS, "block_span": BLOCK_SPAN,
                   "block_stride": BLOCK_STRIDE, "n_ref": N_REF,
                   "order": "decode windows of a run sorted by (interval, t0_s)",
                   "rule": "flag z iff it outscores every held-out reference window under "
                           "that window's own model (fitted on the other 9)",
                   "iforest_kwargs": IFOREST_KWARGS,
                   "nearest": "minimum Euclidean distance after scaling by the reference sd"},
        "verdict_rule": {"source": "spec amendment 2026-11-13", "primary": PRIMARY,
                         "nominal": NOMINAL, "ratio": RATIO, "ceiling": CEILING,
                         "rule": "b = max(S, nominal); low iff O <= ratio*b and O <= ceiling"},
        "verdict_applied": out[PRIMARY]["pooled"]["verdict"],
        "notes": ["exploratory, not pre-registered",
                  "rates are fractions of windows",
                  "the verdict rests on 3 runs x 4 blocks x 2 other runs, and a run's "
                  "windows score almost alike, so these are about 6 near-binary trials",
                  "same-run 'rest' includes windows interleaved with the reference block, "
                  "which overlap reference windows in time if window_s > hop_s",
                  "nearest-neighbour is a diagnostic; the forest's verdict is applied"],
        "decode_runs": {rid: {"session": sessions[rid], "split": by_run[rid][0]["split"],
                              "n_windows": len(by_run[rid])} for rid in runs},
        "targets": {k: {"n_windows": len(v), "runs": sorted({r["run_id"] for r in v})}
                    for k, v in targets.items()},
        "results": out,
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}; verdict applied ({PRIMARY}): "
          f"{out[PRIMARY]['pooled']['verdict']}")


if __name__ == "__main__":
    main()
