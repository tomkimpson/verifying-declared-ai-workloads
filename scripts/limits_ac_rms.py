#!/usr/bin/env python3
"""Task 13.2: AC RMS of the unpaced co-location run and of development decode
(exploratory, descriptive; a disclosure for the Limits).

Per run: the median, interquartile range and count of the windows' `ac_rms_a`
(AC RMS current in amperes over each 5 s window, from the AC-coupled probe), and
the ratio of the unpaced run's median to each development decode run's median.
The runs come from different sessions, and the probe gain may not be comparable
across sessions; re-clamping of the probe between sessions is unchecked.

Reads development and calibration windows only; the calibration decode run is
listed as context. Writes results/heldout/ac_rms.json.
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

OUT = REPO / "results" / "heldout" / "ac_rms.json"
SPLITS = ("development", "calibration")
HOP_S = 5
UNPACED = "panel-mixture-concurrent-0p5"
#: Development decode runs (spec 13.2), and the calibration decode run as context.
DEV_DECODE = ("panel-serving-decode", "panel-core-serving-decode-r1")
CAL_DECODE = ("panel-core-serving-decode-r4",)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def describe(rows: list[dict]) -> dict:
    v = np.array([r["ac_rms_a"] for r in rows])
    q25, q50, q75 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"session": rows[0]["session"], "split": rows[0]["split"],
            "condition": rows[0]["condition"], "n_windows": len(v),
            "median_a": float(q50), "q25_a": float(q25), "q75_a": float(q75)}


def main() -> None:
    if git("status", "--porcelain"):
        raise SystemExit("working tree is dirty (or has untracked files): commit first")
    rows, headers = [], {}
    for split in SPLITS:
        headers[split], r = d.load_windows(split, HOP_S)
        assert headers[split]["corpus_commit"] == corpus.CORPUS_COMMIT
        rows += r
    assert all(r["split"] in SPLITS for r in rows), "a test window was read"

    def run(rid: str, label: str | None = None) -> list[dict]:
        rr = [r for r in rows if r["run_id"] == rid and (label is None or r["label"] == label)]
        assert rr, rid
        return rr

    runs = {UNPACED: describe(run(UNPACED, "training"))}
    for rid in DEV_DECODE + CAL_DECODE:
        runs[rid] = describe(run(rid))
    u = runs[UNPACED]["median_a"]
    ratio = {rid: u / runs[rid]["median_a"] for rid in DEV_DECODE + CAL_DECODE}
    for rid, s in runs.items():
        print(f"{rid:32s} {s['session']} median {s['median_a']:.3f} A "
              f"[{s['q25_a']:.3f}, {s['q75_a']:.3f}] n={s['n_windows']}"
              + (f"  unpaced/this {ratio[rid]:.2f}" if rid in ratio else ""))

    OUT.write_text(json.dumps({
        "generated_by": "scripts/limits_ac_rms.py", "spec_task": "13.2",
        "preregistered": False,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repo_commit": git("rev-parse", "HEAD"),
        "corpus_commit": corpus.CORPUS_COMMIT,
        "windows": {s: {k: headers[s][k] for k in ("window_s", "hop_s", "repo_commit")}
                    for s in SPLITS},
        "splits_read": list(SPLITS),
        "quantity": "ac_rms_a: AC RMS current (A) over each window; the probe has no DC response",
        "notes": ["exploratory, descriptive (spec 13.2); a disclosure for the Limits",
                  "the unpaced run is scored on its training windows; decode on all windows",
                  "runs come from different sessions; probe gain may not be comparable "
                  "across sessions, and re-clamping between sessions is unchecked",
                  "the calibration decode run is listed as context; the spec's comparison "
                  "is development decode"],
        "unpaced_run": UNPACED, "development_decode_runs": list(DEV_DECODE),
        "calibration_decode_runs": list(CAL_DECODE),
        "runs": runs,
        "ratio_unpaced_over_decode": ratio,
        "ratio_unpaced_over_dev_decode_range": [min(ratio[r] for r in DEV_DECODE),
                                                max(ratio[r] for r in DEV_DECODE)],
    }, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
